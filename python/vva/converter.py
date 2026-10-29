"""PyTorch (.pth) / NumPy (.npz) checkpoint -> .vva (version 2).

Supported networks
* the x4 ``RRDBNet`` of basicsr / Real-ESRGAN (and models trained with that code): keys such as
  ``conv_first``, ``body.N.rdb{1,2,3}.conv{1..5}``, ``conv_body``, ``conv_up1``, ``conv_up2``,
  ``conv_hr`` and ``conv_last``;
* the torchvision ``MobileNetV3-Small`` image classifier (any number of classes): keys such as
  ``features.N.block.M...`` and ``classifier.0`` / ``classifier.3``. BatchNorm is folded into the
  convolutions.

The converter writes the complete graph (not just weights), so the engine can run the file without any
further information.
"""

import os

import numpy as np

from .fileformat import VVAFormatError, VVAWriter, read_vva
from .reference import MBV3_SMALL, mobilenetv3_small_forward, rrdbnet_forward, run_graph

_FP16_MAX = 65000.0
_WRAPPER_KEYS = ("params_ema", "params", "state_dict", "model", "net_g", "g_ema")


class ConversionError(Exception):
    """The checkpoint cannot be converted (message is meant for the user)."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _to_numpy(value):
    if hasattr(value, "detach"):  # torch.Tensor
        return value.detach().cpu().float().numpy()
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return None


def _unwrap(ckpt):
    if not isinstance(ckpt, dict):
        raise ConversionError("the checkpoint is not a dictionary of weights")
    for key in _WRAPPER_KEYS:
        if isinstance(ckpt.get(key), dict):
            ckpt = ckpt[key]
            break
    state = {}
    for name, value in ckpt.items():
        if name.endswith("num_batches_tracked"):
            continue
        arr = _to_numpy(value)
        if arr is None:
            continue
        if name.startswith("module."):
            name = name[len("module."):]
        state[name] = arr
    if not state:
        raise ConversionError("no weight tensors found in the checkpoint")
    return state


def load_checkpoint(path, allow_pickle=False):
    """Return {name: float32 ndarray} from a .pth/.pt (needs torch) or .npz file."""
    if not os.path.exists(path):
        raise ConversionError("input file not found: %s" % path)
    if os.path.splitext(path)[1].lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return _unwrap({k: z[k] for k in z.files})
    try:
        import torch
    except ImportError:
        raise ConversionError("reading .pth files needs PyTorch (pip install torch); "
                              "alternatively save the weights with numpy.savez and pass the .npz file")
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=not allow_pickle)
        except TypeError:  # torch < 1.13 has no weights_only
            ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ConversionError(
            "could not read the checkpoint safely (%s). If you trust this file, retry with --allow-pickle." % exc)
    return _unwrap(ckpt)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _need(sd, key, shape_check=None):
    if key not in sd:
        raise ConversionError("missing weight '%s' - this does not look like a basicsr RRDBNet checkpoint" % key)
    arr = sd[key]
    if shape_check is not None and not shape_check(arr.shape):
        raise ConversionError("weight '%s' has unexpected shape %s" % (key, tuple(arr.shape)))
    return arr


def _conv_shape(out_c, in_c):
    return lambda s: len(s) == 4 and s[0] == out_c and s[1] == in_c and s[2] == s[3] and s[2] % 2 == 1


def inspect_rrdbnet(sd):
    """Validate the layout and return (nf, gc, nb)."""
    first = _need(sd, "conv_first.weight", lambda s: len(s) == 4 and s[2] == s[3] and s[2] % 2 == 1)
    nf, in_ch = first.shape[0], first.shape[1]
    if in_ch != 3:
        raise ConversionError("conv_first expects %d input channels; only RGB x4 models (3 channels, no "
                              "pixel-unshuffle) are supported" % in_ch)
    nb = 0
    while "body.%d.rdb1.conv1.weight" % nb in sd:
        nb += 1
    if nb == 0:
        raise ConversionError("no 'body.N.rdb1.conv1.weight' found - not an RRDBNet checkpoint")
    gc = sd["body.0.rdb1.conv1.weight"].shape[0]
    for i in range(nb):
        for r in (1, 2, 3):
            for j in range(1, 6):
                key = "body.%d.rdb%d.conv%d" % (i, r, j)
                out_c = gc if j < 5 else nf
                _need(sd, key + ".weight", _conv_shape(out_c, nf + (j - 1) * gc))
                _need(sd, key + ".bias", lambda s, o=out_c: s == (o,))
    for key, in_c, out_c in (("conv_body", nf, nf), ("conv_up1", nf, nf), ("conv_up2", nf, nf),
                             ("conv_hr", nf, nf), ("conv_last", nf, 3)):
        _need(sd, key + ".weight", _conv_shape(out_c, in_c))
        _need(sd, key + ".bias", lambda s, o=out_c: s == (o,))
    _need(sd, "conv_first.bias", lambda s: s == (nf,))
    return nf, gc, nb


def build_rrdbnet(sd, name="model", fp16=True, lrelu_slope=0.2, res_scale=0.2, author="", license=""):
    """Return a VVAWriter holding the complete RRDBNet graph and its weights."""
    nf, gc, nb = inspect_rrdbnet(sd)
    writer = VVAWriter(name=name, scale=4, author=author, license=license,
                       description="RRDBNet x4 (nf=%d, gc=%d, nb=%d)" % (nf, gc, nb))
    writer.add_input("input", [1, 3, -1, -1])
    used = set()

    def store(key, as_weight):
        arr = sd[key]
        if not np.all(np.isfinite(arr)):
            raise ConversionError("weight '%s' contains NaN or Inf" % key)
        dtype = "f16" if (fp16 and as_weight and float(np.max(np.abs(arr))) <= _FP16_MAX) else "f32"
        writer.add_tensor(key, arr, dtype)
        used.add(key)
        return key

    counter = [0]

    def uid(base):
        counter[0] += 1
        return "%s#%d" % (base, counter[0])

    def conv(prefix, x, out=None):
        k = sd[prefix + ".weight"].shape[2]
        out = out or uid(prefix)
        writer.add_node("Conv2D", [x, store(prefix + ".weight", True), store(prefix + ".bias", False)], [out],
                        {"stride": [1, 1], "padding": [k // 2, k // 2], "dilation": [1, 1], "groups": 1},
                        name=prefix)
        return out

    def lrelu(x, base):
        out = uid(base + ".lrelu")
        writer.add_node("LeakyReLU", [x], [out], {"alpha": float(lrelu_slope)})
        return out

    def scale(x, base):
        out = uid(base + ".scale")
        writer.add_node("Mul", [x], [out], {"scalar": float(res_scale)})
        return out

    def add(a, b, base):
        out = uid(base + ".add")
        writer.add_node("Add", [a, b], [out])
        return out

    def concat(items, base):
        out = uid(base + ".cat")
        writer.add_node("Concat", items, [out], {"axis": 1})
        return out

    def rdb(p, x):
        x1 = lrelu(conv(p + ".conv1", x), p + ".x1")
        x2 = lrelu(conv(p + ".conv2", concat([x, x1], p + ".c2")), p + ".x2")
        x3 = lrelu(conv(p + ".conv3", concat([x, x1, x2], p + ".c3")), p + ".x3")
        x4 = lrelu(conv(p + ".conv4", concat([x, x1, x2, x3], p + ".c4")), p + ".x4")
        x5 = conv(p + ".conv5", concat([x, x1, x2, x3, x4], p + ".c5"))
        return add(scale(x5, p), x, p)

    feat0 = conv("conv_first", "input")
    t = feat0
    for i in range(nb):
        o = rdb("body.%d.rdb1" % i, t)
        o = rdb("body.%d.rdb2" % i, o)
        o = rdb("body.%d.rdb3" % i, o)
        t = add(scale(o, "body.%d" % i), t, "body.%d" % i)
    feat = add(feat0, conv("conv_body", t), "trunk")
    for up in ("conv_up1", "conv_up2"):
        r = uid(up + ".resize")
        writer.add_node("ResizeNearest", [feat], [r], {"scale": [2, 2]})
        feat = lrelu(conv(up, r), up)
    hr = lrelu(conv("conv_hr", feat), "conv_hr")
    conv("conv_last", hr, out="output")
    writer.add_output("output", [1, 3, -1, -1])

    unused = sorted(k for k in sd if k not in used)
    writer.unused_keys = unused
    return writer


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_file(path, sd, lrelu_slope=0.2, res_scale=0.2, size=(10, 13), seed=0):
    """Re-read the written file and check it against the original network.

    Returns (graph_error, quantisation_error), both relative to the output range:
    the first compares the stored graph with the same (dequantised) weights run
    through the plain network and must be ~1e-12; the second is the effect of fp16
    weight storage relative to the original weights.
    """
    meta, stored = read_vva(path)
    x = np.random.RandomState(seed).rand(3, size[0], size[1])
    got = run_graph(meta, stored, x)
    deq = {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()}
    want_same_weights = rrdbnet_forward(deq, x, lrelu_slope, res_scale)
    want_original = rrdbnet_forward({k: np.asarray(v, dtype=np.float64) for k, v in sd.items()}, x,
                                    lrelu_slope, res_scale)
    span = max(float(np.max(np.abs(want_original))), 1e-12)
    return (float(np.max(np.abs(got - want_same_weights))) / span,
            float(np.max(np.abs(got - want_original))) / span)


# ---------------------------------------------------------------------------
# MobileNetV3-Small classifier
# ---------------------------------------------------------------------------

def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _need_bn(sd, prefix, channels):
    for suffix in (".weight", ".bias", ".running_mean", ".running_var"):
        _need(sd, prefix + suffix, lambda s, c=channels: s == (c,))


def inspect_mobilenetv3_small(sd):
    """Validate the layout against torchvision's MobileNetV3-Small; returns the number of classes."""
    _need(sd, "features.0.0.weight", lambda s: s == (16, 3, 3, 3))
    _need_bn(sd, "features.0.1", 16)
    for i, (cin, k, exp, cout, se, _act, _stride) in enumerate(MBV3_SMALL, start=1):
        p = "features.%d.block" % i
        j = 0
        if exp != cin:
            _need(sd, "%s.%d.0.weight" % (p, j), lambda s, a=exp, b=cin: s == (a, b, 1, 1))
            _need_bn(sd, "%s.%d.1" % (p, j), exp)
            j += 1
        _need(sd, "%s.%d.0.weight" % (p, j), lambda s, a=exp, kk=k: s == (a, 1, kk, kk))
        _need_bn(sd, "%s.%d.1" % (p, j), exp)
        j += 1
        if se:
            sq = _make_divisible(exp // 4, 8)
            _need(sd, "%s.%d.fc1.weight" % (p, j), lambda s, a=sq, b=exp: s == (a, b, 1, 1))
            _need(sd, "%s.%d.fc1.bias" % (p, j), lambda s, a=sq: s == (a,))
            _need(sd, "%s.%d.fc2.weight" % (p, j), lambda s, a=exp, b=sq: s == (a, b, 1, 1))
            _need(sd, "%s.%d.fc2.bias" % (p, j), lambda s, a=exp: s == (a,))
            j += 1
        _need(sd, "%s.%d.0.weight" % (p, j), lambda s, a=cout, b=exp: s == (a, b, 1, 1))
        _need_bn(sd, "%s.%d.1" % (p, j), cout)
    _need(sd, "features.12.0.weight", lambda s: s == (576, 96, 1, 1))
    _need_bn(sd, "features.12.1", 576)
    _need(sd, "classifier.0.weight", lambda s: s == (1024, 576))
    _need(sd, "classifier.0.bias", lambda s: s == (1024,))
    last = _need(sd, "classifier.3.weight", lambda s: len(s) == 2 and s[1] == 1024 and s[0] >= 1)
    _need(sd, "classifier.3.bias", lambda s, n=last.shape[0]: s == (n,))
    return int(last.shape[0])


def build_mobilenetv3_small(sd, name="classifier", classes=None, input_size=224, mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225), fp16=True, bn_eps=1e-3, author="", license=""):
    """Return a VVAWriter holding the MobileNetV3-Small classifier graph (BatchNorm folded in)."""
    n_classes = inspect_mobilenetv3_small(sd)
    if classes is not None and len(classes) != n_classes:
        raise ConversionError("the checkpoint has %d classes but %d class names were given" % (n_classes, len(classes)))
    writer = VVAWriter(name=name, scale=1, author=author, license=license,
                       description="MobileNetV3-Small classifier (%d classes)" % n_classes)
    writer.meta_info["task"] = "classify"
    writer.meta_info["input_size"] = int(input_size)
    writer.meta_info["mean"] = [float(v) for v in mean]
    writer.meta_info["std"] = [float(v) for v in std]
    if classes:
        writer.meta_info["classes"] = list(classes)
    writer.add_input("input", [1, 3, -1, -1])

    counter = [0]

    def uid(base):
        counter[0] += 1
        return "%s#%d" % (base, counter[0])

    def put(key, arr, as_weight):
        arr = np.asarray(arr, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ConversionError("weight '%s' contains NaN or Inf" % key)
        dtype = "f16" if (fp16 and as_weight and float(np.max(np.abs(arr))) <= _FP16_MAX) else "f32"
        writer.add_tensor(key, arr, dtype)
        return key

    def conv(prefix, w, b, x, stride=1, pad=0, groups=1, out=None):
        out = out or uid(prefix)
        writer.add_node("Conv2D", [x, put(prefix + ".weight", w, True), put(prefix + ".bias", b, False)], [out],
                        {"stride": [stride, stride], "padding": [pad, pad], "dilation": [1, 1], "groups": groups},
                        name=prefix)
        return out

    def conv_bn(cname, bname, x, stride, pad, groups):
        w = np.asarray(sd[cname + ".weight"], dtype=np.float64)
        g = np.asarray(sd[bname + ".weight"], dtype=np.float64)
        beta = np.asarray(sd[bname + ".bias"], dtype=np.float64)
        mean_ = np.asarray(sd[bname + ".running_mean"], dtype=np.float64)
        var = np.asarray(sd[bname + ".running_var"], dtype=np.float64)
        k = g / np.sqrt(var + bn_eps)
        bias = beta - mean_ * k
        if cname + ".bias" in sd:
            bias = bias + np.asarray(sd[cname + ".bias"], dtype=np.float64) * k
        return conv(cname, w * k[:, None, None, None], bias, x, stride, pad, groups)

    def act(kind, x, base):
        out = uid(base + (".hswish" if kind == "HS" else ".relu"))
        writer.add_node("Hardswish" if kind == "HS" else "ReLU", [x], [out])
        return out

    x = act("HS", conv_bn("features.0.0", "features.0.1", "input", 2, 1, 1), "features.0")
    for i, (cin, k, exp, cout, se, a, stride) in enumerate(MBV3_SMALL, start=1):
        p = "features.%d.block" % i
        j = 0
        block_in = x
        if exp != cin:
            x = act(a, conv_bn("%s.%d.0" % (p, j), "%s.%d.1" % (p, j), x, 1, 0, 1), "%s.%d" % (p, j))
            j += 1
        x = act(a, conv_bn("%s.%d.0" % (p, j), "%s.%d.1" % (p, j), x, stride, k // 2, exp), "%s.%d" % (p, j))
        j += 1
        if se:
            q = "%s.%d" % (p, j)
            pooled = uid(q + ".pool")
            writer.add_node("GlobalAveragePool", [x], [pooled])
            s1 = act("RE", conv(q + ".fc1", sd[q + ".fc1.weight"], sd[q + ".fc1.bias"], pooled), q + ".fc1")
            s2 = conv(q + ".fc2", sd[q + ".fc2.weight"], sd[q + ".fc2.bias"], s1)
            gate = uid(q + ".gate")
            writer.add_node("Hardsigmoid", [s2], [gate])
            scaled = uid(q + ".scaled")
            writer.add_node("Mul", [x, gate], [scaled])
            x = scaled
            j += 1
        x = conv_bn("%s.%d.0" % (p, j), "%s.%d.1" % (p, j), x, 1, 0, 1)
        if stride == 1 and cin == cout:
            summed = uid(p + ".add")
            writer.add_node("Add", [block_in, x], [summed])
            x = summed
    x = act("HS", conv_bn("features.12.0", "features.12.1", x, 1, 0, 1), "features.12")
    pooled = uid("avgpool")
    writer.add_node("GlobalAveragePool", [x], [pooled])
    w0 = np.asarray(sd["classifier.0.weight"], dtype=np.float32)
    h = act("HS", conv("classifier.0", w0.reshape(w0.shape[0], w0.shape[1], 1, 1), sd["classifier.0.bias"], pooled),
            "classifier.0")
    w3 = np.asarray(sd["classifier.3.weight"], dtype=np.float32)
    conv("classifier.3", w3.reshape(w3.shape[0], w3.shape[1], 1, 1), sd["classifier.3.bias"], h, out="output")
    writer.add_output("output", [1, n_classes, -1, -1])
    return writer


def verify_mobilenetv3(path, sd, bn_eps=1e-3, size=64, seed=0):
    """Returns (graph_error, file_error): the exact fp32 graph against the plain network, and the written
    file (with its fp16 weights) against the plain network. Errors are relative to max(|logit|, 1)."""
    x = np.random.RandomState(seed).randn(3, size, size)
    want = mobilenetv3_small_forward({k: np.asarray(v, dtype=np.float64) for k, v in sd.items()}, x, bn_eps)
    span = max(float(np.max(np.abs(want))), 1.0)
    exact = build_mobilenetv3_small(sd, fp16=False, bn_eps=bn_eps)
    meta, tensors = read_vva(exact.to_bytes())
    graph_err = float(np.max(np.abs(run_graph(meta, tensors, x).reshape(-1) - want))) / span
    meta, stored = read_vva(path)
    file_err = float(np.max(np.abs(run_graph(meta, stored, x).reshape(-1) - want))) / span
    return graph_err, file_err


def detect_architecture(sd):
    if "conv_first.weight" in sd and any(k.startswith("body.") for k in sd):
        return "rrdbnet"
    if "features.0.0.weight" in sd and "classifier.3.weight" in sd:
        return "mobilenet_v3_small"
    raise ConversionError("unrecognised checkpoint: supported networks are the x4 RRDBNet (basicsr / Real-ESRGAN "
                          "keys: conv_first, body.N.rdb1...) and torchvision MobileNetV3-Small classifiers "
                          "(keys: features.N.block..., classifier.3)")


def convert(input_path, output_path, fp16=True, name=None, author="", license="", lrelu_slope=0.2,
            res_scale=0.2, verify=True, allow_pickle=False, log=print, classes=None, input_size=224,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), bn_eps=1e-3):
    sd = load_checkpoint(input_path, allow_pickle=allow_pickle)
    name = name or os.path.splitext(os.path.basename(output_path))[0]
    if detect_architecture(sd) == "mobilenet_v3_small":
        return _convert_classifier(sd, output_path, fp16, name, author, license, verify, log, classes, input_size,
                                   mean, std, bn_eps)
    writer = build_rrdbnet(sd, name, fp16, lrelu_slope, res_scale, author, license)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    try:
        size = writer.save(output_path)
    except VVAFormatError as exc:
        raise ConversionError(str(exc))
    nf, gc, nb = inspect_rrdbnet(sd)
    log("converted: %s  (x4 RRDBNet nf=%d gc=%d nb=%d, %.1f MB)" % (output_path, nf, gc, nb, size / 1e6))
    if writer.unused_keys:
        log("note: %d checkpoint entries were not used (e.g. %s)" % (len(writer.unused_keys), writer.unused_keys[0]))
    if verify:
        graph_err, quant_err = verify_file(output_path, sd, lrelu_slope, res_scale)
        if graph_err > 1e-6:
            raise ConversionError("verification failed: the stored graph differs from the network (%.2e)" % graph_err)
        log("verified: graph matches the network (%.1e), fp16 storage error %.1e of the output range"
            % (graph_err, quant_err))
    return output_path


def _convert_classifier(sd, output_path, fp16, name, author, license, verify, log, classes, input_size, mean, std, bn_eps):
    writer = build_mobilenetv3_small(sd, name, classes, input_size, mean, std, fp16, bn_eps, author, license)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        size = writer.save(output_path)
    except VVAFormatError as exc:
        raise ConversionError(str(exc))
    n = inspect_mobilenetv3_small(sd)
    log("converted: %s  (MobileNetV3-Small classifier, %d classes, input %dx%d, %.1f MB)"
        % (output_path, n, input_size, input_size, size / 1e6))
    log("preprocessing stored in the file: resize to %dx%d, mean=%s std=%s" % (input_size, input_size, list(mean), list(std)))
    if verify:
        graph_err, file_err = verify_mobilenetv3(output_path, sd, bn_eps)
        if graph_err > 1e-5:
            raise ConversionError("verification failed: the stored graph differs from the network (%.2e)" % graph_err)
        if file_err > 0.05:
            raise ConversionError("verification failed: fp16 weights change the logits too much (%.2e); use --fp32" % file_err)
        log("verified: graph matches the network (%.1e), fp16 storage changes the logits by %.1e" % (graph_err, file_err))
    return output_path


# Backwards compatible name used by older scripts.
class VVAExporter:
    def __init__(self, pth_path, allow_pickle=False):
        self.pth_path = pth_path
        self.allow_pickle = allow_pickle

    def export(self, output_path, **kwargs):
        return convert(self.pth_path, output_path, allow_pickle=self.allow_pickle, **kwargs)
