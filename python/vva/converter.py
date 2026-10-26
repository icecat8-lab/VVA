"""PyTorch (.pth) / NumPy (.npz) RRDBNet checkpoint -> .vva (version 2).

Supported network: the x4 ``RRDBNet`` of basicsr / Real-ESRGAN (and models trained
with that code), i.e. keys such as ``conv_first``, ``body.N.rdb{1,2,3}.conv{1..5}``,
``conv_body``, ``conv_up1``, ``conv_up2``, ``conv_hr`` and ``conv_last``.
The converter writes the complete graph (not just weights), so the engine can run
the file without any further information.
"""

import os

import numpy as np

from .fileformat import VVAFormatError, VVAWriter, read_vva
from .reference import rrdbnet_forward, run_graph

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


def convert(input_path, output_path, fp16=True, name=None, author="", license="", lrelu_slope=0.2,
            res_scale=0.2, verify=True, allow_pickle=False, log=print):
    sd = load_checkpoint(input_path, allow_pickle=allow_pickle)
    name = name or os.path.splitext(os.path.basename(output_path))[0]
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


# Backwards compatible name used by older scripts.
class VVAExporter:
    def __init__(self, pth_path, allow_pickle=False):
        self.pth_path = pth_path
        self.allow_pickle = allow_pickle

    def export(self, output_path, **kwargs):
        return convert(self.pth_path, output_path, allow_pickle=self.allow_pickle, **kwargs)
