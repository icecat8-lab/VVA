"""Pure-numpy reference implementations used to verify conversions and the engine.

Two independent pieces live here:

* ``run_graph``       executes a ``.vva`` graph (operators from docs/FORMAT.md).
* ``rrdbnet_forward`` runs an RRDBNet straight from its state dict, written the
                      way the PyTorch model is written, without using any graph.

Comparing the two proves that a converted graph computes the same function as the
original network; comparing either with the C++ engine checks the engine.
Everything is computed in float64 on arrays shaped [C, H, W] (batch 1).
"""

import numpy as np


def conv2d(x, w, b=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    out_c, ig, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    _, h, wd = x.shape
    ho = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((out_c, ho, wo), dtype=np.float64)
    og = out_c // groups
    for g in range(groups):
        xs = xp[g * ig:(g + 1) * ig]
        ws = w[g * og:(g + 1) * og]
        for ky in range(kh):
            for kx in range(kw):
                patch = xs[:, ky * dh: ky * dh + sh * (ho - 1) + 1: sh,
                           kx * dw: kx * dw + sw * (wo - 1) + 1: sw]
                out[g * og:(g + 1) * og] += np.tensordot(ws[:, :, ky, kx], patch, axes=([1], [0]))
    if b is not None:
        out += np.asarray(b, dtype=np.float64)[:, None, None]
    return out


def leaky_relu(x, alpha):
    return np.where(x > 0, x, x * alpha)


def resize_nearest(x, sh, sw):
    return np.repeat(np.repeat(x, sh, axis=1), sw, axis=2)


def pixel_shuffle(x, r):
    c, h, w = x.shape
    return x.reshape(c // (r * r), r, r, h, w).transpose(0, 3, 1, 4, 2).reshape(c // (r * r), h * r, w * r)


def run_graph(meta, tensors, x):
    """Execute the graph described by ``meta`` (as returned by read_vva) on x [C,H,W]."""
    values = dict(tensors)
    values[meta["inputs"][0]["name"]] = np.asarray(x, dtype=np.float64)
    for node in meta["nodes"]:
        op, ins, attrs = node["op"], node["inputs"], node.get("attrs", {})
        a = [values[n] for n in ins]
        if op == "Conv2D":
            out = conv2d(a[0], a[1], a[2] if len(a) > 2 else None,
                         tuple(attrs.get("stride", (1, 1))), tuple(attrs.get("padding", (0, 0))),
                         tuple(attrs.get("dilation", (1, 1))), attrs.get("groups", 1))
        elif op == "Add":
            out = a[0] + a[1]
        elif op == "Mul":
            out = a[0] * a[1] if len(a) == 2 else a[0] * float(attrs["scalar"])
        elif op == "LeakyReLU":
            out = leaky_relu(a[0], float(attrs.get("alpha", 0.01)))
        elif op == "ReLU":
            out = np.maximum(a[0], 0.0)
        elif op == "Sigmoid":
            out = 1.0 / (1.0 + np.exp(-a[0]))
        elif op == "Clip":
            out = np.clip(a[0], attrs.get("min", -np.inf), attrs.get("max", np.inf))
        elif op == "Concat":
            out = np.concatenate(a, axis=attrs.get("axis", 1) - 1)
        elif op == "ResizeNearest":
            out = resize_nearest(a[0], *attrs["scale"])
        elif op == "PixelShuffle":
            out = pixel_shuffle(a[0], attrs["upscale"])
        else:
            raise ValueError("reference: unsupported operator " + op)
        values[node["outputs"][0]] = np.asarray(out, dtype=np.float64)
    return values[meta["outputs"][0]["name"]]


def rrdbnet_forward(sd, x, lrelu_slope=0.2, res_scale=0.2):
    """RRDBNet (x4, basicsr / Real-ESRGAN layout) computed directly from a state dict."""
    def conv(prefix, t):
        return conv2d(t, sd[prefix + ".weight"], sd[prefix + ".bias"], padding=(1, 1))

    def act(t):
        return leaky_relu(t, lrelu_slope)

    def rdb(p, t):
        x1 = act(conv(p + ".conv1", t))
        x2 = act(conv(p + ".conv2", np.concatenate([t, x1], axis=0)))
        x3 = act(conv(p + ".conv3", np.concatenate([t, x1, x2], axis=0)))
        x4 = act(conv(p + ".conv4", np.concatenate([t, x1, x2, x3], axis=0)))
        x5 = conv(p + ".conv5", np.concatenate([t, x1, x2, x3, x4], axis=0))
        return x5 * res_scale + t

    nb = 0
    while "body.%d.rdb1.conv1.weight" % nb in sd:
        nb += 1
    feat = conv("conv_first", np.asarray(x, dtype=np.float64))
    t = feat
    for i in range(nb):
        o = rdb("body.%d.rdb1" % i, t)
        o = rdb("body.%d.rdb2" % i, o)
        o = rdb("body.%d.rdb3" % i, o)
        t = o * res_scale + t
    feat = feat + conv("conv_body", t)
    feat = act(conv("conv_up1", resize_nearest(feat, 2, 2)))
    feat = act(conv("conv_up2", resize_nearest(feat, 2, 2)))
    return conv("conv_last", act(conv("conv_hr", feat)))
