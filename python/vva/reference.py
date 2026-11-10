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
    if groups == x.shape[0] == out_c and ig == 1:
        for ky in range(kh):
            for kx in range(kw):
                patch = xp[:, ky * dh: ky * dh + sh * (ho - 1) + 1: sh, kx * dw: kx * dw + sw * (wo - 1) + 1: sw]
                out += w[:, 0, ky, kx][:, None, None] * patch
        if b is not None:
            out += np.asarray(b, dtype=np.float64)[:, None, None]
        return out
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

def hardswish(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0

def hardsigmoid(x):
    return np.clip(x + 3.0, 0.0, 6.0) / 6.0

def resize_nearest(x, sh, sw):
    return np.repeat(np.repeat(x, sh, axis=1), sw, axis=2)

def pixel_shuffle(x, r):
    c, h, w = x.shape
    return x.reshape(c // (r * r), r, r, h, w).transpose(0, 3, 1, 4, 2).reshape(c // (r * r), h * r, w * r)

def pixel_unshuffle(x, r):
    c, h, w = x.shape
    return x.reshape(c, h // r, r, w // r, r).transpose(0, 2, 4, 1, 3).reshape(c * r * r, h // r, w // r)

def conv_transpose2d(x, w, b=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0)):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    in_c, out_c, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    oph, opw = output_padding
    _, h, wd = x.shape
    ho = (h - 1) * sh - 2 * ph + kh + oph
    wo = (wd - 1) * sw - 2 * pw + kw + opw
    full = np.zeros((out_c, (h - 1) * sh + kh + oph, (wd - 1) * sw + kw + opw), dtype=np.float64)
    for iy in range(h):
        for ix in range(wd):
            full[:, iy * sh: iy * sh + kh, ix * sw: ix * sw + kw] += np.tensordot(x[:, iy, ix], w, axes=([0], [0]))
    out = full[:, ph: ph + ho, pw: pw + wo]
    if b is not None:
        out = out + np.asarray(b, dtype=np.float64)[:, None, None]
    return out

def pool2d(x, kernel, stride, padding, kind, count_include_pad=False):
    x = np.asarray(x, dtype=np.float64)
    c, h, w = x.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (w + 2 * pw - kw) // sw + 1
    out = np.zeros((c, ho, wo))
    for oy in range(ho):
        for ox in range(wo):
            ys, xs = oy * sh - ph, ox * sw - pw
            win = x[:, max(ys, 0): min(ys + kh, h), max(xs, 0): min(xs + kw, w)]
            if kind == "max":
                out[:, oy, ox] = win.max(axis=(1, 2))
            else:
                denom = kh * kw if count_include_pad else win.shape[1] * win.shape[2]
                out[:, oy, ox] = win.sum(axis=(1, 2)) / denom
    return out

def resize_bilinear(x, sh, sw, align_corners=False):
    x = np.asarray(x, dtype=np.float64)
    c, h, w = x.shape
    ho, wo = h * sh, w * sw

    def axis(n_in, n_out, scale):
        o = np.arange(n_out, dtype=np.float64)
        if align_corners:
            src = o * (n_in - 1) / (n_out - 1) if n_out > 1 else np.zeros(n_out)
        else:
            src = np.maximum((o + 0.5) / scale - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(int), n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, src - i0

    y0, y1, fy = axis(h, ho, sh)
    x0, x1, fx = axis(w, wo, sw)
    top = x[:, y0][:, :, x0] * (1 - fx) + x[:, y0][:, :, x1] * fx
    bot = x[:, y1][:, :, x0] * (1 - fx) + x[:, y1][:, :, x1] * fx
    return top * (1 - fy)[None, :, None] + bot * fy[None, :, None]

def pad2d(x, pads, mode="constant", value=0.0):
    t, l, b, r = pads
    x = np.asarray(x, dtype=np.float64)
    spec = ((0, 0), (t, b), (l, r))
    if mode == "constant":
        return np.pad(x, spec, mode="constant", constant_values=value)
    return np.pad(x, spec, mode={"reflect": "reflect", "replicate": "edge"}[mode])

def run_graph(meta, tensors, x):
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
        elif op == "Hardswish":
            out = hardswish(a[0])
        elif op == "Hardsigmoid":
            out = hardsigmoid(a[0])
        elif op == "GlobalAveragePool":
            out = a[0].mean(axis=(1, 2), keepdims=True)
        elif op == "Clip":
            out = np.clip(a[0], attrs.get("min", -np.inf), attrs.get("max", np.inf))
        elif op == "Concat":
            out = np.concatenate(a, axis=attrs.get("axis", 1) - 1)
        elif op == "ResizeNearest":
            out = resize_nearest(a[0], *attrs["scale"])
        elif op == "PixelShuffle":
            out = pixel_shuffle(a[0], attrs["upscale"])
        elif op == "PixelUnshuffle":
            out = pixel_unshuffle(a[0], attrs["downscale"])
        elif op == "Sub":
            out = a[0] - a[1]
        elif op == "Tanh":
            out = np.tanh(a[0])
        elif op == "SiLU":
            out = a[0] / (1.0 + np.exp(-a[0]))
        elif op == "PReLU":
            slope = np.asarray(a[1], dtype=np.float64).reshape(-1)
            slope = slope.reshape(-1, 1, 1) if slope.size > 1 else slope.reshape(1, 1, 1)
            out = np.where(a[0] > 0, a[0], a[0] * slope)
        elif op == "ChannelAffine":
            out = a[0] * np.asarray(a[1], dtype=np.float64).reshape(-1, 1, 1) + np.asarray(a[2], dtype=np.float64).reshape(-1, 1, 1)
        elif op == "ConvTranspose2D":
            out = conv_transpose2d(a[0], a[1], a[2] if len(a) > 2 else None, tuple(attrs.get("stride", (1, 1))),
                                   tuple(attrs.get("padding", (0, 0))), tuple(attrs.get("output_padding", (0, 0))))
        elif op in ("MaxPool2D", "AvgPool2D"):
            kernel = tuple(attrs["kernel"])
            out = pool2d(a[0], kernel, tuple(attrs.get("stride", kernel)), tuple(attrs.get("padding", (0, 0))),
                         "max" if op == "MaxPool2D" else "avg", bool(attrs.get("count_include_pad", False)))
        elif op == "ResizeBilinear":
            out = resize_bilinear(a[0], *attrs["scale"], align_corners=bool(attrs.get("align_corners", False)))
        elif op == "Pad":
            out = pad2d(a[0], attrs["pads"], attrs.get("mode", "constant"), attrs.get("value", 0.0))
        elif op == "Slice":
            out = a[0][attrs["start"]:attrs["end"]]
        else:
            raise ValueError("reference: unsupported operator " + op)
        values[node["outputs"][0]] = np.asarray(out, dtype=np.float64)
    return values[meta["outputs"][0]["name"]]

def rrdbnet_forward(sd, x, lrelu_slope=0.2, res_scale=0.2):
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
    x = np.asarray(x, dtype=np.float64)
    factor = {3: 1, 12: 2, 48: 4}[sd["conv_first.weight"].shape[1]]
    if factor > 1:
        x = pixel_unshuffle(x, factor)
    feat = conv("conv_first", x)
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

MBV3_SMALL = [
    (16, 3, 16, 16, True, "RE", 2), (16, 3, 72, 24, False, "RE", 2), (24, 3, 88, 24, False, "RE", 1),
    (24, 5, 96, 40, True, "HS", 2), (40, 5, 240, 40, True, "HS", 1), (40, 5, 240, 40, True, "HS", 1),
    (40, 5, 120, 48, True, "HS", 1), (48, 5, 144, 48, True, "HS", 1), (48, 5, 288, 96, True, "HS", 2),
    (96, 5, 576, 96, True, "HS", 1), (96, 5, 576, 96, True, "HS", 1),
]

def mobilenetv3_small_forward(sd, x, bn_eps=1e-3):
    def conv_bn(conv, bn, t, stride, pad, groups, act):
        t = conv2d(t, sd[conv + ".weight"], None, (stride, stride), (pad, pad), (1, 1), groups)
        g = np.asarray(sd[bn + ".weight"], dtype=np.float64)
        beta = np.asarray(sd[bn + ".bias"], dtype=np.float64)
        mean = np.asarray(sd[bn + ".running_mean"], dtype=np.float64)
        var = np.asarray(sd[bn + ".running_var"], dtype=np.float64)
        t = (t - mean[:, None, None]) / np.sqrt(var + bn_eps)[:, None, None] * g[:, None, None] + beta[:, None, None]
        if act == "HS":
            t = hardswish(t)
        elif act == "RE":
            t = np.maximum(t, 0.0)
        return t

    t = conv_bn("features.0.0", "features.0.1", np.asarray(x, dtype=np.float64), 2, 1, 1, "HS")
    for i, (cin, k, exp, cout, se, act, stride) in enumerate(MBV3_SMALL, start=1):
        p = "features.%d.block" % i
        j = 0
        inp = t
        if exp != cin:
            t = conv_bn("%s.%d.0" % (p, j), "%s.%d.1" % (p, j), t, 1, 0, 1, act)
            j += 1
        t = conv_bn("%s.%d.0" % (p, j), "%s.%d.1" % (p, j), t, stride, k // 2, exp, act)
        j += 1
        if se:
            s = t.mean(axis=(1, 2), keepdims=True)
            s = np.maximum(conv2d(s, sd["%s.%d.fc1.weight" % (p, j)], sd["%s.%d.fc1.bias" % (p, j)]), 0.0)
            s = hardsigmoid(conv2d(s, sd["%s.%d.fc2.weight" % (p, j)], sd["%s.%d.fc2.bias" % (p, j)]))
            t = t * s
            j += 1
        t = conv_bn("%s.%d.0" % (p, j), "%s.%d.1" % (p, j), t, 1, 0, 1, None)
        if stride == 1 and cin == cout:
            t = inp + t
    t = conv_bn("features.12.0", "features.12.1", t, 1, 0, 1, "HS")
    v = t.mean(axis=(1, 2))
    v = hardswish(np.asarray(sd["classifier.0.weight"], dtype=np.float64) @ v + np.asarray(sd["classifier.0.bias"], dtype=np.float64))
    return np.asarray(sd["classifier.3.weight"], dtype=np.float64) @ v + np.asarray(sd["classifier.3.bias"], dtype=np.float64)
