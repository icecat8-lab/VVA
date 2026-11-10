#!/usr/bin/env python3
import io
import os
import contextlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_builder as ob
from common import Checker, WORK, build_binaries, exec_run

from vva import cli
from vva.converter import ConversionError, convert
from vva.fileformat import read_vva
from vva.reference import conv2d

T = Checker("test_onnx")
rng = np.random.RandomState(11)


def rnd(*shape, scale=0.4):
    return (rng.randn(*shape) * scale).astype(np.float32)


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(args)
    return code, out.getvalue(), err.getvalue()


def onnx_eval(nodes, inits, x):
    env = {"x": np.asarray(x, dtype=np.float64)[None]}
    consts = {k: np.asarray(v, dtype=np.float64) if np.asarray(v).dtype != np.int64 else np.asarray(v) for k, v in inits.items()}
    get = lambda n: env[n] if n in env else consts[n]
    for op, ins, outs, attrs in nodes:
        a = [get(i) if i else None for i in ins]
        at = attrs or {}
        if op == "Conv":
            p = at.get("pads", [0, 0, 0, 0])
            padded = np.pad(a[0][0], ((0, 0), (p[0], p[2]), (p[1], p[3])))
            r = conv2d(padded, a[1], a[2] if len(a) > 2 else None, tuple(at.get("strides", (1, 1))), (0, 0),
                       tuple(at.get("dilations", (1, 1))), at.get("group", 1))[None]
        elif op == "BatchNormalization":
            sh = (1, -1, 1, 1)
            r = (a[0] - a[3].reshape(sh)) / np.sqrt(a[4].reshape(sh) + at.get("epsilon", 1e-5)) * a[1].reshape(sh) + a[2].reshape(sh)
        elif op == "Relu":
            r = np.maximum(a[0], 0)
        elif op == "LeakyRelu":
            r = np.where(a[0] > 0, a[0], a[0] * at.get("alpha", 0.01))
        elif op == "PRelu":
            r = np.where(a[0] > 0, a[0], a[0] * a[1].reshape(1, -1, 1, 1))
        elif op == "Sigmoid":
            r = 1 / (1 + np.exp(-a[0]))
        elif op == "Tanh":
            r = np.tanh(a[0])
        elif op == "HardSwish":
            r = a[0] * np.clip(a[0] / 6 + 0.5, 0, 1)
        elif op == "Clip":
            r = np.clip(a[0], a[1] if len(a) > 1 and a[1] is not None else -np.inf, a[2] if len(a) > 2 and a[2] is not None else np.inf)
        elif op == "Add":
            r = a[0] + a[1]
        elif op == "Sub":
            r = a[0] - a[1]
        elif op == "Mul":
            r = a[0] * a[1]
        elif op == "Div":
            r = a[0] / a[1]
        elif op == "Concat":
            r = np.concatenate(a, axis=at.get("axis", 1))
        elif op == "Identity":
            r = a[0]
        elif op == "Resize":
            sc = a[2]
            n, c, h, w = a[0].shape
            ho, wo = int(h * sc[2]), int(w * sc[3])
            r = np.zeros((n, c, ho, wo))
            mode, coord = at.get("mode", "nearest"), at.get("coordinate_transformation_mode", "half_pixel")
            for oy in range(ho):
                for ox in range(wo):
                    if mode == "nearest":
                        r[:, :, oy, ox] = a[0][:, :, int(oy // sc[2]), int(ox // sc[3])]
                        continue
                    if coord == "align_corners":
                        sy = oy * (h - 1) / (ho - 1) if ho > 1 else 0.0
                        sx = ox * (w - 1) / (wo - 1) if wo > 1 else 0.0
                    else:
                        sy, sx = (oy + 0.5) / sc[2] - 0.5, (ox + 0.5) / sc[3] - 0.5
                    y0, x0 = int(np.floor(sy)), int(np.floor(sx))
                    fy, fx = sy - y0, sx - x0
                    cl = lambda v, n_: min(max(v, 0), n_ - 1)
                    v = lambda yy, xx: a[0][:, :, cl(yy, h), cl(xx, w)]
                    r[:, :, oy, ox] = (v(y0, x0) * (1 - fx) + v(y0, x0 + 1) * fx) * (1 - fy) + (v(y0 + 1, x0) * (1 - fx) + v(y0 + 1, x0 + 1) * fx) * fy
        elif op == "DepthToSpace":
            b = at["blocksize"]
            n, c, h, w = a[0].shape
            r = a[0].reshape(n, c // (b * b), b, b, h, w).transpose(0, 1, 4, 2, 5, 3).reshape(n, c // (b * b), h * b, w * b)
        elif op in ("MaxPool", "AveragePool"):
            kh, kw = at["kernel_shape"]
            sh, sw = at.get("strides", [1, 1])
            ph, pw = at.get("pads", [0, 0, 0, 0])[:2]
            n, c, h, w = a[0].shape
            ho, wo = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
            r = np.zeros((n, c, ho, wo))
            for oy in range(ho):
                for ox in range(wo):
                    ys, xs = oy * sh - ph, ox * sw - pw
                    win = a[0][:, :, max(ys, 0):min(ys + kh, h), max(xs, 0):min(xs + kw, w)]
                    if op == "MaxPool":
                        r[:, :, oy, ox] = win.max(axis=(2, 3))
                    else:
                        d = kh * kw if at.get("count_include_pad", 0) else win.shape[2] * win.shape[3]
                        r[:, :, oy, ox] = win.sum(axis=(2, 3)) / d
        elif op == "ConvTranspose":
            w = a[1]
            sh, sw = at.get("strides", [1, 1])
            ph, pw = at.get("pads", [0, 0, 0, 0])[:2]
            oph, opw = at.get("output_padding", [0, 0])
            ci, co, kh, kw = w.shape
            _, _, h, wd = a[0].shape
            full = np.zeros((co, (h - 1) * sh + kh + oph, (wd - 1) * sw + kw + opw))
            for iy in range(h):
                for ix in range(wd):
                    for c_ in range(ci):
                        full[:, iy * sh:iy * sh + kh, ix * sw:ix * sw + kw] += a[0][0, c_, iy, ix] * w[c_]
            ho, wo = (h - 1) * sh - 2 * ph + kh + oph, (wd - 1) * sw - 2 * pw + kw + opw
            r = full[:, ph:ph + ho, pw:pw + wo][None]
            if len(a) > 2:
                r = r + a[2].reshape(1, -1, 1, 1)
        elif op == "Pad":
            p = [int(v) for v in a[1]]
            mode = {"constant": "constant", "reflect": "reflect", "edge": "edge"}[at.get("mode", "constant")]
            r = np.pad(a[0], ((0, 0), (0, 0), (p[2], p[6]), (p[3], p[7])), mode=mode)
        elif op == "Slice":
            r = a[0][:, int(a[1][0]):int(a[2][0])]
        elif op == "Split":
            pos = 0
            for o, size in zip(outs, at["split"]):
                env[o] = a[0][:, pos:pos + size]
                pos += size
            continue
        else:
            raise AssertionError(op)
        env[outs[0]] = r
    return env[nodes[-1][2][0]][0]


def write_model(name, nodes, inits, dims=("N", 3, "H", "W"), opset=13, init_mode="raw"):
    spec = [ob.node(op, ins, outs, attrs) for op, ins, outs, attrs in nodes]
    path = os.path.join(WORK, name + ".onnx")
    with open(path, "wb") as f:
        f.write(ob.model(spec, inits, "x", list(dims), nodes[-1][2][0], ["N", "C", "H", "W"], opset, init_mode))
    return path


def bn_params(c):
    return {"scale": (1 + rnd(c, scale=0.2)).astype(np.float32), "bias": rnd(c, scale=0.2),
            "mean": rnd(c, scale=0.3), "var": (0.5 + rng.rand(c)).astype(np.float32)}


def case(name, nodes, inits, exe, scale, sizes, tol=2e-4, init_mode="raw", dims=("N", 3, "H", "W"), expect_ops=None, expect_not_ops=()):
    path = write_model(name, nodes, inits, dims=dims, init_mode=init_mode)
    out = os.path.join(WORK, name + ".vva")
    buf = []
    try:
        convert(path, out, fp16=False, log=buf.append)
    except ConversionError as exc:
        T.check(False, "%s converts (%s)" % (name, exc))
        return
    meta, _ = read_vva(out)
    T.check(meta["meta"]["scale"] == scale, "%s: scale %d detected (got %s)" % (name, scale, meta["meta"]["scale"]))
    ops = {n["op"] for n in meta["nodes"]}
    if expect_ops:
        T.check(expect_ops <= ops, "%s: uses %s (has %s)" % (name, sorted(expect_ops), sorted(ops)))
    for bad in expect_not_ops:
        T.check(bad not in ops, "%s: %s was folded away" % (name, bad))
    for (h, w) in sizes:
        x = np.random.RandomState(h * 13 + w).rand(3, h, w).astype(np.float32)
        want = onnx_eval(nodes, inits, x).reshape(-1)
        text, got = exec_run(exe, out, x, 1)
        T.check(got is not None, "%s %dx%d runs (%s)" % (name, h, w, text))
        if got is None:
            continue
        err = float(np.max(np.abs(got - want)) / max(float(np.max(np.abs(want))), 1e-6)) if got.size == want.size else -1
        T.check(got.size == want.size and err < tol, "%s %dx%d matches an independent ONNX evaluation (%.2e)" % (name, h, w, err))


def main():
    exe = build_binaries(sanitize=os.environ.get("VVA_NO_SAN") is None)["exec_run"]
    sizes = [(8, 12), (6, 10)]

    bn = bn_params(8)
    inits = {"w1": rnd(8, 3, 3, 3), "b1": rnd(8), "w2": rnd(8, 8, 3, 3), "b2": rnd(8), "w3": rnd(12, 8, 3, 3), "b3": rnd(12),
             "bn_s": bn["scale"], "bn_b": bn["bias"], "bn_m": bn["mean"], "bn_v": bn["var"], "two": np.array(2.0, dtype=np.float32)}
    nodes = [
        ("Conv", ["x", "w1", "b1"], ["c1"], {"pads": [1, 1, 1, 1], "kernel_shape": [3, 3]}),
        ("BatchNormalization", ["c1", "bn_s", "bn_b", "bn_m", "bn_v"], ["n1"], {"epsilon": 1e-5}),
        ("Relu", ["n1"], ["r1"], None),
        ("Conv", ["r1", "w2", "b2"], ["c2"], {"pads": [1, 1, 1, 1]}),
        ("Identity", ["c2"], ["c2i"], None),
        ("Add", ["c2i", "r1"], ["s"], None),
        ("Mul", ["s", "two"], ["s2"], None),
        ("Conv", ["s2", "w3", "b3"], ["c3"], {"pads": [1, 1, 1, 1]}),
        ("DepthToSpace", ["c3"], ["y"], {"blocksize": 2, "mode": "CRD"}),
    ]
    case("sr_like", nodes, inits, exe, 2, sizes, expect_ops={"PixelShuffle", "Conv2D"})
    case("sr_like_floatdata", nodes, inits, exe, 2, sizes, init_mode="float_data")

    sr_nodes = read_vva(os.path.join(WORK, "sr_like.vva"))[0]["nodes"]
    T.check(sum(n["op"] == "ChannelAffine" for n in sr_nodes) == 1, "batch norm is folded; only the Mul by a constant remains")

    inits = {"w1": rnd(6, 3, 3, 3), "b1": rnd(6), "slope": rnd(6, 1, 1, scale=0.2), "w2": rnd(6, 6, 3, 3), "b2": rnd(6),
             "w3": rnd(6, 4, 2, 2), "b3": rnd(4), "w4": rnd(3, 4, 3, 3), "bn_s": bn_params(4)["scale"], "bn_b": bn_params(4)["bias"],
             "bn_m": bn_params(4)["mean"], "bn_v": bn_params(4)["var"]}
    nodes = [
        ("Conv", ["x", "w1", "b1"], ["c1"], {"pads": [1, 1, 1, 1]}),
        ("PRelu", ["c1", "slope"], ["p"], None),
        ("LeakyRelu", ["p"], ["l"], {"alpha": 0.1}),
        ("MaxPool", ["l"], ["mp"], {"kernel_shape": [2, 2], "strides": [2, 2]}),
        ("Conv", ["mp", "w2", "b2"], ["c2"], {"pads": [1, 1, 1, 1]}),
        ("HardSwish", ["c2"], ["hs"], None),
        ("ConvTranspose", ["hs", "w3", "b3"], ["up"], {"strides": [2, 2], "kernel_shape": [2, 2]}),
        ("BatchNormalization", ["up", "bn_s", "bn_b", "bn_m", "bn_v"], ["bn"], {"epsilon": 1e-5}),
        ("Sigmoid", ["bn"], ["sg"], None),
        ("Conv", ["sg", "w4"], ["y"], {"pads": [1, 1, 1, 1]}),
    ]
    case("down_up", nodes, inits, exe, 1, [(16, 24), (8, 8)], expect_ops={"MaxPool2D", "ConvTranspose2D", "PReLU", "Hardswish"},
         expect_not_ops={"ChannelAffine"})

    inits = {"w1": rnd(8, 3, 3, 3), "b1": rnd(8), "mul_c": rnd(1, 8, 1, 1, scale=0.5) + 1, "add_c": np.array([0.3], dtype=np.float32),
             "sub_c": rnd(8, 1, 1), "w2": rnd(6, 8, 3, 3), "pads": np.array([0, 0, 1, 2, 0, 0, 1, 0], dtype=np.int64),
             "scales2": np.array([1, 1, 2, 2], dtype=np.float32), "scales3": np.array([1, 1, 3, 3], dtype=np.float32),
             "starts": np.array([1], dtype=np.int64), "ends": np.array([4], dtype=np.int64), "axes": np.array([1], dtype=np.int64),
             "lo": np.array(-0.2, dtype=np.float32), "hi": np.array(0.6, dtype=np.float32), "roi": np.zeros(0, dtype=np.float32)}
    nodes = [
        ("Conv", ["x", "w1", "b1"], ["c1"], {"pads": [1, 1, 1, 1]}),
        ("Mul", ["c1", "mul_c"], ["m"], None),
        ("Add", ["m", "add_c"], ["a"], None),
        ("Sub", ["sub_c", "a"], ["s"], None),
        ("Split", ["s"], ["s1", "s2"], {"axis": 1, "split": [4, 4]}),
        ("Concat", ["s2", "s1"], ["cc"], {"axis": 1}),
        ("Pad", ["cc", "pads"], ["pd"], {"mode": "reflect"}),
        ("Conv", ["pd", "w2"], ["c2"], {"pads": [0, 0, 0, 0]}),
        ("Resize", ["c2", "roi", "scales2"], ["rz"], {"mode": "linear", "coordinate_transformation_mode": "half_pixel"}),
        ("Slice", ["rz", "starts", "ends", "axes"], ["sl"], None),
        ("Clip", ["sl", "lo", "hi"], ["cl"], None),
        ("Tanh", ["cl"], ["th"], None),
        ("AveragePool", ["th"], ["ap"], {"kernel_shape": [2, 2], "strides": [2, 2]}),
        ("Resize", ["ap", "roi", "scales3"], ["y"], {"mode": "linear", "coordinate_transformation_mode": "align_corners"}),
    ]
    case("many_ops", nodes, inits, exe, 3, [(8, 12), (6, 10)], expect_ops={"ChannelAffine", "Pad", "Slice", "ResizeBilinear", "AvgPool2D"})

    inits = {"w1": rnd(6, 3, 3, 3), "b1": rnd(6), "scales": np.array([1, 1, 2, 2], dtype=np.float32), "roi": np.zeros(0, dtype=np.float32),
             "w2": rnd(3, 6, 3, 3), "div": np.array([2.0], dtype=np.float32)}
    nodes = [
        ("Conv", ["x", "w1", "b1"], ["c1"], {"pads": [1, 2, 1, 0]}),
        ("Relu", ["c1"], ["r"], None),
        ("Resize", ["r", "roi", "scales"], ["nn"], {"mode": "nearest"}),
        ("Div", ["nn", "div"], ["d"], None),
        ("Conv", ["d", "w2"], ["y"], {"pads": [1, 1, 1, 1], "strides": [1, 1], "dilations": [1, 1]}),
    ]
    case("asym_pad_conv", nodes, inits, exe, 2, [(8, 12)], expect_ops={"Pad", "ResizeNearest"})

    print("== fp16 default conversion ==")
    path = write_model("fp16", nodes[:0] or [("Conv", ["x", "w1", "b1"], ["y"], {"pads": [1, 1, 1, 1]})], {"w1": rnd(3, 3, 3, 3), "b1": rnd(3)})
    out = os.path.join(WORK, "fp16.vva")
    code, stdout, stderr = run_cli(["-i", path, "-o", out])
    T.check(code == 0 and "verified" in stdout, "the CLI converts an .onnx file: " + stderr)
    meta, _ = read_vva(out)
    T.check({t["name"]: t["dtype"] for t in meta["tensors"]}.get("__p0_w") == "f16", "weights are f16 by default")

    print("== things that must be refused clearly ==")

    def refuse(label, nodes, inits, needle, dims=("N", 3, "H", "W"), opset=13):
        path = write_model("bad_" + label, nodes, inits, dims=dims, opset=opset)
        code, stdout, stderr = run_cli(["-i", path, "-o", os.path.join(WORK, "bad.vva")])
        T.check(code != 0 and needle in stderr, "%s is refused with a clear message (%s)" % (label, stderr.strip()[:110]))

    refuse("matmul", [("MatMul", ["x", "w"], ["y"], None)], {"w": rnd(3, 3)}, "MatMul")
    refuse("dcr", [("Conv", ["x", "w"], ["c"], {"pads": [1, 1, 1, 1]}), ("DepthToSpace", ["c"], ["y"], {"blocksize": 2})], {"w": rnd(12, 3, 3, 3)}, "CRD")
    refuse("sizes", [("Resize", ["x", "", "", "sz"], ["y"], {"mode": "nearest"})], {"sz": np.array([1, 3, 16, 16], dtype=np.int64)}, "sizes")
    refuse("same_upper", [("Conv", ["x", "w"], ["y"], {"auto_pad": "SAME_UPPER"})], {"w": rnd(3, 3, 3, 3)}, "auto_pad")
    refuse("fractional", [("Resize", ["x", "", "sc"], ["y"], {"mode": "nearest"})], {"sc": np.array([1, 1, 1.5, 1.5], dtype=np.float32)}, "integer")
    refuse("not_multiple", [("Conv", ["x", "w"], ["c"], {"pads": [0, 0, 0, 0]})], {"w": rnd(3, 3, 3, 3)}, "integer multiple")
    refuse("fixed_channels", [("Relu", ["x"], ["y"], None)], {}, "channel", dims=("N", "C", "H", "W"))
    refuse("ceil_mode", [("MaxPool", ["x"], ["y"], {"kernel_shape": [2, 2], "strides": [2, 2], "ceil_mode": 1})], {}, "ceil_mode")
    junk = os.path.join(WORK, "junk.onnx")
    with open(junk, "wb") as f:
        f.write(b"\x0a\x05hello")
    code, stdout, stderr = run_cli(["-i", junk, "-o", os.path.join(WORK, "bad.vva")])
    T.check(code != 0 and "ONNX" in stderr, "a file that is not ONNX is refused: " + stderr.strip()[:90])
    with open(os.path.join(WORK, "fp16.onnx"), "rb") as f:
        data = f.read()
    with open(junk, "wb") as f:
        f.write(data[: len(data) // 2])
    code, stdout, stderr = run_cli(["-i", junk, "-o", os.path.join(WORK, "bad.vva")])
    T.check(code != 0 and "Traceback" not in stderr, "a truncated ONNX file is refused without a crash: " + stderr.strip()[:90])
    return T.finish()


if __name__ == "__main__":
    sys.exit(main())
