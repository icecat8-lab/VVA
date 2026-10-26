#!/usr/bin/env python3
"""Converter tests: checkpoint -> .vva v2, numerical verification and error handling.

Run from the repository root:  python3 tests/test_converter.py
"""

import io
import os
import contextlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, WORK, make_rrdbnet_state, save_npz  # noqa: E402

from vva import cli  # noqa: E402
from vva.converter import ConversionError, _unwrap, build_rrdbnet, inspect_rrdbnet, load_checkpoint  # noqa: E402
from vva.fileformat import read_vva  # noqa: E402
from vva.reference import rrdbnet_forward, run_graph  # noqa: E402

T = Checker("test_converter")


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(args)
    return code, out.getvalue(), err.getvalue()


def main():
    sd = make_rrdbnet_state(nf=6, gc=5, nb=2, seed=3)
    npz = save_npz(sd, "ok")

    print("== normal conversion ==")
    out = os.path.join(WORK, "ok.vva")
    code, stdout, stderr = run_cli(["-i", npz, "-o", out, "--name", "test-model", "--author", "me"])
    T.check(code == 0, "conversion succeeds (exit code 0): " + stderr)
    T.check("verified" in stdout, "numerical verification ran")
    with open(out, "rb") as f:
        T.check(f.read(4) == b"VVA2", "output is a version-2 file (VVA2 magic)")
    meta, tensors = read_vva(out)
    T.check(meta["meta"]["scale"] == 4 and meta["meta"]["name"] == "test-model", "metadata name and scale")
    T.check(meta["inputs"][0]["shape"] == [1, 3, -1, -1], "dynamic input size")
    ops = {n["op"] for n in meta["nodes"]}
    T.check(ops == {"Conv2D", "LeakyReLU", "Mul", "Add", "Concat", "ResizeNearest"}, "graph operator set: %s" % sorted(ops))
    dtypes = {t["name"]: t["dtype"] for t in meta["tensors"]}
    T.check(all(v == "f16" for k, v in dtypes.items() if k.endswith(".weight")), "weights are stored as f16")
    T.check(all(v == "f32" for k, v in dtypes.items() if k.endswith(".bias")), "biases are stored as f32")
    T.check(not os.path.exists(os.path.join(WORK, "ok.json")), "no stray .json sidecar is written")

    print("== graph == network ==")
    x = np.random.RandomState(7).rand(3, 9, 11)
    deq = {k: np.asarray(v, dtype=np.float64) for k, v in tensors.items()}
    a = run_graph(meta, tensors, x)
    b = rrdbnet_forward(deq, x)
    T.check(a.shape == (3, 36, 44), "x4 output shape %s" % (a.shape,))
    T.check(float(np.max(np.abs(a - b))) < 1e-9, "stored graph equals the plain network (max diff %.2e)" % np.max(np.abs(a - b)))
    ref = rrdbnet_forward({k: v.astype(np.float64) for k, v in sd.items()}, x)
    span = float(np.max(np.abs(ref)))
    T.check(float(np.max(np.abs(a - ref))) / span < 5e-3, "fp16 storage error stays small")

    print("== --fp32 ==")
    out32 = os.path.join(WORK, "ok32.vva")
    code, stdout, stderr = run_cli(["-i", npz, "-o", out32, "--fp32"])
    T.check(code == 0, "fp32 conversion succeeds")
    meta32, tensors32 = read_vva(out32)
    T.check(all(t["dtype"] == "f32" for t in meta32["tensors"]), "everything is f32")
    got = run_graph(meta32, tensors32, x)
    T.check(float(np.max(np.abs(got - ref))) / span < 1e-5, "fp32 file reproduces the original weights")

    print("== default output path and .npz wrapper keys ==")
    code, _, _ = run_cli(["-i", npz, "--no-verify"])
    T.check(code == 0 and os.path.exists(npz[:-4] + ".vva"), "default output goes next to the input")
    state = _unwrap({"params_ema": {"module." + k: v for k, v in sd.items()},
                     "params": {"ignored": np.zeros(1, np.float32)}})
    T.check(set(state) == set(sd), "params_ema is picked and 'module.' is stripped")
    T.check("conv_first.weight" in state, "names are clean")

    print("== large weights fall back to f32 ==")
    big = {k: v.copy() for k, v in sd.items()}
    big["conv_hr.weight"][0, 0, 0, 0] = 1.0e5
    meta_big = build_rrdbnet(big, "big").to_meta()
    kinds = {t["name"]: t["dtype"] for t in meta_big["tensors"]}
    T.check(kinds["conv_hr.weight"] == "f32" and kinds["conv_up1.weight"] == "f16", "only the tensor that overflows fp16 is f32")

    print("== errors are reported, not raised ==")
    bad = {k: v for k, v in sd.items() if k != "body.1.rdb2.conv3.weight"}
    code, _, err = run_cli(["-i", save_npz(bad, "missing"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code == 1 and "body.1.rdb2.conv3.weight" in err, "missing weight names the key")
    x2 = {k: v.copy() for k, v in sd.items()}
    x2["conv_first.weight"] = np.zeros((6, 12, 3, 3), np.float32)
    code, _, err = run_cli(["-i", save_npz(x2, "x2"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code == 1 and "12 input channels" in err, "pixel-unshuffle (x2/x1) models are rejected clearly")
    wrong = {k: v.copy() for k, v in sd.items()}
    wrong["body.0.rdb1.conv5.weight"] = np.zeros((6, 7, 3, 3), np.float32)
    code, _, err = run_cli(["-i", save_npz(wrong, "wrong"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code == 1 and "body.0.rdb1.conv5.weight" in err, "wrong shape names the key")
    nan = {k: v.copy() for k, v in sd.items()}
    nan["conv_up2.bias"][0] = np.nan
    code, _, err = run_cli(["-i", save_npz(nan, "nan"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code == 1 and "NaN" in err, "NaN weights are rejected")
    code, _, err = run_cli(["-i", os.path.join(WORK, "does-not-exist.pth")])
    T.check(code == 1 and "not found" in err, "missing input file")
    garbage = os.path.join(WORK, "garbage.pth")
    with open(garbage, "wb") as f:
        f.write(b"not a checkpoint")
    code, _, err = run_cli(["-i", garbage])
    T.check(code == 1 and err.startswith("error:"), "garbage .pth gives an error message (exit 1)")
    try:
        inspect_rrdbnet({"foo": np.zeros(1, np.float32)})
        T.check(False, "unrelated checkpoint should raise")
    except ConversionError as exc:
        T.check("conv_first" in str(exc), "unrelated checkpoint is explained")
    try:
        load_checkpoint(os.path.join(WORK, "garbage.pth"))
    except ConversionError:
        pass

    return T.finish()


if __name__ == "__main__":
    sys.exit(main())
