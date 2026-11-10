#!/usr/bin/env python3
import io
import os
import contextlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, WORK, build_binaries, exec_run, make_rrdbnet_state, run, ENV, save_npz

from vva import cli
from vva.converter import ConversionError, load_checkpoint, remap_old_esrgan, rrdbnet_unshuffle, srvgg_forward
from vva.fileformat import read_vva
from vva.reference import rrdbnet_forward

T = Checker("test_archs")


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(args)
    return code, out.getvalue(), err.getvalue()


def rrdb_with_input_channels(in_ch, seed):
    sd = make_rrdbnet_state(nf=6, gc=4, nb=2, seed=seed)
    rng = np.random.RandomState(seed + 9)
    sd["conv_first.weight"] = (rng.randn(6, in_ch, 3, 3) * 0.7 / np.sqrt(in_ch * 9)).astype(np.float32)
    return sd


def to_old_esrgan(sd):
    nb = 0
    while "body.%d.rdb1.conv1.weight" % nb in sd:
        nb += 1
    old = {}
    fixed = {"conv_first": "model.0", "conv_body": "model.1.sub.%d" % nb, "conv_up1": "model.3", "conv_up2": "model.6",
             "conv_hr": "model.8", "conv_last": "model.10"}
    for key, value in sd.items():
        head, kind = key.rsplit(".", 1)
        if head in fixed:
            old[fixed[head] + "." + kind] = value
        else:
            _, i, rdb, conv = head.split(".")
            old["model.1.sub.%s.RDB%s.%s.0.%s" % (i, rdb[3:], conv, kind)] = value
    return old


def make_srvgg_state(nf, nc, up, seed, prelu_scalar=False):
    rng = np.random.RandomState(seed)
    sd = {}

    def conv(i, out_c, in_c):
        sd["body.%d.weight" % i] = (rng.randn(out_c, in_c, 3, 3) * 0.8 / np.sqrt(in_c * 9)).astype(np.float32)
        sd["body.%d.bias" % i] = (rng.randn(out_c) * 0.05).astype(np.float32)

    def prelu(i):
        n = 1 if prelu_scalar else nf
        sd["body.%d.weight" % i] = (0.1 + rng.rand(n) * 0.3).astype(np.float32)

    conv(0, nf, 3)
    prelu(1)
    for k in range(nc):
        conv(2 + 2 * k, nf, nf)
        prelu(3 + 2 * k)
    conv(2 + 2 * nc, 3 * up * up, nf)
    return sd


def check_model(exe, path, want_fn, sizes, label):
    meta, tensors = read_vva(path)
    for (h, w) in sizes:
        x = np.random.RandomState(h * 7 + w).rand(3, h, w).astype(np.float32)
        want = want_fn(x.astype(np.float64)).reshape(-1)
        text, got = exec_run(exe, path, x, 1)
        T.check(got is not None, "%s %dx%d runs (%s)" % (label, h, w, text))
        if got is None:
            continue
        err = float(np.max(np.abs(got - want)) / max(float(np.max(np.abs(want))), 1e-6)) if got.size == want.size else -1
        T.check(got.size == want.size and err < 0.05, "%s %dx%d matches the float32 network within fp16 error (%.2e)" % (label, h, w, err))


def main():
    exes = build_binaries(sanitize=os.environ.get("VVA_NO_SAN") is None)
    exe = exes["exec_run"]

    print("== RRDBNet x2 / x1 (pixel-unshuffle input) ==")
    for in_ch, scale in ((12, 2), (48, 1)):
        sd = rrdb_with_input_channels(in_ch, in_ch)
        T.check(rrdbnet_unshuffle(sd) == 4 // scale, "x%d: unshuffle factor detected" % scale)
        npz = save_npz(sd, "rrdb_x%d" % scale)
        out = os.path.join(WORK, "rrdb_x%d.vva" % scale)
        code, stdout, stderr = run_cli(["-i", npz, "-o", out])
        T.check(code == 0 and "verified" in stdout and ("x%d RRDBNet" % scale) in stdout, "x%d converts and verifies: %s %s" % (scale, stdout[-120:], stderr))
        meta, tensors = read_vva(out)
        T.check(meta["meta"]["scale"] == scale, "x%d: scale stored in the file" % scale)
        T.check(any(n["op"] == "PixelUnshuffle" for n in meta["nodes"]) == (scale != 4), "x%d: unshuffle node present" % scale)
        check_model(exe, out, lambda x, sd=sd: rrdbnet_forward(sd, x), [(8, 12), (16, 8)], "RRDBNet x%d" % scale)

    print("== old ESRGAN key layout ==")
    sd = make_rrdbnet_state(nf=6, gc=4, nb=2, seed=21)
    old = to_old_esrgan(sd)
    back = remap_old_esrgan(old)
    T.check(set(back) == set(sd) and all(np.array_equal(back[k], sd[k]) for k in sd), "old keys map back to basicsr keys")
    npz = save_npz(old, "old_esrgan")
    out = os.path.join(WORK, "old_esrgan.vva")
    code, stdout, stderr = run_cli(["-i", npz, "-o", out])
    T.check(code == 0 and "verified" in stdout, "old-format checkpoint converts: %s %s" % (stdout[-100:], stderr))
    check_model(exe, out, lambda x, sd=sd: rrdbnet_forward(sd, x), [(8, 12)], "old ESRGAN")

    print("== SRVGGNetCompact (Real-ESRGAN general / anime video) ==")
    for up in (4, 2, 1):
        sd = make_srvgg_state(nf=8, nc=3, up=up, seed=up)
        npz = save_npz(sd, "srvgg_x%d" % up)
        out = os.path.join(WORK, "srvgg_x%d.vva" % up)
        code, stdout, stderr = run_cli(["-i", npz, "-o", out])
        T.check(code == 0 and "verified" in stdout and ("SRVGGNetCompact x%d" % up) in stdout, "x%d converts and verifies: %s %s" % (up, stdout[-140:], stderr))
        meta, _ = read_vva(out)
        T.check(meta["meta"]["scale"] == up, "x%d: scale stored" % up)
        check_model(exe, out, lambda x, sd=sd: srvgg_forward(sd, x), [(7, 9), (1, 1), (12, 5)], "SRVGG x%d" % up)
    sd = make_srvgg_state(nf=8, nc=2, up=4, seed=5, prelu_scalar=True)
    out = os.path.join(WORK, "srvgg_scalar.vva")
    code, stdout, stderr = run_cli(["-i", save_npz(sd, "srvgg_scalar"), "-o", out])
    T.check(code == 0, "a PReLU with a single shared slope converts: " + stderr)
    check_model(exe, out, lambda x, sd=sd: srvgg_forward(sd, x), [(6, 6)], "SRVGG shared slope")

    print("== things that must be refused clearly ==")
    sd = make_srvgg_state(nf=8, nc=2, up=4, seed=6)
    del sd["body.3.weight"]
    code, stdout, stderr = run_cli(["-i", save_npz(sd, "srvgg_nolayer"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code != 0 and "PReLU" in stderr, "missing activation weights are explained: " + stderr[:120])
    sd = make_srvgg_state(nf=8, nc=2, up=4, seed=6)
    sd["body.%d.weight" % (2 + 2 * 2)] = sd["body.%d.weight" % (2 + 2 * 2)][:10]
    sd["body.%d.bias" % (2 + 2 * 2)] = sd["body.%d.bias" % (2 + 2 * 2)][:10]
    code, stdout, stderr = run_cli(["-i", save_npz(sd, "srvgg_badout"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code != 0 and "upscale" in stderr, "bad output channel count is explained: " + stderr[:120])
    sd = rrdb_with_input_channels(5, 3)
    code, stdout, stderr = run_cli(["-i", save_npz(sd, "rrdb_badin"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code != 0 and "12" in stderr, "RRDBNet with an odd input channel count is refused: " + stderr[:120])
    code, stdout, stderr = run_cli(["-i", save_npz({"foo.weight": np.zeros((2, 2), dtype=np.float32)}, "junk"), "-o", os.path.join(WORK, "x.vva")])
    T.check(code != 0 and "ONNX" in stderr, "unknown networks point to the ONNX route: " + stderr[:100])

    print("== engine: image sizes that are not a multiple of the model's input block ==")
    for label, up, name in (("RRDBNet x2", 2, "rrdb_x2"), ("RRDBNet x1", 1, "rrdb_x1")):
        r = run([exes["engine_multiple_test"], os.path.join(WORK, name + ".vva"), str(up)], env=ENV)
        T.check(r.returncode == 0, "%s engine test passes: %s %s" % (label, r.stdout[-300:], r.stderr[-200:]))
    return T.finish()


if __name__ == "__main__":
    sys.exit(main())
