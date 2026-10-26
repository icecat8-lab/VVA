#!/usr/bin/env python3
"""Engine tests: the production C++ executor/engine against the numpy reference.

Builds the real sources for the host with ASan+UBSan (no Vulkan), converts a
random RRDBNet, and checks numerics, threading, tiling, odd sizes, every
operator, hostile model files and the Engine API.

Run from the repository root:  python3 tests/test_engine.py
"""

import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ENV, ROOT, WORK, Checker, build_binaries, exec_run, make_rrdbnet_state, run, save_npz,  # noqa: E402
                    VVAWriter, read_vva)

from vva.converter import convert  # noqa: E402
from vva.reference import rrdbnet_forward, run_graph  # noqa: E402

T = Checker("test_engine")
EXE = build_binaries(sanitize=True)


def to_model(sd, name, **kw):
    npz = save_npz(sd, name)
    out = os.path.join(WORK, name + ".vva")
    convert(npz, out, name="test-model", log=lambda *_: None, **kw)
    return out


def rel_err(a, b):
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-12)


def check_rrdbnet(nf, gc, nb, h, w, seed):
    sd = make_rrdbnet_state(nf, gc, nb, seed=seed)
    model = to_model(sd, "rrdb_%d_%d_%d" % (nf, gc, nb))
    meta, stored = read_vva(model)
    deq = {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()}
    x = np.random.RandomState(seed).rand(3, h, w).astype(np.float32)
    want = rrdbnet_forward(deq, x.astype(np.float64)).reshape(-1)
    text, got1 = exec_run(EXE["exec_run.san"] if "exec_run.san" in EXE else EXE["exec_run"], model, x, 1)
    ok = got1 is not None
    T.check(ok, "nf=%d gc=%d nb=%d %dx%d runs (%s)" % (nf, gc, nb, h, w, text))
    if not ok:
        return model
    T.check("scale=4" in text, "reports scale 4")
    err = rel_err(got1.astype(np.float64), want)
    T.check(err < 3e-5, "nf=%d gc=%d nb=%d %dx%d matches numpy (rel err %.2e)" % (nf, gc, nb, h, w, err))
    for threads in (2, 5):
        _, gotn = exec_run(EXE["exec_run"], model, x, threads)
        T.check(gotn is not None and np.array_equal(got1, gotn), "%d threads give bit-identical output" % threads)
    return model


def write_graph(path, builder):
    w = builder()
    w.save(path)
    return path


def rnd(rng, *shape, scale=0.5):
    return (rng.randn(*shape) * scale).astype(np.float32)


def custom_graph_cases():
    rng = np.random.RandomState(11)
    cases = {}

    def g_convs():
        w = VVAWriter(name="convs", scale=1)
        w.add_input("input", [1, 3, -1, -1])
        w.add_tensor("w1", rnd(rng, 4, 3, 3, 3), "f16"); w.add_tensor("b1", rnd(rng, 4), "f32")
        w.add_node("Conv2D", ["input", "w1", "b1"], ["a"], {"stride": [2, 2], "padding": [1, 1]})
        w.add_node("ReLU", ["a"], ["a_r"])
        w.add_node("ResizeNearest", ["a_r"], ["up"], {"scale": [2, 2]})
        w.add_tensor("w2", rnd(rng, 6, 4, 1, 1), "f16")
        w.add_node("Conv2D", ["up", "w2"], ["c2"], {})                       # 1x1, no bias, no padding
        w.add_tensor("w3", rnd(rng, 6, 3, 5, 5, scale=0.1), "f16"); w.add_tensor("b3", rnd(rng, 6), "f32")
        w.add_node("Conv2D", ["input", "w3", "b3"], ["c3"], {"padding": [2, 2]})  # 5x5
        w.add_node("Add", ["c2", "c3"], ["s"])
        w.add_tensor("w4", rnd(rng, 6, 3, 3, 3, scale=0.2), "f32")             # groups=2, dilation 2
        w.add_node("Conv2D", ["s", "w4"], ["c4"], {"padding": [2, 2], "dilation": [2, 2], "groups": 2})
        w.add_tensor("w5", rnd(rng, 6, 1, 3, 3, scale=0.3), "f16"); w.add_tensor("b5", rnd(rng, 6), "f32")
        w.add_node("Conv2D", ["c4", "w5", "b5"], ["c5"], {"padding": [1, 1], "groups": 6})  # depthwise
        w.add_tensor("w6", rnd(rng, 3, 6, 3, 3, scale=0.2), "f16")
        w.add_node("Conv2D", ["c5", "w6"], ["output"], {"padding": [1, 1]})
        w.add_output("output", [1, 3, -1, -1])
        return w
    cases["convs"] = (g_convs, 3, 12, 10)

    def g_ops():
        w = VVAWriter(name="ops", scale=1)
        w.add_input("input", [1, 3, -1, -1])
        w.add_tensor("w1", rnd(rng, 4, 3, 3, 3), "f16"); w.add_tensor("b1", rnd(rng, 4), "f32")
        w.add_node("Conv2D", ["input", "w1", "b1"], ["a"], {"padding": [1, 1]})
        w.add_node("Sigmoid", ["a"], ["sg"])
        w.add_node("Mul", ["a", "sg"], ["silu"])                    # tensor * tensor
        w.add_node("Add", ["silu", "silu"], ["twice"])              # same operand twice (in-place edge case)
        w.add_node("Mul", ["twice"], ["scaled"], {"scalar": 0.37})
        w.add_node("Clip", ["scaled"], ["clipped"], {"min": -0.2, "max": 0.4})
        w.add_node("Clip", ["a"], ["lowclip"], {"min": 0.1})
        w.add_node("LeakyReLU", ["a"], ["lk"], {"alpha": 0.1})
        w.add_node("LeakyReLU", ["a"], ["lk_default"])
        w.add_node("Concat", ["clipped", "lowclip", "lk", "lk_default", "a"], ["cat"], {"axis": 1})
        w.add_tensor("w2", rnd(rng, 3, 20, 1, 1, scale=0.3), "f16")
        w.add_node("Conv2D", ["cat", "w2"], ["output"], {})
        w.add_output("output", [1, 3, -1, -1])
        return w
    cases["ops"] = (g_ops, 3, 7, 9)

    for r, name in ((2, "shuffle2"), (3, "shuffle3")):
        def g_shuffle(r=r):
            w = VVAWriter(name="shuffle", scale=r)
            w.add_input("input", [1, 3, -1, -1])
            w.add_tensor("w", rnd(rng, 3 * r * r, 3, 3, 3, scale=0.3), "f16"); w.add_tensor("b", rnd(rng, 3 * r * r), "f32")
            w.add_node("Conv2D", ["input", "w", "b"], ["c"], {"padding": [1, 1]})
            w.add_node("PixelShuffle", ["c"], ["output"], {"upscale": r})
            w.add_output("output", [1, 3, -1, -1])
            return w
        cases[name] = (g_shuffle, 3, 6, 5)

    def g_aniso():
        # Different factors per axis: nearest resize [2,3] undone by a conv with stride [2,3].
        w = VVAWriter(name="aniso", scale=1)
        w.add_input("input", [1, 2, -1, -1])
        w.add_node("ResizeNearest", ["input"], ["up"], {"scale": [2, 3]})
        w.add_tensor("w", rnd(rng, 2, 2, 3, 3, scale=0.3), "f16")
        w.add_node("Conv2D", ["up", "w"], ["output"], {"padding": [1, 1], "stride": [2, 3]})
        w.add_output("output", [1, 2, -1, -1])
        return w
    cases["aniso"] = (g_aniso, 2, 5, 4)
    return cases


def half_exactness():
    bits = np.array([0x0001, 0x0002, 0x03FF, 0x0400, 0x0401, 0x3C00, 0x7BFF, 0x8001, 0x83FF, 0x8400, 0xBC00,
                     0xFBFF, 0x0000, 0x8000, 0x3555, 0x5640], dtype=np.uint16)
    vals = bits.view(np.float16).astype(np.float32)
    w = VVAWriter(name="halves", scale=1)
    w.add_input("input", [1, 1, -1, -1])
    w.add_tensor("w", vals.reshape(-1, 1, 1, 1), "f16")
    w.add_node("Conv2D", ["input", "w"], ["output"], {})
    w.add_output("output", [1, len(vals), -1, -1])
    path = os.path.join(WORK, "halves.vva")
    w.save(path)
    text, got = exec_run(EXE["exec_run"], path, np.ones((1, 2, 3), np.float32), 1)
    T.check(got is not None, "half-float model runs (%s)" % text)
    if got is not None:
        got = got.reshape(len(vals), -1)
        T.check(all(np.all(got[i] == vals[i]) for i in range(len(vals))), "all fp16 values (subnormals included) convert exactly")


def hostile_models():
    """Valid containers with nonsense graphs: must be rejected cleanly, never crash."""
    rng = np.random.RandomState(5)

    def base(op, inputs, attrs, tensors, in_shape=(1, 3, -1, -1), out_shape=(1, 3, -1, -1)):
        w = VVAWriter(name="h", scale=1)
        w.add_input("input", list(in_shape))
        for name, arr in tensors.items():
            w.add_tensor(name, arr, "f32")
        w.add_node(op, inputs, ["output"], attrs)
        w.add_output("output", list(out_shape))
        return w

    k3 = {"w": rnd(rng, 3, 3, 3, 3)}
    cases = [
        ("unknown operator", base("Gemm", ["input"], {}, {}), "unsupported operator"),
        ("fixed input size", base("ReLU", ["input"], {}, {}, in_shape=(1, 3, 64, 64)), "fixed size"),
        ("batch 2", base("ReLU", ["input"], {}, {}, in_shape=(2, 3, -1, -1)), "batch 1"),
        ("zero stride", base("Conv2D", ["input", "w"], {"stride": [0, 1], "padding": [1, 1]}, k3), "stride"),
        ("negative padding", base("Conv2D", ["input", "w"], {"padding": [-1, 1]}, k3), "padding"),
        ("huge padding", base("Conv2D", ["input", "w"], {"padding": [1000000000, 1]}, k3), "padding"),
        ("groups 0", base("Conv2D", ["input", "w"], {"groups": 0, "padding": [1, 1]}, k3), "groups"),
        ("groups mismatch", base("Conv2D", ["input", "w"], {"groups": 2, "padding": [1, 1]}, k3), "channels"),
        ("channels mismatch", base("Conv2D", ["input", "w"], {"padding": [1, 1]}, {"w": rnd(rng, 3, 4, 3, 3)}), "channels"),
        ("valid conv shrinks", base("Conv2D", ["input", "w"], {}, k3), "integer multiple"),
        ("kernel larger than input", base("Conv2D", ["input", "w"], {}, {"w": rnd(rng, 3, 3, 40, 40)}), "smaller than the kernel"),
        ("bias wrong size", base("Conv2D", ["input", "w", "b"], {"padding": [1, 1]}, {**k3, "b": rnd(rng, 5)}), "bias"),
        ("conv weight is not constant", base("Conv2D", ["input", "input"], {}, {}), "weight"),
        ("concat on axis 2", base("Concat", ["input", "input"], {"axis": 2}, {}), "channel axis"),
        ("concat one input", base("Concat", ["input"], {}, {}), "two inputs"),
        ("clip min > max", base("Clip", ["input"], {"min": 1.0, "max": 0.0}, {}), "min"),
        ("mul without scalar", base("Mul", ["input"], {}, {}), "two inputs"),
        ("add one input", base("Add", ["input"], {}, {}), "two inputs"),
        ("resize without scale", base("ResizeNearest", ["input"], {}, {}), "scale"),
        ("resize scale 0", base("ResizeNearest", ["input"], {"scale": [0, 2]}, {}), "scale"),
        ("pixelshuffle bad channels", base("PixelShuffle", ["input"], {"upscale": 2}, {}), "divisible"),
        ("pixelshuffle huge", base("PixelShuffle", ["input"], {"upscale": 1000}, {}), "upscale"),
        ("non-integer scale", base("PixelShuffle", ["input"], {"upscale": 1}, {}, out_shape=(1, 3, -1, -1)), None),
        ("wrong declared output channels", base("ReLU", ["input"], {}, {}, out_shape=(1, 7, -1, -1)), "output channels"),
        ("scale mismatch", base("ReLU", ["input"], {}, {}), None),
    ]
    for label, writer, expect in cases:
        path = os.path.join(WORK, "hostile.vva")
        if label == "scale mismatch":
            writer.meta_info["scale"] = 4
            expect = "declares scale"
        writer.save(path, validate=False)
        text, got = exec_run(EXE["exec_run"], path, np.ones((3, 8, 8), np.float32), 1)
        if label == "non-integer scale":  # identity-like graph is valid, scale 1
            T.check(got is not None, "valid identity-like graph runs")
            continue
        T.check(got is None and ("PREPARE_ERROR" in text or "LOAD_ERROR" in text), "%s -> rejected cleanly (%s)" % (label, text[:80]))
        if expect:
            T.check(expect in text, "%s: message mentions '%s' (%s)" % (label, expect, text[:100]))
        T.check("Sanitizer" not in text and "runtime error" not in text, "%s: no sanitizer report" % label)

    # Mutated real model files must never crash the executor.
    sd = make_rrdbnet_state(4, 3, 1, seed=2)
    model = to_model(sd, "fuzz_base")
    raw = bytearray(open(model, "rb").read())
    rng = np.random.RandomState(99)
    crashes = 0
    for _ in range(40):
        mutated = bytearray(raw)
        for _ in range(rng.randint(1, 6)):
            mutated[rng.randint(0, len(mutated))] = rng.randint(0, 256)
        path = os.path.join(WORK, "mut.vva")
        open(path, "wb").write(mutated)
        r = run([EXE["exec_run"], path, os.path.join(WORK, "in.f32"), "3", "8", "8", os.path.join(WORK, "o.f32"), "1"], env=ENV)
        if r.returncode in (98, 99) or "Sanitizer" in r.stderr or r.returncode < 0:
            crashes += 1
    T.check(crashes == 0, "40 corrupted model files: no crashes (%d)" % crashes)


def cli_tests(model, sd_deq):
    cli = EXE["vva-cli"]
    r = run([cli, "info", model], env=ENV)
    T.check(r.returncode == 0 and "x4" in r.stdout and "test-model" in r.stdout, "vva-cli info shows the model: " + r.stdout.strip())
    r = run([cli, "info", os.path.join(WORK, "nope.vva")], env=ENV)
    T.check(r.returncode == 1 and "error" in r.stderr, "vva-cli info on a missing file fails")
    legacy = os.path.join(WORK, "legacy.vva")
    open(legacy, "wb").write(b"VVA1" + b"\0" * 100)
    r = run([cli, "info", legacy], env=ENV)
    T.check(r.returncode == 1 and "legacy" in r.stderr, "legacy v1 file is explained")
    r = run([cli, "--version"], env=ENV)
    T.check(r.returncode == 0 and "vva-cli" in r.stdout, "--version")
    r = run([cli, "--info"], env=ENV)
    T.check(r.returncode == 2, "--info reports no Vulkan on this host")

    rng = np.random.RandomState(8)
    h, w = 9, 110
    img = rng.randint(0, 256, size=(h, w, 3)).astype(np.uint8)
    ppm = os.path.join(WORK, "in.ppm")
    with open(ppm, "wb") as f:
        f.write(b"P6\n# comment\n%d %d\n255\n" % (w, h))
        f.write(img.tobytes())
    out = os.path.join(WORK, "out.ppm")
    r = run([cli, "run", model, ppm, out, "--threads", "2"], env=ENV)
    T.check(r.returncode == 0 and "-> 440x36" in r.stdout, "vva-cli run works: " + (r.stdout + r.stderr).strip()[-120:])
    data = open(out, "rb").read()
    header, rest = data.split(b"\n255\n", 1)
    T.check(header == b"P6\n440 36", "output PPM header")
    got = np.frombuffer(rest, dtype=np.uint8).reshape(36, 440, 3).astype(int)
    x = (img.astype(np.float64) / 255.0).transpose(2, 0, 1)
    ref = rrdbnet_forward(sd_deq, x)
    want = np.floor(np.clip(ref, 0, 1) * 255 + 0.5).transpose(1, 2, 0).astype(int)
    diff = np.abs(got - want)
    T.check(diff.max() <= 1, "vva-cli output matches numpy to within one level (max %d)" % diff.max())
    T.check(float(np.mean(diff == 0)) > 0.97, "almost all pixels are identical (%.3f)" % np.mean(diff == 0))
    T.check(want.std() > 5, "test image is informative (std %.1f)" % want.std())

    r = run([cli, "run", model, ppm, out, "--tile", "16", "--pad", "40"], env=ENV)
    T.check(r.returncode == 0, "tiled run works")
    tiled = np.frombuffer(open(out, "rb").read().split(b"\n255\n", 1)[1], dtype=np.uint8).reshape(36, 440, 3).astype(int)
    T.check(np.abs(tiled - want).max() <= 1, "tiled CLI output equals the whole-image result")
    with open(os.path.join(WORK, "bad.ppm"), "wb") as f:
        f.write(b"P6\n10 10\n255\nshort")
    r = run([cli, "run", model, os.path.join(WORK, "bad.ppm"), out], env=ENV)
    T.check(r.returncode == 1 and "truncated" in r.stderr, "truncated PPM is reported")
    r = run([cli, "run", model, ppm, out, "--bogus"], env=ENV)
    T.check(r.returncode == 1 and "unknown option" in r.stderr, "unknown option is reported")


def main():
    print("== RRDBNet vs numpy (odd sizes exercise every kernel remainder) ==")
    models = []
    for nf, gc, nb, h, w, seed in ((6, 5, 2, 11, 9, 1), (7, 3, 1, 9, 21, 2), (8, 4, 2, 5, 33, 3), (4, 4, 1, 1, 1, 4),
                                   (5, 2, 1, 2, 17, 5)):
        models.append(check_rrdbnet(nf, gc, nb, h, w, seed))

    print("== every operator and conv variant vs numpy ==")
    for name, (builder, c, h, w) in custom_graph_cases().items():
        path = os.path.join(WORK, "g_%s.vva" % name)
        builder().save(path)
        meta, stored = read_vva(path)
        x = np.random.RandomState(1).rand(c, h, w).astype(np.float32)
        want = run_graph(meta, {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()}, x.astype(np.float64)).reshape(-1)
        text, got = exec_run(EXE["exec_run"], path, x, 1)
        T.check(got is not None, "%s runs (%s)" % (name, text))
        if got is not None:
            T.check(got.size == want.size and rel_err(got.astype(np.float64), want) < 3e-5,
                    "%s matches numpy (rel err %.2e)" % (name, rel_err(got.astype(np.float64), want) if got.size == want.size else -1))
            _, got3 = exec_run(EXE["exec_run"], path, x, 3)
            T.check(got3 is not None and np.array_equal(got, got3), "%s: threads do not change the result" % name)
    half_exactness()

    print("== hostile and corrupted models ==")
    hostile_models()

    print("== Engine API ==")
    sd = make_rrdbnet_state(8, 4, 2, seed=21)
    model = to_model(sd, "engine_model")
    bad = os.path.join(WORK, "bad_for_engine.vva")
    open(bad, "wb").write(b"VVA2" + b"\0" * 200)
    r = run([EXE["engine_test"], model, bad], env=ENV)
    print(r.stdout.rstrip())
    T.check(r.returncode == 0, "engine_test passes (exit %d) %s" % (r.returncode, r.stderr[-300:]))

    print("== JNI pixel logic (host build with stub Android headers) ==")
    jni_bin = os.path.join(WORK, "jni_logic_test")
    core = [os.path.join(ROOT, "core", "src", f) for f in ("vva.cpp", "model.cpp", "exec.cpp")]
    r = run(["g++", "-std=c++17", "-g", "-O1", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
             "-DVVA_NO_VULKAN", "-pthread", "-I", os.path.join(ROOT, "tests", "stubs"),
             "-I", os.path.join(ROOT, "core", "include"), "-I", os.path.join(ROOT, "core", "src"),
             os.path.join(ROOT, "tests", "jni_logic_test.cpp")] + core + ["-o", jni_bin])
    T.check(r.returncode == 0, "jni_logic_test builds: " + r.stderr[-300:])
    if r.returncode == 0:
        r = run([jni_bin, model], env=ENV)
        print(r.stdout.rstrip())
        T.check(r.returncode == 0, "jni_logic_test passes (exit %d) %s" % (r.returncode, r.stderr[-300:]))

    print("== vva-cli ==")
    meta, stored = read_vva(model)
    cli_tests(model, {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()})

    print("== speed (informational, unsanitised -O2 host build) ==")
    fast = build_binaries(sanitize=False)
    sd = make_rrdbnet_state(32, 16, 3, seed=5)
    model = to_model(sd, "speed")
    x = np.random.RandomState(0).rand(3, 96, 96).astype(np.float32)
    t0 = time.time()
    text, got = exec_run(fast["exec_run"], model, x, 1)
    dt = time.time() - t0
    macs = 0
    meta, stored = read_vva(model)
    # conv MACs per run: output pixels * out_c * in_c * k*k
    shapes = {"input": (3, 96, 96)}
    for n in meta["nodes"]:
        a = n.get("attrs", {})
        if n["op"] == "Conv2D":
            wt = stored[n["inputs"][1]]
            c, h, w_ = shapes[n["inputs"][0]]
            macs += wt.shape[0] * wt.shape[1] * wt.shape[2] * wt.shape[3] * h * w_
            shapes[n["outputs"][0]] = (wt.shape[0], h, w_)
        elif n["op"] == "Concat":
            s = [shapes[i] for i in n["inputs"]]
            shapes[n["outputs"][0]] = (sum(v[0] for v in s), s[0][1], s[0][2])
        elif n["op"] == "ResizeNearest":
            c, h, w_ = shapes[n["inputs"][0]]
            shapes[n["outputs"][0]] = (c, h * 2, w_ * 2)
        else:
            shapes[n["outputs"][0]] = shapes[n["inputs"][0]]
    T.check(got is not None, "speed model runs")
    print("  %.2f GMAC in %.2f s on 1 thread = %.1f GFLOP/s" % (macs / 1e9, dt, 2 * macs / 1e9 / dt))

    return T.finish()


if __name__ == "__main__":
    sys.exit(main())
