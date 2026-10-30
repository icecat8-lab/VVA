#!/usr/bin/env python3
"""GPU path tests that need no GPU: the planner (fusion, memory layout, weight packing, grids) and the very
kernel sources that become shaders are replayed on the host and compared with the numpy reference.

Run from the repository root:  python3 tests/test_gpu_plan.py
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENV, ROOT, WORK, Checker, VVAWriter, make_rrdbnet_state, read_vva, run, save_npz  # noqa: E402

from vva.converter import convert  # noqa: E402
from vva.reference import run_graph  # noqa: E402

T = Checker("test_gpu_plan")


def build():
    exe = os.path.join(WORK, "gpu_emu_run")
    cmd = ["g++", "-std=c++17", "-g", "-O1", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
           "-DVVA_NO_VULKAN", "-pthread", "-I", os.path.join(ROOT, "core", "src", "gpu"), "-I", os.path.join(ROOT, "core", "src"),
           "-I", os.path.join(ROOT, "tests"),
           os.path.join(ROOT, "tests", "gpu_emu_run.cpp"), os.path.join(ROOT, "tests", "gpu_emu.cpp"),
           os.path.join(ROOT, "core", "src", "gpu", "gpu_plan.cpp"), os.path.join(ROOT, "core", "src", "model.cpp"), "-o", exe]
    r = run(cmd)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit("cannot build the emulator")
    return exe


def emu(exe, model, x, second=None):
    c, h, w = x.shape
    inp, out = os.path.join(WORK, "g_in.f32"), os.path.join(WORK, "g_out.f32")
    np.ascontiguousarray(x, dtype=np.float32).tofile(inp)
    if os.path.exists(out):
        os.remove(out)
    cmd = [exe, model, inp, str(c), str(h), str(w), out]
    if second:
        cmd += [str(second[0]), str(second[1])]
    r = run(cmd, env=ENV)
    text = (r.stdout + r.stderr).strip()
    return text, (np.fromfile(out, dtype=np.float32) if r.returncode == 0 else None)


def rel(a, b):
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-9)


def rnd(rng, *shape, scale=0.3):
    return (rng.randn(*shape) * scale).astype(np.float32)


def graphs():
    rng = np.random.RandomState(3)
    out = {}

    def conv(w, name, cin, cout, x, y, bias=True, **attrs):
        w.add_tensor(name + ".w", rnd(rng, cout, cin, 3, 3, scale=0.25), "f32")
        ins = [x, name + ".w"]
        if bias:
            w.add_tensor(name + ".b", rnd(rng, cout, scale=0.2), "f32")
            ins.append(name + ".b")
        w.add_node("Conv2D", ins, [y], {"stride": [1, 1], "padding": [1, 1], **attrs})

    def new(c=3):
        w = VVAWriter(name="g", scale=1)
        w.add_input("input", [1, c, -1, -1])
        return w

    # 1. epilogue fusion: conv+lrelu, conv+mul+add(residual defined earlier)
    w = new()
    conv(w, "a", 3, 8, "input", "a0")
    w.add_node("LeakyReLU", ["a0"], ["a1"], {"alpha": 0.1})
    conv(w, "b", 8, 8, "a1", "b0")
    w.add_node("Mul", ["b0"], ["b1"], {"scalar": 0.5})
    w.add_node("Add", ["b1", "a1"], ["b2"])
    conv(w, "c", 8, 3, "b2", "output")
    w.add_output("output", [1, 3, -1, -1])
    out["fuse_basic"] = (w, 3)

    # 2. ordering hazard: the Add's other operand is produced after the first convolution
    w = new()
    conv(w, "a", 3, 8, "input", "a0")
    w.add_node("Mul", ["a0"], ["a1"], {"scalar": 2.0})
    conv(w, "b", 3, 8, "input", "b0")
    w.add_node("Add", ["a1", "b0"], ["s"])
    w.add_node("ReLU", ["s"], ["r"])
    conv(w, "c", 8, 8, "r", "output")
    w.add_output("output", [1, 8, -1, -1])
    out["order_hazard"] = (w, 3)

    # 3. concat of three tensors, then convolution, with a tensor used by several consumers
    w = new()
    conv(w, "a", 3, 8, "input", "a0")
    conv(w, "b", 8, 16, "a0", "b0")
    w.add_node("Concat", ["a0", "b0", "a0"], ["cat"], {"axis": 1})
    conv(w, "c", 32, 8, "cat", "c0")
    w.add_node("Add", ["c0", "a0"], ["output"])
    w.add_output("output", [1, 8, -1, -1])
    out["concat_multi_use"] = (w, 3)

    # 4. nearest resize (including different factors per axis) followed by convolutions
    w = new()
    conv(w, "a", 3, 8, "input", "a0")
    w.add_node("ResizeNearest", ["a0"], ["u"], {"scale": [2, 3]})
    conv(w, "b", 8, 8, "u", "b0")
    w.add_node("LeakyReLU", ["b0"], ["b1"], {"alpha": 0.2})
    w.add_node("ResizeNearest", ["b1"], ["output"], {"scale": [2, 1]})
    w.add_output("output", [1, 8, -1, -1])
    out["resize"] = (w, 3)

    # 5. element-wise only operators, x + x, standalone scale and activation, output from an element-wise op
    w = new()
    conv(w, "a", 3, 8, "input", "a0")
    w.add_node("Add", ["a0", "a0"], ["d"])
    w.add_node("Mul", ["d"], ["m"], {"scalar": -0.75})
    w.add_node("LeakyReLU", ["m"], ["l"], {"alpha": 0.3})
    conv(w, "b", 8, 8, "l", "b0", bias=False)
    w.add_node("Add", ["b0", "l"], ["output"])
    w.add_output("output", [1, 8, -1, -1])
    out["eltwise_mix"] = (w, 3)

    # 6. awkward channel counts that are allowed: 3-channel input, 5-channel and 11-channel conv outputs
    w = new()
    conv(w, "a", 3, 5, "input", "a0")
    conv(w, "b", 5, 11, "a0", "output")
    w.add_output("output", [1, 11, -1, -1])
    out["odd_channels"] = (w, 3)
    return out


def main():
    exe = build()
    rng = np.random.RandomState(1)

    print("== RRDBNet on the GPU program vs numpy ==")
    for nf, gc, nb, h, w, seed in ((16, 8, 2, 11, 9, 1), (16, 8, 1, 7, 21, 2), (8, 8, 1, 3, 33, 3), (8, 8, 1, 1, 1, 4),
                                   (16, 8, 1, 2, 17, 5), (16, 8, 2, 16, 16, 6)):
        sd = make_rrdbnet_state(nf, gc, nb, seed=seed)
        model = os.path.join(WORK, "gp_rrdb_%d.vva" % seed)
        convert(save_npz(sd, "gp_rrdb_%d" % seed), model, name="t", log=lambda *_: None)
        meta, stored = read_vva(model)
        deq = {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()}
        x = rng.rand(3, h, w).astype(np.float32)
        text, got = emu(exe, model, x)
        T.check(got is not None, "nf=%d gc=%d nb=%d %dx%d plans and runs (%s)" % (nf, gc, nb, h, w, text[-120:]))
        if got is not None:
            want = run_graph(meta, deq, x.astype(np.float64)).reshape(-1)
            T.check(got.size == want.size and rel(got.astype(np.float64), want) < 5e-5,
                    "nf=%d gc=%d nb=%d %dx%d matches numpy (rel err %.1e)" % (nf, gc, nb, h, w, rel(got.astype(np.float64), want) if got.size == want.size else -1))
        # Memory layout changes between images: the second run must not see leftovers of the first.
        text2, got2 = emu(exe, model, x, second=(h + 5, w + 3))
        T.check(got2 is not None and got is not None and np.array_equal(got, got2), "%dx%d: a previous image of another size leaves no trace" % (h, w))

    print("== work splitting: tiny dispatch/submit targets give identical results ==")
    sd = make_rrdbnet_state(16, 8, 1, seed=11)
    model = os.path.join(WORK, "gp_split.vva")
    convert(save_npz(sd, "gp_split"), model, name="t", log=lambda *_: None)
    x = rng.rand(3, 13, 19).astype(np.float32)
    _, whole = emu(exe, model, x)
    env = dict(ENV, VVA_TEST_DISPATCH_FLOPS="20000")
    inp, out = os.path.join(WORK, "g_in.f32"), os.path.join(WORK, "g_out2.f32")
    x.tofile(inp)
    r = run([exe, model, inp, "3", "13", "19", out], env=env)
    split = np.fromfile(out, dtype=np.float32) if r.returncode == 0 else None
    T.check(split is not None and whole is not None and np.array_equal(whole, split), "split dispatches equal the single-dispatch result: " + (r.stdout + r.stderr)[-120:])
    T.check("commands=" in r.stdout and int(r.stdout.split("commands=")[1].split()[0]) > 150, "tiny targets really produce many commands (" + r.stdout.strip()[-60:] + ")")

    print("== channel rule: growth channels must be multiples of 8 ==")
    sd = make_rrdbnet_state(16, 4, 1, seed=12)
    model = os.path.join(WORK, "gp_gc4.vva")
    convert(save_npz(sd, "gp_gc4"), model, name="t", log=lambda *_: None)
    text, _ = emu(exe, model, rng.rand(3, 8, 8).astype(np.float32))
    T.check("UNSUPPORTED" in text and "multiples of 8" in text, "gc=4 stays on the CPU (" + text[:100] + ")")

    print("== fusion, layout and unusual graphs ==")
    for name, (writer, cin) in graphs().items():
        path = os.path.join(WORK, "gp_%s.vva" % name)
        writer.save(path)
        meta, stored = read_vva(path)
        deq = {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()}
        for (h, w) in ((6, 10), (9, 5), (1, 7), (5, 33)):
            x = rng.rand(cin, h, w).astype(np.float32)
            text, got = emu(exe, path, x)
            T.check(got is not None, "%s %dx%d runs (%s)" % (name, h, w, text[-100:]))
            if got is not None:
                want = run_graph(meta, deq, x.astype(np.float64)).reshape(-1)
                e = rel(got.astype(np.float64), want) if got.size == want.size else -1
                T.check(0 <= e < 5e-5, "%s %dx%d matches numpy (rel err %.1e)" % (name, h, w, e))

    print("== models the GPU path must refuse (they stay on the CPU) ==")
    rng2 = np.random.RandomState(8)

    def refuse(label, writer, expect):
        p = os.path.join(WORK, "gp_refuse.vva")
        writer.save(p, validate=False)
        text, _ = emu(exe, p, np.ones((3, 8, 8), np.float32))
        T.check("UNSUPPORTED" in text and expect in text, "%s -> refused (%s)" % (label, text[:110]))

    w = VVAWriter(name="r", scale=1)
    w.add_input("input", [1, 3, -1, -1])
    w.add_tensor("w", rnd(rng2, 8, 3, 1, 1), "f32")
    w.add_node("Conv2D", ["input", "w"], ["output"], {})
    w.add_output("output", [1, 8, -1, -1])
    refuse("1x1 convolution", w, "3x3")

    w = VVAWriter(name="r", scale=1)
    w.add_input("input", [1, 3, -1, -1])
    w.add_tensor("w", rnd(rng2, 8, 3, 3, 3), "f32")
    w.add_node("Conv2D", ["input", "w"], ["output"], {"stride": [2, 2], "padding": [1, 1]})
    w.add_output("output", [1, 8, -1, -1])
    refuse("stride 2", w, "stride 1")

    w = VVAWriter(name="r", scale=1)
    w.add_input("input", [1, 3, -1, -1])
    w.add_node("Sigmoid", ["input"], ["output"])
    w.add_output("output", [1, 3, -1, -1])
    refuse("Sigmoid", w, "not available")

    w = VVAWriter(name="r", scale=1)
    w.add_input("input", [1, 3, -1, -1])
    w.add_node("ReLU", ["input"], ["output"])
    w.add_output("output", [1, 3, -1, -1])
    refuse("element-wise on 3 channels", w, "multiples of 8")

    w = VVAWriter(name="r", scale=1)
    w.add_input("input", [1, 3, -1, -1])
    w.add_tensor("w", rnd(rng2, 8, 3, 3, 3), "f32")
    w.add_node("Conv2D", ["input", "w"], ["a"], {"padding": [1, 1]})
    w.add_node("Mul", ["a", "a"], ["output"])
    w.add_output("output", [1, 8, -1, -1])
    refuse("Mul of two tensors", w, "scalar")

    print("== engine with a GPU backend: selection, self-test, fallback ==")
    sd = make_rrdbnet_state(16, 8, 1, seed=3)
    model = os.path.join(WORK, "gp_engine.vva")
    convert(save_npz(sd, "gp_engine"), model, name="t", log=lambda *_: None)
    exe2 = os.path.join(WORK, "engine_gpu_test")
    core = ["vva.cpp", "model.cpp", "exec.cpp", "gpu/gpu_plan.cpp", "gpu/gpu_vulkan.cpp"]
    r = run(["g++", "-std=c++17", "-g", "-O1", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined", "-DVVA_NO_VULKAN",
             "-pthread", "-I", os.path.join(ROOT, "core", "include"), "-I", os.path.join(ROOT, "core", "src"),
             "-I", os.path.join(ROOT, "core", "src", "gpu"), "-I", os.path.join(ROOT, "tests"),
             os.path.join(ROOT, "tests", "engine_gpu_test.cpp"), os.path.join(ROOT, "tests", "gpu_emu.cpp")]
            + [os.path.join(ROOT, "core", "src", f) for f in core] + ["-o", exe2])
    T.check(r.returncode == 0, "engine_gpu_test builds: " + r.stderr[-300:])
    if r.returncode == 0:
        r = run([exe2, model], env=ENV)
        print(r.stdout.rstrip())
        T.check(r.returncode == 0, "engine_gpu_test passes (exit %d) %s" % (r.returncode, (r.stdout + r.stderr)[-400:]))

    return T.finish()


if __name__ == "__main__":
    sys.exit(main())
