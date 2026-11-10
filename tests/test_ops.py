#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker, WORK, build_binaries, exec_run, VVAWriter, read_vva
from vva.reference import run_graph

T = Checker("test_ops")
rng = np.random.RandomState(5)


def rnd(*shape, scale=0.5):
    return (rng.randn(*shape) * scale).astype(np.float32)


def writer(scale=1, channels=3):
    w = VVAWriter(name="ops", scale=scale)
    w.add_input("input", [1, channels, -1, -1])
    return w


def head(w, out_c=6, k=3, pad=1):
    w.add_tensor("hw", rnd(out_c, w._inputs[0]["shape"][1], k, k, scale=0.3), "f32")
    w.add_tensor("hb", rnd(out_c), "f32")
    w.add_node("Conv2D", ["input", "hw", "hb"], ["h"], {"padding": [pad, pad]})
    return "h"


def finish(w, last, out_c):
    w.add_output(last, [1, out_c, -1, -1])
    return w


def g_activations():
    w = writer()
    h = head(w, 6)
    w.add_tensor("slope", rnd(6, 1, 1, scale=0.3), "f32")
    w.add_node("PReLU", [h, "slope"], ["p"])
    w.add_tensor("slope1", np.array([0.25], dtype=np.float32), "f32")
    w.add_node("PReLU", [h, "slope1"], ["p1"])
    w.add_node("SiLU", ["p"], ["s"])
    w.add_node("Tanh", ["p1"], ["t"])
    w.add_node("Sub", ["s", "t"], ["d"])
    w.add_tensor("scale", rnd(6, scale=1.0), "f32")
    w.add_tensor("shift", rnd(6), "f32")
    w.add_node("ChannelAffine", ["d", "scale", "shift"], ["a"])
    w.add_node("Sub", ["a", "d"], ["out"])
    return finish(w, "out", 6)


def g_convt(k, s, p, op, kh=None, kw=None, stride=None, pad=None, outp=None):
    def build():
        w = writer()
        h = head(w, 4)
        w.add_tensor("tw", rnd(4, 5, kh or k, kw or k, scale=0.3), "f32")
        w.add_tensor("tb", rnd(5), "f32")
        attrs = {"stride": stride or [s, s], "padding": pad or [p, p], "output_padding": outp or [op, op]}
        w.add_node("ConvTranspose2D", [h, "tw", "tb"], ["t"], attrs)
        w.add_node("ReLU", ["t"], ["out"])
        return finish(w, "out", 5)
    return build


def g_pool(kind, k, s, p, include=False):
    def build():
        w = writer()
        h = head(w, 4)
        w.add_node(kind, [h], ["out"], {"kernel": [k, k], "stride": [s, s], "padding": [p, p], "count_include_pad": include})
        return finish(w, "out", 4)
    return build


def g_bilinear(sh, sw, align):
    def build():
        w = writer()
        h = head(w, 4)
        w.add_node("ResizeBilinear", [h], ["out"], {"scale": [sh, sw], "align_corners": align})
        return finish(w, "out", 4)
    return build


def g_pad(pads, mode, value=None):
    def build():
        w = writer()
        h = head(w, 4)
        attrs = {"pads": pads, "mode": mode}
        if value is not None:
            attrs["value"] = value
        w.add_node("Pad", [h], ["pd"], attrs)
        w.add_tensor("cw", rnd(3, 4, 3, 3, scale=0.3), "f32")
        w.add_node("Conv2D", ["pd", "cw"], ["out"], {})
        return finish(w, "out", 3)
    return build


def g_unshuffle(r):
    def build():
        w = writer()
        w.add_node("PixelUnshuffle", ["input"], ["u"], {"downscale": r})
        w.add_tensor("cw", rnd(4, 3 * r * r, 3, 3, scale=0.2), "f32")
        w.add_node("Conv2D", ["u", "cw"], ["c"], {"padding": [1, 1]})
        w.add_node("PixelShuffle", ["c"], ["out"], {"upscale": 2})
        return finish(w, "out", 1)
    return build


def g_slice():
    w = writer()
    h = head(w, 8)
    w.add_node("Slice", [h], ["a"], {"start": 0, "end": 3})
    w.add_node("Slice", [h], ["b"], {"start": 5, "end": 8})
    w.add_node("Add", ["a", "b"], ["s"])
    w.add_node("Slice", ["s"], ["out"], {"start": 1, "end": 3})
    return finish(w, "out", 2)


def g_inplace_chain():
    w = writer()
    h = head(w, 5)
    w.add_tensor("slope", rnd(5, scale=0.3), "f32")
    w.add_node("PReLU", [h, "slope"], ["p"])
    w.add_node("Tanh", ["p"], ["t"])
    w.add_node("SiLU", ["t"], ["s"])
    w.add_node("Sub", ["s", "t"], ["d"])
    w.add_node("Sub", ["d", "d"], ["z"])
    w.add_node("Add", ["z", "s"], ["out"])
    return finish(w, "out", 5)


def run_case(name, builder, sizes, exe):
    path = os.path.join(WORK, "ops_%s.vva" % name)
    builder().save(path)
    meta, stored = read_vva(path)
    deq = {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()}
    for (h, w) in sizes:
        x = np.random.RandomState(h * 31 + w).rand(3, h, w).astype(np.float32)
        want = run_graph(meta, deq, x.astype(np.float64)).reshape(-1)
        text, got = exec_run(exe, path, x, 1, any_output=True)
        T.check(got is not None, "%s %dx%d runs (%s)" % (name, h, w, text))
        if got is None:
            continue
        err = float(np.max(np.abs(got.astype(np.float64) - want)) / max(float(np.max(np.abs(want))), 1e-6)) if got.size == want.size else -1
        T.check(got.size == want.size and err < 3e-5, "%s %dx%d matches numpy (max rel err %.2e)" % (name, h, w, err))
        _, got3 = exec_run(exe, path, x, 3, any_output=True)
        T.check(got3 is not None and np.array_equal(got, got3), "%s %dx%d: threads do not change the result" % (name, h, w))


def expect_fail(name, builder, size, exe, needle):
    path = os.path.join(WORK, "bad_%s.vva" % name)
    builder().save(path)
    x = np.random.RandomState(1).rand(3, *size).astype(np.float32)
    text, got = exec_run(exe, path, x, 1, any_output=True)
    T.check(got is None and needle in text, "%s is rejected with a clear message (%s)" % (name, text[:90]))


def main():
    exe = build_binaries(sanitize=os.environ.get("VVA_NO_SAN") is None)["exec_run"]
    sizes = [(7, 9), (1, 1), (6, 13)]
    cases = [
        ("activations", g_activations, sizes),
        ("convt_k2s2", g_convt(2, 2, 0, 0), sizes),
        ("convt_k4s2p1", g_convt(4, 2, 1, 0), sizes),
        ("convt_k3s2p1op1", g_convt(3, 2, 1, 1), sizes),
        ("convt_k3s3", g_convt(3, 3, 0, 0), sizes),
        ("convt_aniso", g_convt(3, 2, 0, 0, kh=2, kw=4, stride=[2, 3], pad=[0, 1], outp=[1, 0]), sizes),
        ("maxpool_k2s2", g_pool("MaxPool2D", 2, 2, 0), [(8, 10), (7, 9)]),
        ("maxpool_k3s2p1", g_pool("MaxPool2D", 3, 2, 1), [(8, 10), (7, 9), (3, 3)]),
        ("avgpool_excl", g_pool("AvgPool2D", 3, 1, 1, False), [(8, 10), (3, 5)]),
        ("avgpool_incl", g_pool("AvgPool2D", 3, 2, 1, True), [(8, 10), (5, 5)]),
        ("avgpool_k2", g_pool("AvgPool2D", 2, 2, 0), [(8, 10), (7, 9)]),
        ("bilinear_2x", g_bilinear(2, 2, False), sizes),
        ("bilinear_3x_align", g_bilinear(3, 3, True), sizes),
        ("bilinear_aniso", g_bilinear(2, 3, False), sizes),
        ("pad_const", g_pad([1, 2, 3, 0], "constant", 0.5), sizes),
        ("pad_reflect", g_pad([2, 1, 1, 2], "reflect"), [(7, 9), (6, 13)]),
        ("pad_replicate", g_pad([3, 2, 1, 4], "replicate"), sizes),
        ("unshuffle2", g_unshuffle(2), [(8, 10), (4, 2), (2, 2)]),
        ("unshuffle4", g_unshuffle(4), [(8, 12), (4, 4)]),
        ("slice", g_slice, sizes),
        ("inplace_chain", g_inplace_chain, sizes),
    ]
    for name, builder, sz in cases:
        run_case(name, builder, sz, exe)

    expect_fail("unshuffle_odd_size", g_unshuffle(2), (7, 8), exe, "cannot run")
    expect_fail("pad_reflect_too_big", g_pad([4, 0, 0, 0], "reflect"), (3, 3), exe, "cannot run")

    def bad_pool():
        w = writer()
        h = head(w, 4)
        w.add_node("MaxPool2D", [h], ["out"], {"kernel": [2, 2], "padding": [2, 2]})
        return finish(w, "out", 4)
    expect_fail("pool_pad", bad_pool, (6, 6), exe, "padding")

    def bad_convt():
        w = writer()
        h = head(w, 4)
        w.add_tensor("tw", rnd(4, 2, 3, 3), "f32")
        w.add_node("ConvTranspose2D", [h, "tw"], ["out"], {"stride": [2, 2], "output_padding": [2, 2]})
        return finish(w, "out", 2)
    expect_fail("convt_output_padding", bad_convt, (6, 6), exe, "output_padding")

    def bad_slice():
        w = writer()
        h = head(w, 4)
        w.add_node("Slice", [h], ["out"], {"start": 2, "end": 9})
        return finish(w, "out", 7)
    expect_fail("slice_range", bad_slice, (6, 6), exe, "range")

    def bad_prelu():
        w = writer()
        h = head(w, 4)
        w.add_tensor("slope", rnd(3), "f32")
        w.add_node("PReLU", [h, "slope"], ["out"])
        return finish(w, "out", 4)
    expect_fail("prelu_length", bad_prelu, (6, 6), exe, "channel")
    return T.finish()


if __name__ == "__main__":
    sys.exit(main())
