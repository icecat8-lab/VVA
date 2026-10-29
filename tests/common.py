"""Shared helpers for the VVA tests (no PyTorch needed)."""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from vva.fileformat import VVAWriter, read_vva  # noqa: E402
from vva.reference import rrdbnet_forward  # noqa: E402

WORK = tempfile.mkdtemp(prefix="vva_test_")


class Checker:
    def __init__(self, title):
        self.title = title
        self.count = 0
        self.failures = []

    def check(self, condition, description):
        self.count += 1
        if not condition:
            self.failures.append(description)
            print("  FAIL:", description)

    def finish(self):
        print("%s: %d checks, %d failed" % (self.title, self.count, len(self.failures)))
        return 1 if self.failures else 0


def make_rrdbnet_state(nf, gc, nb, seed=0, out_std=0.15):
    """Random but well-behaved basicsr-style RRDBNet weights (float32)."""
    rng = np.random.RandomState(seed)
    sd = {}

    def conv(name, out_c, in_c, k=3, std=None):
        fan = in_c * k * k
        sd[name + ".weight"] = (rng.randn(out_c, in_c, k, k) * (std or 0.7 / np.sqrt(fan))).astype(np.float32)
        sd[name + ".bias"] = (rng.randn(out_c) * 0.05).astype(np.float32)

    conv("conv_first", nf, 3)
    for i in range(nb):
        for r in (1, 2, 3):
            for j in range(1, 6):
                conv("body.%d.rdb%d.conv%d" % (i, r, j), gc if j < 5 else nf, nf + (j - 1) * gc)
    for name in ("conv_body", "conv_up1", "conv_up2", "conv_hr"):
        conv(name, nf, nf)
    conv("conv_last", 3, nf)
    # Sprinkle tiny values so fp16 subnormals are exercised too.
    for key in ("conv_first.weight", "conv_hr.weight"):
        flat = sd[key].reshape(-1)
        flat[:4] = [1e-6, -3e-5, 5.9e-5, 6.2e-5]
    # Centre the output around 0.5 with a visible spread so 8-bit results are informative.
    x = np.random.RandomState(seed + 1).rand(3, 6, 7)
    probe = rrdbnet_forward(sd, x)
    sd["conv_last.weight"] *= np.float32(out_std / max(float(np.std(probe)), 1e-6))
    sd["conv_last.bias"][:] = 0.5
    return sd


def save_npz(sd, name):
    path = os.path.join(WORK, name)
    np.savez(path, **sd)
    return path if path.endswith(".npz") else path + ".npz"


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def build_binaries(sanitize=True):
    """Compile the production C++ sources for the host. Returns {name: path}."""
    flags = ["-std=c++17", "-g", "-O1" if sanitize else "-O2", "-DVVA_NO_VULKAN", "-pthread",
             "-I", os.path.join(ROOT, "core", "include"), "-I", os.path.join(ROOT, "core", "src")]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"]
    core = [os.path.join(ROOT, "core", "src", f) for f in ("vva.cpp", "model.cpp", "exec.cpp")]
    objs = []
    for src in core:
        obj = os.path.join(WORK, os.path.basename(src) + (".san.o" if sanitize else ".o"))
        result = run(["g++"] + flags + ["-c", src, "-o", obj])
        if result.returncode != 0:
            print(result.stderr)
            sys.exit("cannot compile " + src)
        objs.append(obj)
    exes = {}
    for name, src in (("exec_run", "tests/exec_run.cpp"), ("engine_test", "tests/engine_test.cpp"),
                      ("vva-cli", "cli/main.cpp")):
        exe = os.path.join(WORK, name + (".san" if sanitize else ""))
        result = run(["g++"] + flags + [os.path.join(ROOT, src)] + objs + ["-o", exe])
        if result.returncode != 0:
            print(result.stderr)
            sys.exit("cannot build " + name)
        exes[name] = exe
    return exes


ENV = dict(os.environ, ASAN_OPTIONS="exitcode=99:detect_leaks=0", UBSAN_OPTIONS="halt_on_error=1:exitcode=98")


def exec_run(exe, model, x, threads=1):
    """Run the production executor on x [C,H,W]; returns (status_text, output array or None)."""
    c, h, w = x.shape
    inp = os.path.join(WORK, "in.f32")
    out = os.path.join(WORK, "out.f32")
    np.ascontiguousarray(x, dtype=np.float32).tofile(inp)
    if os.path.exists(out):
        os.remove(out)
    r = run([exe, model, inp, str(c), str(h), str(w), out, str(threads)], env=ENV)
    text = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        return text, None
    return text, np.fromfile(out, dtype=np.float32)


def make_mobilenetv3_small_state(n_classes=2, seed=0, bn_eps=1e-3):
    """Random but well-behaved torchvision MobileNetV3-Small weights (float32), BatchNorm statistics included."""
    from vva.reference import MBV3_SMALL, mobilenetv3_small_forward
    rng = np.random.RandomState(seed)
    sd = {}

    def conv(name, out_c, in_c, k, groups=1, bias=False):
        fan = (in_c // groups) * k * k
        sd[name + ".weight"] = (rng.randn(out_c, in_c // groups, k, k) * np.sqrt(2.0 / fan)).astype(np.float32)
        if bias:
            sd[name + ".bias"] = (rng.randn(out_c) * 0.05).astype(np.float32)

    def bn(name, c):
        sd[name + ".weight"] = rng.uniform(0.8, 1.2, c).astype(np.float32)
        sd[name + ".bias"] = (rng.randn(c) * 0.1).astype(np.float32)
        sd[name + ".running_mean"] = (rng.randn(c) * 0.1).astype(np.float32)
        sd[name + ".running_var"] = rng.uniform(0.5, 2.0, c).astype(np.float32)
        sd[name + ".num_batches_tracked"] = np.array(100.0, dtype=np.float32)

    conv("features.0.0", 16, 3, 3)
    bn("features.0.1", 16)
    for i, (cin, k, exp, cout, se, _a, _s) in enumerate(MBV3_SMALL, start=1):
        p = "features.%d.block" % i
        j = 0
        if exp != cin:
            conv("%s.%d.0" % (p, j), exp, cin, 1)
            bn("%s.%d.1" % (p, j), exp)
            j += 1
        conv("%s.%d.0" % (p, j), exp, exp, k, groups=exp)
        bn("%s.%d.1" % (p, j), exp)
        j += 1
        if se:
            sq = max(8, int(exp // 4 + 4) // 8 * 8)
            conv("%s.%d.fc1" % (p, j), sq, exp, 1, bias=True)
            conv("%s.%d.fc2" % (p, j), exp, sq, 1, bias=True)
            j += 1
        conv("%s.%d.0" % (p, j), cout, exp, 1)
        bn("%s.%d.1" % (p, j), cout)
    conv("features.12.0", 576, 96, 1)
    bn("features.12.1", 576)
    sd["classifier.0.weight"] = (rng.randn(1024, 576) * np.sqrt(2.0 / 576)).astype(np.float32)
    sd["classifier.0.bias"] = (rng.randn(1024) * 0.05).astype(np.float32)
    sd["classifier.3.weight"] = (rng.randn(n_classes, 1024) * np.sqrt(1.0 / 1024)).astype(np.float32)
    sd["classifier.3.bias"] = (rng.randn(n_classes) * 0.1).astype(np.float32)
    # Scale the final layer so the logits have a spread of about one (probabilities stay informative).
    x = np.random.RandomState(seed + 1).randn(3, 64, 64)
    probe = mobilenetv3_small_forward({k: np.asarray(v, dtype=np.float64) for k, v in sd.items()}, x, bn_eps)
    sd["classifier.3.weight"] *= np.float32(1.5 / max(float(np.std(probe)), 1e-3))
    return sd
