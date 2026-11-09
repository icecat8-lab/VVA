#!/usr/bin/env python3

import os
import stat
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Checker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gen_spirv

T = Checker("test_gen_spirv")

def fake_glslc(directory, fail_on=()):
    path = os.path.join(directory, "glslc")
    words = struct.pack("<8I", 0x07230203, 0x10300, 0, 1, 0, 1, 2, 3)
    with open(os.path.join(directory, "blob.spv"), "wb") as f:
        f.write(words)
    cases = "\n".join('  *%s) echo "error: simulated compile error in %s" >&2; exit 1;;' % (n, n) for n in fail_on)
    with open(path, "w") as f:
        f.write("#!/bin/sh\nfor last; do :; done\nout=\"\"\nwhile [ $# -gt 0 ]; do [ \"$1\" = \"-o\" ] && out=\"$2\"; src=\"$1\"; shift; done\n"
                "case \"$src\" in\n%s\n  *) cp \"%s\" \"$out\";;\nesac\n" % (cases, os.path.join(directory, "blob.spv")))
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path

def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "spirv_data.h")

        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_spirv.py"), "--glslc", fake_glslc(tmp), "--out", out],
                           capture_output=True, text=True)
        T.check(r.returncode == 0, "all shaders compile: " + r.stderr[-200:])
        header = open(out).read()
        for name, _, macro in gen_spirv.SHADERS:
            T.check("static const uint32_t %s[]" % name in header, "header holds %s" % name)
        T.check("#define VVA_SPV_CONV4OC 1" in header and "#define VVA_SPV_CONV8PX 1" in header, "both optional variants are marked present")

        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_spirv.py"), "--glslc",
                            fake_glslc(tmp, fail_on=("conv3x3_4oc.comp",)), "--out", out], capture_output=True, text=True)
        T.check(r.returncode == 0, "an optional shader may fail: " + r.stderr[-200:])
        header = open(out).read()
        T.check("#define VVA_SPV_CONV4OC 0" in header and "#define VVA_SPV_CONV8PX 1" in header, "only the failed variant is marked missing")
        T.check("simulated compile error" in r.stderr and "leaving out" in r.stderr, "the compiler message is shown")

        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_spirv.py"), "--glslc",
                            fake_glslc(tmp, fail_on=("eltwise.comp",)), "--out", os.path.join(tmp, "x.h")], capture_output=True, text=True)
        T.check(r.returncode != 0 and "required shader" in r.stderr, "a required shader failing stops the build")
        T.check(not os.path.exists(os.path.join(tmp, "x.h")), "no header is written then")
    return T.finish()

if __name__ == "__main__":
    sys.exit(main())
