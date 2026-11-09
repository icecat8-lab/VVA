#!/usr/bin/env python3

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = os.path.join(ROOT, "core", "src", "gpu", "kernels")

RESERVED = set("""
common partition active asm class union enum typedef template this resource goto inline noinline public static
extern external interface long short half fixed unsigned superp input output filter sizeof cast namespace using
sample patch buffer shared coherent volatile restrict uniform varying attribute smooth flat centroid noperspective
subroutine invariant precise lowp mediump highp precision switch case default do while double layout readonly
writeonly in out inout struct discard true false bool uint
sampler1D sampler2D sampler3D samplerCube image1D image2D image3D hvec2 hvec3 hvec4 fvec2 fvec3 fvec4 dvec2 dvec3
dvec4 mat2 mat3 mat4 ivec2 ivec3 ivec4 uvec2 uvec3 uvec4 bvec2 bvec3 bvec4
""".split())

BUILTINS = set("abs sign floor ceil fract mod min max clamp mix step smoothstep sqrt length distance dot cross "
               "normalize pow exp log round trunc barrier main".split())

def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)

def main():
    problems = []
    files = sorted(f for f in os.listdir(KERNELS) if f.endswith(".inc"))
    if not files:
        problems.append("no kernel sources found")
    for name in files:
        text = strip_comments(open(os.path.join(KERNELS, name)).read())
        for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text):
            word = m.group(0)
            if word in RESERVED:
                line = text.count("\n", 0, m.start()) + 1
                problems.append("%s:%d uses the reserved GLSL word '%s'" % (name, line, word))
        for m in re.finditer(r"\b(?:int|float|vec4|void)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[=;(,)]", text):
            if m.group(1) in BUILTINS and m.group(1) != "main":
                problems.append("%s declares '%s', the name of a GLSL built-in" % (name, m.group(1)))
        if "__" in text:
            problems.append("%s contains a double underscore (reserved in GLSL)" % name)
        if re.search(r"\b\d+\.\d*f\b|\b\d+f\b", text):
            problems.append("%s uses a C-style float suffix (not valid GLSL without an extension)" % name)
    for p in problems:
        print("FAIL:", p)
    print("check_glsl: %d kernel files, %d problems" % (len(files), len(problems)))
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
