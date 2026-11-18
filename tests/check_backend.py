#!/usr/bin/env python3
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU = os.path.join(ROOT, "core", "src", "gpu")

header = open(os.path.join(GPU, "gpu_backend.h")).read()
source = open(os.path.join(GPU, "gpu_vulkan.cpp")).read()

pure = re.findall(r"virtual\s+[^;=~]*?\b(\w+)\s*\([^;]*\)\s*(?:const)?\s*=\s*0\s*;", header)
problems = []
for name in pure:
    if not re.search(r"\b%s\s*\([^)]*\)\s*(?:const)?\s*override" % name, source):
        problems.append("VulkanBackend does not override %s()" % name)

for token in ("{", "("):
    close = "}" if token == "{" else ")"
    if source.count(token) != source.count(close):
        problems.append("unbalanced %s%s in gpu_vulkan.cpp" % (token, close))

for p in problems:
    print("FAIL:", p)
print("check_backend: %d interface methods checked, %d problems" % (len(pure), len(problems)))
sys.exit(1 if problems else 0)
