#!/usr/bin/env python3
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find(path, pattern):
    text = open(os.path.join(ROOT, path)).read()
    m = re.search(pattern, text)
    return m.group(1) if m else None


versions = {
    "CMakeLists.txt": find("CMakeLists.txt", r"project\(VVA VERSION ([0-9.]+)"),
    "python/setup.py": find("python/setup.py", r'version="([0-9.]+)"'),
    "core/include/vva/version.h": find("core/include/vva/version.h", r'VVA_ENGINE_VERSION "([0-9.]+)"'),
    ".github/workflows/build.yml": (find(".github/workflows/build.yml", r"default: 'v([0-9.]+)'")),
}
problems = ["%s has no version" % k for k, v in versions.items() if v is None]
if len(set(v for v in versions.values() if v)) > 1:
    problems.append("version numbers disagree: " + ", ".join("%s=%s" % kv for kv in versions.items()))
for p in problems:
    print("FAIL:", p)
print("check_version: %s, %d problems" % (", ".join("%s" % v for v in versions.values()), len(problems)))
sys.exit(1 if problems else 0)
