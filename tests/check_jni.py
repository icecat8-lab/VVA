#!/usr/bin/env python3
"""Checks that every `native` method in VVAEngine.java has a matching JNI function in vva_jni.cpp
(same name, same parameter and return types). A mismatch only shows up at run time on a phone
(UnsatisfiedLinkError or a crash), so it is checked here at build time.

Run from the repository root:  python3 tests/check_jni.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAVA = os.path.join(ROOT, "android/vva-android/src/main/java/com/vva/engine/VVAEngine.java")
CPP = os.path.join(ROOT, "android/vva-android/src/main/cpp/vva_jni.cpp")

JAVA_TO_JNI = {"long": "jlong", "int": "jint", "boolean": "jboolean", "float": "jfloat", "void": "void",
               "String": "jstring"}


def jni_type(java_type):
    return JAVA_TO_JNI.get(java_type, "jobject")


def main():
    java = open(JAVA).read()
    cpp = open(CPP).read()
    declared = {}
    for m in re.finditer(r"private\s+static\s+native\s+(\w+)\s+(\w+)\(([^)]*)\)\s*;", java):
        ret, name, params = m.groups()
        types = [p.strip().split()[0] for p in params.split(",") if p.strip()]
        declared[name] = (jni_type(ret), [jni_type(t) for t in types])
    implemented = {}
    for m in re.finditer(r"JNIEXPORT\s+(\w+)\s+JNICALL\s+Java_com_vva_engine_VVAEngine_(\w+)\(([^)]*)\)", cpp, re.S):
        ret, name, params = m.groups()
        types = [p.strip().split()[0] for p in params.replace("\n", " ").split(",") if p.strip()]
        implemented[name] = (ret, types[2:])  # drop JNIEnv* and jclass
    problems = []
    for name, sig in sorted(declared.items()):
        if name not in implemented:
            problems.append("Java declares %s but vva_jni.cpp does not implement it" % name)
        elif implemented[name] != sig:
            problems.append("%s: Java expects %s but C++ has %s" % (name, sig, implemented[name]))
    for name in implemented:
        if name not in declared:
            problems.append("vva_jni.cpp implements %s but VVAEngine.java does not declare it" % name)
    if not declared:
        problems.append("no native methods found - the check itself is broken")
    for p in problems:
        print("FAIL:", p)
    print("check_jni: %d native methods, %d problems" % (len(declared), len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
