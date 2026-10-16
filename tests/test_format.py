#!/usr/bin/env python3
"""Round-trip and robustness tests for the .vva v2 container.

Python writes files, the production C++ reader (built here with sanitizers when
available) reads them, and the results are cross-checked. Malformed files must be
rejected with a message and must never crash the reader.

Run from the repository root:  python3 tests/test_format.py
"""

import importlib.util
import os
import random
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location("vva_fileformat", os.path.join(ROOT, "python", "vva", "fileformat.py"))
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)

WORK = tempfile.mkdtemp(prefix="vva_test_")
FAILURES = []
CHECKS = 0


def check(condition, description):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(description)
        print("  FAIL:", description)


def build_reader():
    exe = os.path.join(WORK, "read_test")
    base = ["g++", "-std=c++17", "-g", "-O1", "-I", os.path.join(ROOT, "core", "src"),
            os.path.join(ROOT, "tests", "read_test.cpp"), os.path.join(ROOT, "core", "src", "model.cpp"), "-o", exe]
    san = base + ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined"]
    result = subprocess.run(san, capture_output=True, text=True)
    if result.returncode == 0:
        print("reader built with ASan+UBSan")
        return exe, True
    print("sanitizers unavailable, building without them:\n", result.stderr[-300:])
    result = subprocess.run(base, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit("cannot build the C++ reader")
    return exe, False


READER, SANITIZED = build_reader()
ENV = dict(os.environ, ASAN_OPTIONS="exitcode=99:detect_leaks=0", UBSAN_OPTIONS="halt_on_error=1:exitcode=98")


def run_reader(path):
    result = subprocess.run([READER, path], capture_output=True, text=True, env=ENV, timeout=60)
    return result.returncode, result.stdout, result.stderr


def sample_writer():
    """A small RRDB-style graph: conv, leaky relu, concat, scalar mul, residual add, nearest upsample."""
    rng = np.random.default_rng(1234)
    w = ff.VVAWriter(name="test-rrdb", scale=4, author="tests", license="BSD-3-Clause")
    w.add_input("input", [1, 3, -1, -1])
    w.add_tensor("conv_first.weight", rng.normal(0, 0.1, (8, 3, 3, 3)), "f16")
    w.add_tensor("conv_first.bias", rng.normal(0, 0.1, (8,)), "f32")
    w.add_tensor("conv1.weight", rng.normal(0, 0.1, (4, 8, 3, 3)), "f16")
    w.add_tensor("conv1.bias", rng.normal(0, 0.1, (4,)), "f32")
    w.add_tensor("conv_last.weight", rng.normal(0, 0.1, (3, 12, 3, 3)), "f16")
    w.add_tensor("conv_last.bias", rng.normal(0, 0.1, (3,)), "f32")
    conv = {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}
    w.add_node("Conv2D", ["input", "conv_first.weight", "conv_first.bias"], ["f0"], conv, "conv_first")
    w.add_node("LeakyReLU", ["f0"], ["f1"], {"alpha": 0.2})
    w.add_node("Conv2D", ["f1", "conv1.weight", "conv1.bias"], ["f2"], conv)
    w.add_node("Concat", ["f1", "f2"], ["f3"], {"axis": 1})
    w.add_node("Mul", ["f3"], ["f4"], {"scalar": 0.2})
    w.add_node("Add", ["f4", "f3"], ["f5"])
    w.add_node("ResizeNearest", ["f5"], ["f6"], {"scale": [2, 2]})
    w.add_node("Conv2D", ["f6", "conv_last.weight", "conv_last.bias"], ["output"], conv)
    w.add_output("output", [1, 3, -1, -1])
    return w


def parse_summary(stdout):
    info = {"nodes_list": [], "tensor_map": {}}
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ("scale", "inputs", "outputs", "nodes", "tensors") and len(parts) == 2:
            info[parts[0] + "_count" if parts[0] != "scale" else "scale"] = int(parts[1])
        elif parts[0] == "name":
            info["name"] = " ".join(parts[1:])
        elif parts[0] == "node":
            info["nodes_list"].append(parts[1])
        elif parts[0] == "tensor":
            info["tensor_map"][parts[1]] = {"dtype": parts[2], "offset": int(parts[3]), "size": int(parts[4]),
                                         "crc": int(parts[5], 16), "aligned": parts[6] == "aligned=1"}
    return info


def write(name, data):
    path = os.path.join(WORK, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def expect_rejected(name, data, message_part):
    path = write(name, data)
    code, out, err = run_reader(path)
    check(code == 1, "%s: reader must reject (exit 1), got %d %s" % (name, code, err[-200:]))
    check(message_part in out, "%s: expected '%s' in output, got: %s" % (name, message_part, out.strip()))


def raw_file(meta_bytes, data=b"", header_overrides=None):
    """Build a file image from raw metadata bytes (valid checksum), optionally patching header fields."""
    meta_offset = ff.HEADER_SIZE
    data_offset = (meta_offset + len(meta_bytes) + 63) // 64 * 64
    pad = b"\x00" * (data_offset - meta_offset - len(meta_bytes))
    fields = dict(meta_offset=meta_offset, meta_size=len(meta_bytes), data_offset=data_offset,
                  data_size=len(data), file_size=data_offset + len(data), crc=ff.crc32(meta_bytes))
    fields.update(header_overrides or {})
    header = ff._HEADER.pack(ff.MAGIC, 64, 0, 0, fields["meta_offset"], fields["meta_size"], fields["data_offset"],
                             fields["data_size"], fields["file_size"], fields["crc"], 0)
    return header + meta_bytes + pad + data


def main():
    import json

    print("== positive: Python writer -> C++ reader ==")
    writer = sample_writer()
    good = writer.to_bytes()
    good_path = write("good.vva", good)
    code, out, err = run_reader(good_path)
    check(code == 0, "good file must load (exit 0), got %d: %s %s" % (code, out.strip(), err[-200:]))
    info = parse_summary(out)
    meta = writer.to_meta()
    check(info.get("scale") == 4, "scale is read back")
    check(info.get("name") == "test-rrdb", "name is read back")
    check(info.get("nodes_count") == len(meta["nodes"]), "node count matches")
    check(info.get("tensors_count") == len(meta["tensors"]), "tensor count matches")
    check(info.get("inputs_count") == 1 and info.get("outputs_count") == 1, "input/output counts")
    check(info["nodes_list"] == [n["op"] for n in meta["nodes"]], "node order and ops match")
    _, py_tensors = ff.read_vva(good)
    for t in meta["tensors"]:
        got = info["tensor_map"].get(t["name"])
        check(got is not None, "tensor %s present in C++ reader output" % t["name"])
        if got:
            expected_crc = zlib.crc32(py_tensors[t["name"]].tobytes()) & 0xFFFFFFFF
            check(got["crc"] == expected_crc, "tensor %s bytes match (crc)" % t["name"])
            check(got["aligned"], "tensor %s is 64-byte aligned in memory" % t["name"])
            check(got["dtype"] == t["dtype"] and got["size"] == t["size"], "tensor %s dtype/size" % t["name"])

    print("== positive: Python reader round trip ==")
    py_meta, arrays = ff.read_vva(good)
    check(py_meta["meta"]["scale"] == 4, "python reader: scale")
    rng = np.random.default_rng(1234)
    ref = rng.normal(0, 0.1, (8, 3, 3, 3)).astype(np.float16)
    check(np.array_equal(arrays["conv_first.weight"], ref), "python reader: f16 weights round trip exactly")
    check(arrays["conv_first.bias"].dtype == np.float32, "python reader: f32 bias dtype")

    print("== writer-side validation ==")
    bad = ff.VVAWriter()
    bad.add_input("input", [1, 3, -1, -1])
    bad.add_node("Add", ["input", "missing"], ["output"])
    bad.add_output("output", [1, 3, -1, -1])
    try:
        bad.to_bytes()
        check(False, "writer must reject undefined inputs")
    except ff.VVAFormatError as e:
        check("not defined" in str(e), "writer rejects undefined inputs")
    try:
        ff.VVAWriter().add_tensor("big", np.array([1e6], dtype=np.float32), "f16")
        check(False, "writer must reject fp16 overflow")
    except ff.VVAFormatError as e:
        check("fp16" in str(e), "writer rejects fp16 overflow")

    print("== negative: structural problems ==")
    expect_rejected("truncated.vva", good[:-10], "file size mismatch")
    expect_rejected("trailing.vva", good + b"\x00", "file size mismatch")
    expect_rejected("magic.vva", b"XXXX" + good[4:], "bad magic")
    expect_rejected("v1.vva", b"VVA1" + good[4:], "legacy v1")
    expect_rejected("tiny.vva", b"VVA2", "too small")
    corrupted = bytearray(good)
    corrupted[ff.HEADER_SIZE + 20] ^= 0xFF
    expect_rejected("crc.vva", bytes(corrupted), "checksum mismatch")
    meta_bytes = good[ff.HEADER_SIZE:ff.HEADER_SIZE + struct.unpack("<Q", good[24:32])[0]]
    blob = good[struct.unpack("<Q", good[32:40])[0]:]
    expect_rejected("meta_off.vva", raw_file(meta_bytes, blob, {"meta_offset": 2 ** 40}), "metadata block is out of range")
    expect_rejected("data_misaligned.vva", raw_file(meta_bytes, blob, {"data_offset": 130}), "out of range or misaligned")
    total_len = len(raw_file(meta_bytes, blob))
    expect_rejected("overlap.vva", raw_file(meta_bytes, blob, {"data_offset": 64, "data_size": total_len - 64}),
                    "overlap")

    print("== negative: metadata problems (valid checksum) ==")

    def variant(mutator):
        m = json.loads(json.dumps(writer.to_meta()))
        mutator(m)
        return raw_file(json.dumps(m, separators=(",", ":")).encode(), bytes(writer._blob))

    expect_rejected("version.vva", variant(lambda m: m.update(version=3)), "unsupported metadata version")
    expect_rejected("scale0.vva", variant(lambda m: m["meta"].update(scale=0)), "meta.scale")
    expect_rejected("dtype.vva", variant(lambda m: m["tensors"][0].update(dtype="i8")), "unsupported dtype")
    expect_rejected("tsize.vva", variant(lambda m: m["tensors"][0].update(size=7)), "does not match")
    expect_rejected("toff.vva", variant(lambda m: m["tensors"][0].update(offset=10 ** 9)), "outside the data block")
    expect_rejected("tmis.vva", variant(lambda m: m["tensors"][1].update(offset=m["tensors"][1]["offset"] + 32)),
                    "outside the data block or is misaligned")
    expect_rejected("tdup.vva", variant(lambda m: m["tensors"][1].update(name="conv_first.weight")), "duplicate")
    expect_rejected("undefined.vva", variant(lambda m: m["nodes"][1].update(inputs=["nope"])), "not defined")
    expect_rejected("redefined.vva", variant(lambda m: m["nodes"][2].update(outputs=["f0"])), "more than once")
    expect_rejected("order.vva", variant(lambda m: m["nodes"].reverse()), "not defined")
    expect_rejected("noout.vva", variant(lambda m: m["outputs"][0].update(name="ghost")), "not produced")
    expect_rejected("nonodes.vva", variant(lambda m: m.update(nodes=[])), "'nodes' must be a non-empty array")
    expect_rejected("badop.vva", variant(lambda m: m["nodes"][0].update(op="")), "node #0")
    expect_rejected("badattrs.vva", variant(lambda m: m["nodes"][0].update(attrs=[1])), "non-object")
    expect_rejected("badjson.vva", raw_file(b'{"version":', b""), "invalid metadata")
    expect_rejected("deep.vva", raw_file(b"[" * 2000, b""), "nesting is too deep")
    expect_rejected("notobj.vva", raw_file(b"[1,2,3]", b""), "must be a JSON object")
    expect_rejected("trailing_json.vva", raw_file(b'{"version":2} x', b""), "unexpected characters")
    expect_rejected("badescape.vva", raw_file(b'{"a":"\\q"}', b""), "invalid escape")
    expect_rejected("surrogate.vva", raw_file(b'{"a":"\\ud800"}', b""), "surrogate")

    print("== fuzz: random garbage and mutated files must never crash ==")
    rnd = random.Random(7)
    crashes = 0
    cases = 0
    meta_len = len(meta_bytes)
    for i in range(400):
        mode = i % 4
        if mode == 0:
            data = bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 300)))
        elif mode == 1:
            data = bytearray(good)
            for _ in range(rnd.randint(1, 6)):
                data[rnd.randrange(0, ff.HEADER_SIZE)] = rnd.getrandbits(8)
            data = bytes(data)
        elif mode == 2:
            mutated = bytearray(meta_bytes)
            for _ in range(rnd.randint(1, 4)):
                mutated[rnd.randrange(meta_len)] = rnd.choice(b'{}[]",:0123456789-eE.\\ntfu')
            data = raw_file(bytes(mutated), blob)
        else:
            cut = rnd.randint(0, meta_len)
            data = raw_file(meta_bytes[:cut], blob)
        path = write("fuzz.vva", data)
        code, out, err = run_reader(path)
        cases += 1
        if code not in (0, 1):
            crashes += 1
            print("  crash/sanitizer failure on fuzz case", i, "exit", code, err[-300:])
    check(crashes == 0, "fuzz: %d/%d cases crashed the reader" % (crashes, cases))

    print("\n%d checks, %d failed" % (CHECKS, len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
