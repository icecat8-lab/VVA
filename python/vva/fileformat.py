"""Reader and writer for the .vva (version 2) model container.

The format is described in docs/FORMAT.md. This module only needs numpy,
so it can be used on any machine (PC, Colab, Termux) without PyTorch.
"""

import json
import struct
import zlib

import numpy as np

MAGIC = b"VVA2"
VERSION = 2
HEADER_SIZE = 64
ALIGN = 64
_HEADER = struct.Struct("<4sIIIQQQQQII")

_DTYPES = {"f32": np.dtype("<f4"), "f16": np.dtype("<f2")}
_NAMES = {v: k for k, v in _DTYPES.items()}


class VVAFormatError(ValueError):
    """Raised when a .vva file or a model description is invalid."""


def _align_up(value, alignment=ALIGN):
    return (value + alignment - 1) // alignment * alignment


def crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF


def pack(meta, data):
    """Serialize a metadata dict and a data blob into a .vva file image."""
    meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    meta_offset = HEADER_SIZE
    data_offset = _align_up(meta_offset + len(meta_bytes))
    padding = b"\x00" * (data_offset - meta_offset - len(meta_bytes))
    file_size = data_offset + len(data)
    header = _HEADER.pack(
        MAGIC, HEADER_SIZE, 0, 0,
        meta_offset, len(meta_bytes),
        data_offset, len(data),
        file_size, crc32(meta_bytes), 0,
    )
    return header + meta_bytes + padding + bytes(data)


def _validate_graph(meta):
    """Same structural rules as the C++ reader (single assignment, topological order)."""
    inputs = meta.get("inputs") or []
    outputs = meta.get("outputs") or []
    if not inputs or not outputs:
        raise VVAFormatError("a model needs at least one input and one output")
    available = set()
    for io in inputs:
        if io["name"] in available:
            raise VVAFormatError("duplicate input name '%s'" % io["name"])
        available.add(io["name"])
    for t in meta["tensors"]:
        if t["name"] in available:
            raise VVAFormatError("duplicate tensor or input name '%s'" % t["name"])
        available.add(t["name"])
    if not meta["nodes"]:
        raise VVAFormatError("a model needs at least one node")
    for i, node in enumerate(meta["nodes"]):
        for name in node["inputs"]:
            if name not in available:
                raise VVAFormatError("node #%d (%s) uses '%s' which is not defined" % (i, node["op"], name))
        for name in node["outputs"]:
            if name in available:
                raise VVAFormatError("node #%d (%s) defines '%s' more than once" % (i, node["op"], name))
            available.add(name)
    for io in outputs:
        if io["name"] not in available:
            raise VVAFormatError("graph output '%s' is not produced by any node" % io["name"])


class VVAWriter:
    """Builds a .vva v2 model: graph inputs/outputs, constant tensors and nodes."""

    def __init__(self, name="", scale=1, author="", license="", description=""):
        self.meta_info = {"scale": int(scale)}
        for key, value in (("name", name), ("author", author), ("license", license), ("description", description)):
            if value:
                self.meta_info[key] = value
        self._inputs = []
        self._outputs = []
        self._tensors = []
        self._nodes = []
        self._blob = bytearray()
        self._names = set()

    def add_input(self, name, shape):
        self._inputs.append({"name": name, "shape": [int(d) for d in shape]})
        self._names.add(name)
        return name

    def add_output(self, name, shape):
        self._outputs.append({"name": name, "shape": [int(d) for d in shape]})
        return name

    def add_tensor(self, name, array, dtype="f16"):
        """Store a constant tensor. Values are converted to little-endian `dtype`."""
        if dtype not in _DTYPES:
            raise VVAFormatError("unsupported dtype '%s' (use 'f32' or 'f16')" % dtype)
        if name in self._names:
            raise VVAFormatError("duplicate tensor or input name '%s'" % name)
        source = np.asarray(array)
        with np.errstate(over="ignore"):
            converted = np.ascontiguousarray(source, dtype=_DTYPES[dtype])
        if dtype == "f16" and np.all(np.isfinite(source)) and not np.all(np.isfinite(converted)):
            raise VVAFormatError("tensor '%s' has values outside the fp16 range; store it as f32" % name)
        self._blob.extend(b"\x00" * (_align_up(len(self._blob)) - len(self._blob)))
        offset = len(self._blob)
        raw = converted.tobytes()
        self._blob.extend(raw)
        self._tensors.append({
            "name": name, "dtype": dtype, "shape": [int(d) for d in converted.shape],
            "offset": offset, "size": len(raw),
        })
        self._names.add(name)
        return name

    def add_node(self, op, inputs, outputs, attrs=None, name=""):
        node = {"op": op, "inputs": list(inputs), "outputs": list(outputs), "attrs": dict(attrs or {})}
        if name:
            node["name"] = name
        self._nodes.append(node)
        return node

    def to_meta(self):
        return {
            "format": "vva",
            "version": VERSION,
            "meta": self.meta_info,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "tensors": self._tensors,
            "nodes": self._nodes,
        }

    def to_bytes(self, validate=True):
        meta = self.to_meta()
        if validate:
            _validate_graph(meta)
        return pack(meta, self._blob)

    def save(self, path, validate=True):
        data = self.to_bytes(validate=validate)
        with open(path, "wb") as f:
            f.write(data)
        return len(data)


def read_vva(source):
    """Read a .vva v2 file (path or bytes). Returns (meta_dict, {tensor_name: ndarray})."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
    else:
        with open(source, "rb") as f:
            raw = f.read()
    if len(raw) < HEADER_SIZE:
        raise VVAFormatError("file is too small to be a .vva file")
    if raw[:4] == b"VVA1":
        raise VVAFormatError("legacy v1 .vva file; convert the model again with the current converter")
    (magic, header_size, flags, _r0, meta_off, meta_size, data_off, data_size, file_size, meta_crc, _r1) = \
        _HEADER.unpack(raw[:HEADER_SIZE])
    if magic != MAGIC:
        raise VVAFormatError("not a .vva file (bad magic)")
    if header_size != HEADER_SIZE or flags != 0:
        raise VVAFormatError("unsupported header")
    if file_size != len(raw):
        raise VVAFormatError("file size mismatch: the file is truncated or corrupted")
    if meta_off + meta_size > data_off or data_off + data_size != len(raw) or data_off % ALIGN:
        raise VVAFormatError("block layout is invalid")
    meta_bytes = raw[meta_off:meta_off + meta_size]
    if crc32(meta_bytes) != meta_crc:
        raise VVAFormatError("metadata checksum mismatch: the file is corrupted")
    meta = json.loads(meta_bytes.decode("utf-8"))
    if meta.get("version") != VERSION:
        raise VVAFormatError("unsupported metadata version")
    blob = memoryview(raw)[data_off:data_off + data_size]
    tensors = {}
    for t in meta["tensors"]:
        dt = _DTYPES[t["dtype"]]
        chunk = blob[t["offset"]:t["offset"] + t["size"]]
        tensors[t["name"]] = np.frombuffer(chunk, dtype=dt).reshape(t["shape"])
    return meta, tensors
