import struct

import numpy as np


def varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def key(field, wire):
    return varint((field << 3) | wire)


def f_varint(field, value):
    return key(field, 0) + varint(value)


def f_bytes(field, data):
    return key(field, 2) + varint(len(data)) + bytes(data)


def f_str(field, text):
    return f_bytes(field, text.encode())


def f_float(field, value):
    return key(field, 5) + struct.pack("<f", value)


def tensor(name, arr, mode="raw"):
    arr = np.asarray(arr)
    dtype = {np.dtype("float32"): 1, np.dtype("int64"): 7, np.dtype("float64"): 11}[arr.dtype]
    out = b"".join(f_varint(1, d) for d in arr.shape) + f_varint(2, dtype)
    if mode == "raw":
        out += f_bytes(9, arr.tobytes())
    elif mode == "float_data":
        out += f_bytes(4, arr.astype("<f4").tobytes())
    elif mode == "int64_data":
        out += f_bytes(7, b"".join(varint(int(v)) for v in arr.reshape(-1)))
    return out + f_str(8, name)


def attribute(name, value):
    out = f_str(1, name)
    if isinstance(value, np.ndarray):
        return out + f_bytes(5, tensor("", value)) + f_varint(20, 4)
    if isinstance(value, float):
        return out + f_float(2, value) + f_varint(20, 1)
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return out + f_varint(3, value) + f_varint(20, 2)
    if isinstance(value, str):
        return out + f_bytes(4, value.encode()) + f_varint(20, 3)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], float):
            return out + f_bytes(7, struct.pack("<%df" % len(value), *value)) + f_varint(20, 6)
        return out + f_bytes(8, b"".join(varint(int(v)) for v in value)) + f_varint(20, 7)
    raise TypeError(type(value))


def node(op, ins, outs, attrs=None, name=""):
    out = b"".join(f_str(1, i) for i in ins) + b"".join(f_str(2, o) for o in outs)
    out += f_str(3, name or outs[0]) + f_str(4, op)
    for k, v in (attrs or {}).items():
        out += f_bytes(5, attribute(k, v))
    return out


def value_info(name, dims):
    shape = b""
    for d in dims:
        shape += f_bytes(1, f_varint(1, d) if isinstance(d, int) else f_str(2, d))
    tensor_type = f_varint(1, 1) + f_bytes(2, shape)
    return f_str(1, name) + f_bytes(2, f_bytes(1, tensor_type))


def model(nodes, inits, in_name, in_dims, out_name, out_dims, opset=13, init_mode="raw"):
    g = b"".join(f_bytes(1, n) for n in nodes) + f_str(2, "g")
    for name, arr in inits.items():
        mode = init_mode
        if np.asarray(arr).dtype == np.int64 and mode == "float_data":
            mode = "int64_data"
        g += f_bytes(5, tensor(name, arr, mode))
    g += f_bytes(11, value_info(in_name, in_dims)) + f_bytes(12, value_info(out_name, out_dims))
    return f_varint(1, 8) + f_bytes(8, f_str(1, "") + f_varint(2, opset)) + f_bytes(7, g)
