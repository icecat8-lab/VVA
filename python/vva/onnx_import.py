import os
import struct

import numpy as np

from .converter import ConversionError
from .fileformat import VVAFormatError, VVAWriter, read_vva
from .reference import run_graph


def _varint(buf, pos):
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ConversionError("the ONNX file is truncated or corrupted")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 70:
            raise ConversionError("the ONNX file is corrupted (bad varint)")


def _signed(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def _fields(buf):
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value = bytes(buf[pos:pos + 8])
            pos += 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            if pos + length > n:
                raise ConversionError("the ONNX file is truncated or corrupted")
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            value = bytes(buf[pos:pos + 4])
            pos += 4
        else:
            raise ConversionError("the ONNX file uses an unsupported protobuf encoding")
        if pos > n:
            raise ConversionError("the ONNX file is truncated or corrupted")
        yield field, wire, value


def _packed_varints(chunk):
    out = []
    pos = 0
    while pos < len(chunk):
        v, pos = _varint(chunk, pos)
        out.append(_signed(v))
    return out


_NP_TYPES = {1: np.float32, 10: np.float16, 11: np.float64, 6: np.int32, 7: np.int64, 2: np.uint8, 3: np.int8,
             5: np.int16, 9: np.bool_}


def _parse_tensor(buf):
    dims, dtype, name, raw = [], 1, "", None
    floats, ints, int64s, doubles = [], [], [], []
    external = False
    for f, w, v in _fields(buf):
        if f == 1:
            if w == 2:
                dims.extend(_packed_varints(v))
            else:
                dims.append(_signed(v))
        elif f == 2:
            dtype = v
        elif f == 4:
            if w == 2:
                floats.append(np.frombuffer(bytes(v), dtype="<f4"))
            else:
                floats.append(np.frombuffer(v, dtype="<f4"))
        elif f == 5:
            ints.extend(_packed_varints(v) if w == 2 else [_signed(v)])
        elif f == 7:
            int64s.extend(_packed_varints(v) if w == 2 else [_signed(v)])
        elif f == 8:
            name = bytes(v).decode("utf-8", "replace")
        elif f == 9:
            raw = bytes(v)
        elif f == 10:
            doubles.append(np.frombuffer(bytes(v), dtype="<f8") if w == 2 else np.frombuffer(v, dtype="<f8"))
        elif f == 14 and v == 1:
            external = True
    if external:
        raise ConversionError("tensor '%s' is stored in an external data file; re-export the model with the weights "
                              "embedded in the .onnx file" % name)
    if dtype not in _NP_TYPES:
        raise ConversionError("tensor '%s' has an unsupported data type (%d)" % (name, dtype))
    np_type = _NP_TYPES[dtype]
    count = int(np.prod(dims)) if dims else 1
    if raw is not None:
        arr = np.frombuffer(raw, dtype=np.dtype(np_type).newbyteorder("<")) if dtype != 9 else np.frombuffer(raw, dtype=np.uint8).astype(bool)
    elif floats:
        arr = np.concatenate(floats).astype(np_type)
    elif doubles:
        arr = np.concatenate(doubles).astype(np_type)
    elif int64s:
        arr = np.asarray(int64s, dtype=np_type)
    elif ints:
        arr = np.asarray(ints, dtype=np.int64).astype(np_type)
    else:
        arr = np.zeros(0, dtype=np_type)
    if arr.size != count:
        raise ConversionError("tensor '%s' has %d values but its shape needs %d" % (name, arr.size, count))
    return name, arr.reshape(dims) if dims else arr.reshape(())


def _parse_attribute(buf):
    name, atype = "", 0
    f_val, i_val, s_val, t_val = None, None, None, None
    floats, ints, strings = [], [], []
    for f, w, v in _fields(buf):
        if f == 1:
            name = bytes(v).decode("utf-8", "replace")
        elif f == 2:
            f_val = struct.unpack("<f", v)[0]
        elif f == 3:
            i_val = _signed(v)
        elif f == 4:
            s_val = bytes(v).decode("utf-8", "replace")
        elif f == 5:
            t_val = _parse_tensor(v)[1]
        elif f == 7:
            if w == 2:
                floats.extend(struct.unpack("<%df" % (len(v) // 4), bytes(v)))
            else:
                floats.append(struct.unpack("<f", v)[0])
        elif f == 8:
            ints.extend(_packed_varints(v) if w == 2 else [_signed(v)])
        elif f == 9:
            strings.append(bytes(v).decode("utf-8", "replace"))
        elif f == 20:
            atype = v
    if atype == 1:
        return name, f_val if f_val is not None else 0.0
    if atype == 2:
        return name, i_val if i_val is not None else 0
    if atype == 3:
        return name, s_val if s_val is not None else ""
    if atype == 4:
        return name, t_val
    if atype == 6:
        return name, list(floats)
    if atype == 7:
        return name, list(ints)
    if atype == 8:
        return name, list(strings)
    if f_val is not None:
        return name, f_val
    if i_val is not None:
        return name, i_val
    if s_val is not None:
        return name, s_val
    if t_val is not None:
        return name, t_val
    if floats:
        return name, list(floats)
    if ints:
        return name, list(ints)
    return name, None


def _parse_node(buf):
    node = {"in": [], "out": [], "name": "", "op": "", "attrs": {}}
    for f, w, v in _fields(buf):
        if f == 1:
            node["in"].append(bytes(v).decode("utf-8", "replace"))
        elif f == 2:
            node["out"].append(bytes(v).decode("utf-8", "replace"))
        elif f == 3:
            node["name"] = bytes(v).decode("utf-8", "replace")
        elif f == 4:
            node["op"] = bytes(v).decode("utf-8", "replace")
        elif f == 5:
            k, val = _parse_attribute(v)
            node["attrs"][k] = val
    return node


def _parse_value_info(buf):
    name, dims = "", None
    for f, w, v in _fields(buf):
        if f == 1:
            name = bytes(v).decode("utf-8", "replace")
        elif f == 2:
            for f2, w2, v2 in _fields(v):
                if f2 != 1:
                    continue
                for f3, w3, v3 in _fields(v2):
                    if f3 != 2:
                        continue
                    dims = []
                    for f4, w4, v4 in _fields(v3):
                        if f4 != 1:
                            continue
                        dv = -1
                        for f5, w5, v5 in _fields(v4):
                            if f5 == 1:
                                dv = _signed(v5)
                        dims.append(dv if dv > 0 else -1)
    return name, dims


def parse_onnx(path):
    with open(path, "rb") as f:
        raw = f.read()
    view = memoryview(raw)
    graph_buf = None
    opset = None
    for f, w, v in _fields(view):
        if f == 7 and w == 2:
            graph_buf = v
        elif f == 8 and w == 2:
            domain, version = "", None
            for f2, w2, v2 in _fields(v):
                if f2 == 1:
                    domain = bytes(v2).decode("utf-8", "replace")
                elif f2 == 2:
                    version = v2
            if domain in ("", "ai.onnx"):
                opset = version
    if graph_buf is None:
        raise ConversionError("this file is not an ONNX model (no graph found)")
    nodes, inits, inputs, outputs = [], {}, [], []
    for f, w, v in _fields(graph_buf):
        if f == 1:
            nodes.append(_parse_node(v))
        elif f == 5:
            name, arr = _parse_tensor(v)
            inits[name] = arr
        elif f == 11:
            inputs.append(_parse_value_info(v))
        elif f == 12:
            outputs.append(_parse_value_info(v))
    return {"nodes": nodes, "inits": inits, "inputs": inputs, "outputs": outputs, "opset": opset or 0}


def _ints(attrs, key, default):
    v = attrs.get(key)
    return list(v) if v is not None else list(default)


class _Importer:
    def __init__(self, model, log):
        self.model = model
        self.log = log
        self.inits = dict(model["inits"])
        self.alias = {}
        self.chan = {}
        self.ir = []
        self.producer = {}
        self.counter = 0
        self.warnings = []

    def fail(self, node, message):
        raise ConversionError("ONNX operator '%s' (node '%s'): %s" % (node["op"], node["name"] or node["out"][0], message))

    def res(self, name):
        while name in self.alias:
            name = self.alias[name]
        return name

    def const(self, name):
        if not name:
            return None
        return self.inits.get(name)

    def emit(self, op, ins, out, attrs=None, params=None, channels=None):
        ins = [self.res(n) for n in ins]
        entry = {"op": op, "in": ins, "out": out, "attrs": dict(attrs or {}), "params": dict(params or {}), "id": self.counter}
        self.counter += 1
        self.ir.append(entry)
        self.producer[out] = entry
        if channels is None:
            channels = self.chan[ins[0]]
        self.chan[out] = channels
        return entry

    def symmetric(self, node, attrs):
        auto = attrs.get("auto_pad", "NOTSET") or "NOTSET"
        if auto == "VALID":
            return [0, 0], None
        if auto != "NOTSET":
            self.fail(node, "auto_pad=%s is not supported; export with explicit paddings" % auto)
        pads = _ints(attrs, "pads", [0, 0, 0, 0])
        if len(pads) != 4:
            self.fail(node, "only 2-D operators are supported")
        t, l, b, r = pads
        if t == b and l == r:
            return [t, l], None
        return [0, 0], [t, l, b, r]

    def run(self):
        model = self.model
        data_inputs = [(n, d) for n, d in model["inputs"] if n not in self.inits]
        if len(data_inputs) != 1:
            raise ConversionError("the model must have exactly one image input (found %d)" % len(data_inputs))
        if len(model["outputs"]) != 1:
            raise ConversionError("the model must have exactly one output (found %d)" % len(model["outputs"]))
        in_name, dims = data_inputs[0]
        if dims is None or len(dims) != 4:
            raise ConversionError("the model input must be a 4-D NCHW tensor")
        if dims[1] < 1:
            raise ConversionError("the model input needs a fixed channel count")
        if dims[0] not in (-1, 1):
            raise ConversionError("only batch size 1 is supported")
        if dims[2] > 0 or dims[3] > 0:
            self.warnings.append("the input has a fixed size (%s x %s); it was made dynamic. This is only valid for "
                                 "fully convolutional networks." % (dims[2], dims[3]))
        self.in_name = in_name
        self.in_channels = dims[1]
        self.chan[in_name] = dims[1]
        self.out_name = model["outputs"][0][0]

        self.consumers = {}
        for node in model["nodes"]:
            for n in node["in"]:
                self.consumers[n] = self.consumers.get(n, 0) + 1
        self.consumers[self.out_name] = self.consumers.get(self.out_name, 0) + 1

        for node in model["nodes"]:
            self.convert_node(node)

        out = self.res(self.out_name)
        if out == in_name or out not in self.producer:
            raise ConversionError("the graph output is not produced by any supported operator")
        self.out_resolved = out

    def single_consumer(self, name):
        return self.consumers.get(name, 0) == 1

    def convert_node(self, node):
        op, attrs, ins, outs = node["op"], node["attrs"], node["in"], node["out"]
        out = outs[0] if outs else ""
        act = lambda i: self.res(ins[i])
        is_act = lambda i: i < len(ins) and ins[i] != "" and ins[i] not in self.inits and self.res(ins[i]) in self.chan

        if op in ("Identity", "Dropout"):
            if is_act(0):
                self.alias[out] = self.res(ins[0])
            else:
                self.inits[out] = self.inits[ins[0]]
            return
        if op == "Constant":
            value = attrs.get("value")
            if value is None:
                for key in ("value_float", "value_int"):
                    if key in attrs:
                        value = np.asarray(attrs[key])
            if value is None:
                self.fail(node, "unsupported Constant form")
            self.inits[out] = np.asarray(value)
            return

        if op == "Conv":
            w = self.const(ins[1])
            if w is None or w.ndim != 4:
                self.fail(node, "the weight must be a constant 4-D tensor (2-D convolution)")
            b = self.const(ins[2]) if len(ins) > 2 else None
            group = int(attrs.get("group", 1))
            stride = _ints(attrs, "strides", [1, 1])
            dil = _ints(attrs, "dilations", [1, 1])
            pad, asym = self.symmetric(node, attrs)
            x = act(0)
            if asym:
                pad_out = out + "__pad"
                self.emit("Pad", [x], pad_out, {"pads": asym, "mode": "constant", "value": 0.0})
                x = pad_out
            self.emit("Conv2D", [x], out, {"stride": stride, "padding": pad, "dilation": dil, "groups": group},
                          {"w": np.asarray(w, dtype=np.float32), **({"b": np.asarray(b, dtype=np.float32)} if b is not None else {})},
                          channels=w.shape[0])
            return
        if op == "ConvTranspose":
            w = self.const(ins[1])
            if w is None or w.ndim != 4:
                self.fail(node, "the weight must be a constant 4-D tensor")
            if int(attrs.get("group", 1)) != 1:
                self.fail(node, "grouped transposed convolutions are not supported")
            if any(d != 1 for d in _ints(attrs, "dilations", [1, 1])):
                self.fail(node, "dilated transposed convolutions are not supported")
            if "output_shape" in attrs:
                self.fail(node, "output_shape is not supported; use pads and output_padding")
            pad, asym = self.symmetric(node, attrs)
            if asym:
                self.fail(node, "asymmetric padding is not supported")
            b = self.const(ins[2]) if len(ins) > 2 else None
            self.emit("ConvTranspose2D", [act(0)], out,
                      {"stride": _ints(attrs, "strides", [1, 1]), "padding": pad,
                       "output_padding": _ints(attrs, "output_padding", [0, 0])},
                      {"w": np.asarray(w, dtype=np.float32), **({"b": np.asarray(b, dtype=np.float32)} if b is not None else {})},
                      channels=w.shape[1])
            return
        if op == "BatchNormalization":
            scale, beta, mean, var = (self.const(ins[i]) for i in range(1, 5))
            if any(v is None for v in (scale, beta, mean, var)):
                self.fail(node, "the batch-norm parameters must be constants")
            if len(outs) > 1 and outs[1]:
                self.fail(node, "training-mode batch norm is not supported")
            eps = float(attrs.get("epsilon", 1e-5))
            s = np.asarray(scale, dtype=np.float64) / np.sqrt(np.asarray(var, dtype=np.float64) + eps)
            t = np.asarray(beta, dtype=np.float64) - np.asarray(mean, dtype=np.float64) * s
            src = self.producer.get(act(0))
            if src is not None and src["op"] in ("Conv2D", "ConvTranspose2D") and ins[0] == src["out"] and self.single_consumer(ins[0]):
                w = src["params"]["w"].astype(np.float64)
                if src["op"] == "Conv2D":
                    w = w * s[:, None, None, None]
                else:
                    w = w * s[None, :, None, None]
                src["params"]["w"] = w.astype(np.float32)
                bias = src["params"].get("b")
                src["params"]["b"] = ((np.asarray(bias, dtype=np.float64) * s if bias is not None else 0.0) + t).astype(np.float32)
                self.alias[out] = act(0)
                return
            c = self.chan[act(0)]
            self.emit("ChannelAffine", [act(0)], out, {}, {"scale": s.astype(np.float32).reshape(c), "shift": t.astype(np.float32).reshape(c)})
            return
        if op in ("Relu", "Sigmoid", "Tanh", "HardSwish"):
            self.emit({"Relu": "ReLU", "Sigmoid": "Sigmoid", "Tanh": "Tanh", "HardSwish": "Hardswish"}[op], [act(0)], out)
            return
        if op == "LeakyRelu":
            self.emit("LeakyReLU", [act(0)], out, {"alpha": float(attrs.get("alpha", 0.01))})
            return
        if op == "HardSigmoid":
            alpha, beta = float(attrs.get("alpha", 0.2)), float(attrs.get("beta", 0.5))
            if abs(alpha - 1.0 / 6.0) > 1e-4 or abs(beta - 0.5) > 1e-4:
                self.fail(node, "only HardSigmoid with alpha=1/6 and beta=0.5 is supported (got %g, %g)" % (alpha, beta))
            self.emit("Hardsigmoid", [act(0)], out)
            return
        if op == "PRelu":
            slope = self.const(ins[1])
            if slope is None:
                self.fail(node, "the slope must be a constant")
            slope = np.asarray(slope, dtype=np.float32).reshape(-1)
            c = self.chan[act(0)]
            if slope.size not in (1, c):
                self.fail(node, "the slope needs 1 or %d values (one per channel)" % c)
            self.emit("PReLU", [act(0)], out, {}, {"slope": slope})
            return
        if op == "Clip":
            lo, hi = attrs.get("min"), attrs.get("max")
            if len(ins) > 1 and ins[1]:
                lo = self.const(ins[1])
                if lo is None:
                    self.fail(node, "min must be a constant")
                lo = float(np.asarray(lo).reshape(-1)[0])
            if len(ins) > 2 and ins[2]:
                hi = self.const(ins[2])
                if hi is None:
                    self.fail(node, "max must be a constant")
                hi = float(np.asarray(hi).reshape(-1)[0])
            a = {}
            if lo is not None:
                a["min"] = float(lo)
            if hi is not None:
                a["max"] = float(hi)
            self.emit("Clip", [act(0)], out, a)
            return
        if op in ("Add", "Sub", "Mul", "Div"):
            a_act, b_act = is_act(0), is_act(1)
            if a_act and b_act:
                if op == "Div":
                    self.fail(node, "division of two activations is not supported")
                self.emit({"Add": "Add", "Sub": "Sub", "Mul": "Mul"}[op], [act(0), act(1)], out)
                return
            if not a_act and not b_act:
                self.fail(node, "both operands are constants; this should have been folded by the exporter")
            const_first = not a_act
            x = act(1) if const_first else act(0)
            cval = self.const(ins[0] if const_first else ins[1])
            if cval is None:
                self.fail(node, "unsupported operand")
            cval = np.asarray(cval, dtype=np.float64)
            c = self.chan[x]
            if cval.size == 1:
                cval = np.full(c, float(cval.reshape(-1)[0]))
            else:
                shape = cval.shape
                ok = (cval.ndim == 1 and shape[0] == c) or (cval.ndim == 3 and shape == (c, 1, 1)) or \
                     (cval.ndim == 4 and shape == (1, c, 1, 1))
                if not ok:
                    self.fail(node, "the constant must be a scalar or one value per channel (shape [C], [C,1,1] or "
                                    "[1,C,1,1]); got shape %s with %d channels" % (list(shape), c))
                cval = cval.reshape(c)
            if op == "Mul":
                scale, shift = cval, np.zeros(c)
            elif op == "Add":
                scale, shift = np.ones(c), cval
            elif op == "Sub":
                scale, shift = (-np.ones(c), cval) if const_first else (np.ones(c), -cval)
            else:
                if const_first:
                    self.fail(node, "a constant divided by an activation is not supported")
                scale, shift = 1.0 / cval, np.zeros(c)
            self.emit("ChannelAffine", [x], out, {}, {"scale": scale.astype(np.float32), "shift": shift.astype(np.float32)})
            return
        if op == "Concat":
            axis = int(attrs.get("axis", 1))
            if axis not in (1, -3):
                self.fail(node, "only concatenation along the channel axis is supported")
            if not all(is_act(i) for i in range(len(ins))):
                self.fail(node, "all inputs must be activations")
            self.emit("Concat", [act(i) for i in range(len(ins))], out, {"axis": 1}, channels=sum(self.chan[act(i)] for i in range(len(ins))))
            return
        if op in ("Resize", "Upsample"):
            mode = attrs.get("mode", "nearest")
            scales = None
            if op == "Upsample":
                scales = self.const(ins[1]) if len(ins) > 1 else None
                if scales is None and "scales" in attrs:
                    scales = np.asarray(attrs["scales"])
                if mode != "nearest":
                    self.fail(node, "only mode 'nearest' is supported for Upsample; use Resize with half_pixel or align_corners")
                coord = "asymmetric"
            else:
                coord = attrs.get("coordinate_transformation_mode", "half_pixel")
                if len(ins) > 2 and ins[2]:
                    scales = self.const(ins[2])
                if (scales is None or np.asarray(scales).size == 0) and len(ins) > 3 and ins[3]:
                    self.fail(node, "Resize with a fixed output 'sizes' is not supported; export with scale factors")
            if scales is None or np.asarray(scales).size != 4:
                self.fail(node, "the scale factors must be a constant of 4 values")
            sc = np.asarray(scales, dtype=np.float64).reshape(-1)
            if abs(sc[0] - 1) > 1e-6 or abs(sc[1] - 1) > 1e-6:
                self.fail(node, "batch and channel scales must be 1")
            sh, sw = int(round(sc[2])), int(round(sc[3]))
            if abs(sc[2] - sh) > 1e-6 or abs(sc[3] - sw) > 1e-6 or not (1 <= sh <= 64 and 1 <= sw <= 64):
                self.fail(node, "only integer scale factors from 1 to 64 are supported (got %g, %g)" % (sc[2], sc[3]))
            if mode == "nearest":
                nmode = attrs.get("nearest_mode", "round_prefer_floor")
                if coord not in ("asymmetric", "half_pixel", "pytorch_half_pixel") or nmode not in ("floor", "round_prefer_floor"):
                    self.fail(node, "unsupported nearest-neighbour variant (%s / %s)" % (coord, nmode))
                self.emit("ResizeNearest", [act(0)], out, {"scale": [sh, sw]})
            elif mode == "linear":
                if coord in ("half_pixel", "pytorch_half_pixel"):
                    align = False
                elif coord == "align_corners":
                    align = True
                else:
                    self.fail(node, "coordinate_transformation_mode '%s' is not supported for linear resize" % coord)
                self.emit("ResizeBilinear", [act(0)], out, {"scale": [sh, sw], "align_corners": align})
            else:
                self.fail(node, "mode '%s' is not supported (use nearest or linear)" % mode)
            return
        if op == "DepthToSpace":
            bs = int(attrs.get("blocksize", 0))
            if attrs.get("mode", "DCR") != "CRD":
                self.fail(node, "only mode CRD is supported (this is what torch.nn.PixelShuffle exports); DCR orders channels differently")
            c = self.chan[act(0)]
            if bs < 1 or c % (bs * bs):
                self.fail(node, "invalid block size")
            self.emit("PixelShuffle", [act(0)], out, {"upscale": bs}, channels=c // (bs * bs))
            return
        if op == "GlobalAveragePool":
            self.emit("GlobalAveragePool", [act(0)], out)
            return
        if op in ("MaxPool", "AveragePool"):
            if int(attrs.get("ceil_mode", 0)) != 0:
                self.fail(node, "ceil_mode is not supported")
            if any(d != 1 for d in _ints(attrs, "dilations", [1, 1])):
                self.fail(node, "dilated pooling is not supported")
            if "kernel_shape" not in attrs or len(attrs["kernel_shape"]) != 2:
                self.fail(node, "2-D pooling with kernel_shape is required")
            pad, asym = self.symmetric(node, attrs)
            if asym:
                self.fail(node, "asymmetric padding is not supported")
            a = {"kernel": list(attrs["kernel_shape"]), "stride": _ints(attrs, "strides", [1, 1]), "padding": pad}
            if op == "AveragePool":
                a["count_include_pad"] = bool(int(attrs.get("count_include_pad", 0)))
            self.emit("MaxPool2D" if op == "MaxPool" else "AvgPool2D", [act(0)], out, a)
            return
        if op == "Pad":
            if len(ins) > 1 and ins[1]:
                pads = self.const(ins[1])
                if pads is None:
                    self.fail(node, "pads must be a constant")
                pads = [int(v) for v in np.asarray(pads).reshape(-1)]
            else:
                pads = [int(v) for v in attrs.get("pads", [])]
            if len(pads) != 8:
                self.fail(node, "only 4-D (NCHW) padding is supported")
            if any(pads[i] for i in (0, 1, 4, 5)):
                self.fail(node, "padding the batch or channel axis is not supported")
            if any(p < 0 for p in pads):
                self.fail(node, "negative padding is not supported")
            mode = attrs.get("mode", "constant")
            value = 0.0
            if len(ins) > 2 and ins[2]:
                cv = self.const(ins[2])
                if cv is None:
                    self.fail(node, "constant_value must be a constant")
                value = float(np.asarray(cv).reshape(-1)[0])
            elif "value" in attrs:
                value = float(attrs["value"])
            mode = {"constant": "constant", "reflect": "reflect", "edge": "replicate"}.get(mode)
            if mode is None:
                self.fail(node, "unsupported padding mode")
            sp = [pads[2], pads[3], pads[6], pads[7]]
            if not any(sp):
                self.alias[out] = act(0)
                return
            a = {"pads": sp, "mode": mode}
            if mode == "constant":
                a["value"] = value
            self.emit("Pad", [act(0)], out, a)
            return
        if op in ("Slice", "Split"):
            c = self.chan[act(0)]
            if op == "Slice":
                if len(ins) > 1:
                    starts, ends = self.const(ins[1]), self.const(ins[2])
                    axes = self.const(ins[3]) if len(ins) > 3 and ins[3] else None
                    steps = self.const(ins[4]) if len(ins) > 4 and ins[4] else None
                else:
                    starts, ends, axes, steps = attrs.get("starts"), attrs.get("ends"), attrs.get("axes"), None
                if starts is None or ends is None:
                    self.fail(node, "starts and ends must be constants")
                starts, ends = [int(v) for v in np.asarray(starts).reshape(-1)], [int(v) for v in np.asarray(ends).reshape(-1)]
                axes = [int(v) for v in np.asarray(axes).reshape(-1)] if axes is not None else list(range(len(starts)))
                if steps is not None and any(int(v) != 1 for v in np.asarray(steps).reshape(-1)):
                    self.fail(node, "slice steps other than 1 are not supported")
                if len(starts) != 1 or axes[0] not in (1, -3):
                    self.fail(node, "only slicing along the channel axis is supported")
                s, e = starts[0], min(ends[0], c)
                s = s + c if s < 0 else s
                e = e + c if e < 0 else e
                if not 0 <= s < e <= c:
                    self.fail(node, "the channel range is empty or outside the input")
                self.emit("Slice", [act(0)], out, {"start": s, "end": e}, channels=e - s)
                return
            axis = int(attrs.get("axis", 0))
            if axis not in (1, -3):
                self.fail(node, "only splitting along the channel axis is supported")
            sizes = attrs.get("split")
            if sizes is None and len(ins) > 1 and ins[1]:
                sizes = [int(v) for v in np.asarray(self.const(ins[1])).reshape(-1)]
            if sizes is None:
                if c % len(outs):
                    self.fail(node, "the channels cannot be split evenly")
                sizes = [c // len(outs)] * len(outs)
            if sum(sizes) != c:
                self.fail(node, "the split sizes do not add up to the channel count")
            pos = 0
            for o, size in zip(outs, sizes):
                self.emit("Slice", [act(0)], o, {"start": pos, "end": pos + size}, channels=size)
                pos += size
            return
        self.fail(node, "this operator is not supported by VVA. Supported: Conv, ConvTranspose, BatchNormalization, "
                        "Relu, LeakyRelu, PRelu, Sigmoid, Tanh, HardSwish, HardSigmoid, Clip, Add, Sub, Mul, Div "
                        "(with constants per channel), Concat, Resize (integer scales), Upsample, DepthToSpace (CRD), "
                        "MaxPool, AveragePool, GlobalAveragePool, Pad, Slice, Split, Identity")


def _build_writer(imp, name, fp16, author, license):
    w = VVAWriter(name=name, scale=1, author=author, license=license)
    w.add_input(imp.in_name, [1, imp.in_channels, -1, -1])
    originals = {}
    for e in imp.ir:
        pnames = []
        for role, arr in e["params"].items():
            tname = "__p%d_%s" % (e["id"], role)
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            use16 = fp16 and role in ("w",) and arr.size > 0 and float(np.max(np.abs(arr))) < 60000.0
            w.add_tensor(tname, arr, "f16" if use16 else "f32")
            originals[tname] = arr.astype(np.float64)
            pnames.append((role, tname))
        order = {"Conv2D": ["w", "b"], "ConvTranspose2D": ["w", "b"], "PReLU": ["slope"], "ChannelAffine": ["scale", "shift"]}
        names = dict(pnames)
        extra = [names[r] for r in order.get(e["op"], []) if r in names]
        w.add_node(e["op"], list(e["in"]) + extra, [e["out"]], e["attrs"])
    w.add_output(imp.out_resolved, [1, -1, -1, -1])
    return w, originals


def _infer_scale(meta, tensors, in_channels):
    sizes = ((16, 24), (24, 16))
    ratios = []
    for h, wd in sizes:
        x = np.random.RandomState(0).rand(in_channels, h, wd)
        try:
            y = run_graph(meta, tensors, x)
        except (ValueError, IndexError) as exc:
            raise ConversionError("the converted graph cannot run on a %dx%d test image (%s); models that downsample need input sizes that are a multiple of 16" % (h, wd, exc))
        if y.shape[1] % h or y.shape[2] % wd or y.shape[1] // h != y.shape[2] // wd:
            raise ConversionError("the model output size is not an exact integer multiple of the input size (%dx%d -> "
                                  "%dx%d); VVA runs image-to-image models with a fixed integer scale"
                                  % (h, wd, y.shape[1], y.shape[2]))
        ratios.append(y.shape[1] // h)
    if ratios[0] != ratios[1]:
        raise ConversionError("the model scale depends on the input size")
    return ratios[0], y.shape[0]


def convert_onnx(input_path, output_path, fp16=True, name=None, author="", license="", verify=True, log=print):
    model = parse_onnx(input_path)
    imp = _Importer(model, log)
    imp.run()
    name = name or os.path.splitext(os.path.basename(output_path))[0]
    writer, originals = _build_writer(imp, name, fp16, author, license)
    meta = writer.to_meta()
    scale, out_channels = _infer_scale(meta, originals, imp.in_channels)
    writer.meta_info["scale"] = int(scale)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        size = writer.save(output_path)
    except VVAFormatError as exc:
        raise ConversionError(str(exc))
    ops = {}
    for e in imp.ir:
        ops[e["op"]] = ops.get(e["op"], 0) + 1
    log("converted: %s  (%d operators, %d channels in, %d out, scale x%d, %.1f MB)"
        % (output_path, len(imp.ir), imp.in_channels, out_channels, scale, size / 1e6))
    log("operators: " + ", ".join("%s x%d" % kv for kv in sorted(ops.items())))
    for message in imp.warnings:
        log("note: " + message)
    if verify:
        meta2, stored = read_vva(output_path)
        stored = {k: np.asarray(v, dtype=np.float64) for k, v in stored.items()}
        x = np.random.RandomState(1).rand(imp.in_channels, 16, 24)
        want = run_graph(meta, originals, x)
        got = run_graph(meta2, stored, x)
        denom = max(float(np.max(np.abs(want))), 1e-6)
        err = float(np.max(np.abs(got - want))) / denom
        if err > 0.05:
            raise ConversionError("verification failed: fp16 weights change the output too much (%.2e); use --fp32" % err)
        log("verified: the file reproduces the network, fp16 storage changes the output by %.1e of its range" % err)
    return output_path
