# The .vva model format (version 2)

A `.vva` file is a single binary container holding an image-to-image neural
network: its **graph** (operators and how they connect) plus its **weights**.
It is designed to be memory-mapped: tensors are read in place, with no parsing
and no copying.

All integers are little-endian. Readers must reject (never crash on) any file
that does not follow these rules.

## File layout

```
+----------------------+  offset 0
| header (64 bytes)    |
+----------------------+  offset 64 (= meta_offset)
| metadata (UTF-8 JSON)|
+----------------------+
| zero padding         |  up to the next multiple of 64
+----------------------+  data_offset (multiple of 64)
| tensor data blob     |
+----------------------+  end of file
```

### Header (64 bytes)

| Offset | Size | Type    | Field         | Rule |
|-------:|-----:|---------|---------------|------|
| 0  | 4 | char[4] | magic         | `"VVA2"` (`"VVA1"` is the legacy format and is rejected) |
| 4  | 4 | u32     | header_size   | `64` |
| 8  | 4 | u32     | flags         | `0` (non-zero is rejected) |
| 12 | 4 | u32     | reserved      | `0` |
| 16 | 8 | u64     | meta_offset   | `>= 64` |
| 24 | 8 | u64     | meta_size     | `1 .. 64 MiB` |
| 32 | 8 | u64     | data_offset   | multiple of 64, `>= meta_offset + meta_size` |
| 40 | 8 | u64     | data_size     | `data_offset + data_size == file_size` |
| 48 | 8 | u64     | file_size     | must equal the real file size (detects truncation) |
| 56 | 4 | u32     | meta_crc32    | CRC-32 (ISO-HDLC, same as zlib) of the metadata bytes |
| 60 | 4 | u32     | reserved      | `0` |

## Metadata (JSON object)

```json
{
  "format": "vva",
  "version": 2,
  "meta":    { "name": "my-upscaler", "scale": 4, "author": "...", "license": "..." },
  "inputs":  [ { "name": "input",  "shape": [1, 3, -1, -1] } ],
  "outputs": [ { "name": "output", "shape": [1, 3, -1, -1] } ],
  "tensors": [
    { "name": "conv_first.weight", "dtype": "f16", "shape": [64, 3, 3, 3], "offset": 0, "size": 3456 }
  ],
  "nodes": [
    { "op": "Conv2D", "name": "conv_first",
      "inputs": ["input", "conv_first.weight", "conv_first.bias"], "outputs": ["f0"],
      "attrs": { "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1 } }
  ]
}
```

- `version` must be `2`.
- `meta.scale` (optional, 1..64): the output/input size ratio for super-resolution models. Defaults to 1.
- `inputs` / `outputs`: named graph endpoints, layout **NCHW**. `-1` marks a dynamic dimension.
- `tensors`: constants stored in the data blob.
  - `dtype`: `"f32"` or `"f16"` (IEEE 754 little-endian).
  - `size` must equal `product(shape) * bytes_per_element`.
  - `offset` is relative to `data_offset`, a multiple of 64, and `offset + size <= data_size`.
- `nodes`: the graph, in **execution order**.

### Graph rules

The graph is in single-assignment form. Every name is a graph input, a constant
tensor, or a node output. A node may only use names defined **before** it
(topological order). No name may be defined twice. Every graph output must be
produced by a node. Names must be unique across inputs, tensors and node outputs.

## Core operators

Readers must **not** reject unknown operators while loading; an executor reports
`unsupported operator` when it is asked to run one. The operators below are the
v2 core set.

| Op | Inputs | Attributes |
|----|--------|------------|
| `Conv2D` | `x`, `weight [O, I/groups, kh, kw]`, optional `bias [O]` | `stride [sh,sw]` (1,1), `padding [ph,pw]` (0,0, **zero padding**), `dilation [dh,dw]` (1,1), `groups` (1) |
| `Add` | `a`, `b` (same shape) | - |
| `Mul` | `a` and `b` (same shape), or `a` with `scalar` | `scalar` (float, optional) |
| `LeakyReLU` | `x` | `alpha` (0.01) |
| `ReLU` | `x` | - |
| `Sigmoid` | `x` | - |
| `Clip` | `x` | `min`, `max` |
| `Concat` | two or more tensors | `axis` (1) |
| `ResizeNearest` | `x` | `scale [sh, sw]` (integers; nearest-neighbour like `torch.nn.functional.interpolate(mode="nearest")`) |
| `PixelShuffle` | `x` | `upscale` (integer) |

## What the VVA engine executes

`Engine::loadModel()` runs the CPU executor's checks, so an unsupported file is
rejected when it is loaded (with the reason), not in the middle of a job. A model
runs when:

- it has exactly one input and one output, laid out NCHW with batch 1, a fixed
  channel count and dynamic height and width, e.g. `[1, 3, -1, -1]`;
- the engine entry point (`processImageTile`) additionally needs 3 input and 3
  output channels (RGB, values in 0..1);
- every operator is in the core set above; constants are only accepted as the
  `weight` / `bias` inputs of `Conv2D`, all other operands must be activations;
- `Concat` joins along the channel axis (`axis` 1);
- the output is an exact integer multiple of the input size, the same for height
  and width (this is what makes tiling possible) and equals `meta.scale` when that
  is declared. Valid (unpadded) convolutions that shrink the image are therefore
  not supported.

Weights stored as `f16` are converted to float32 when the model is loaded; all
arithmetic is float32.

## Models produced by `vva-convert`

`vva-convert` writes the x4 `RRDBNet` of basicsr / Real-ESRGAN completely:
`conv_first`, `body.N.rdb{1,2,3}.conv{1..5}` (dense blocks with `Concat`, `LeakyReLU`
and the `Mul` + `Add` residual scaling), `conv_body`, two `ResizeNearest` + `Conv2D`
+ `LeakyReLU` upsampling stages, `conv_hr` and `conv_last`. Weights are `f16`,
biases `f32`. The graph needs no side files.

## Versioning

A reader that supports version 2 must reject any other `version`, any non-zero
`flags`, and any other magic. New features that older readers cannot ignore
require a new version number.

## Python

```python
from vva.fileformat import VVAWriter, read_vva

w = VVAWriter(name="demo", scale=4)
w.add_input("input", [1, 3, -1, -1])
w.add_tensor("conv.weight", weights, "f16")
w.add_tensor("conv.bias", bias, "f32")
w.add_node("Conv2D", ["input", "conv.weight", "conv.bias"], ["output"],
           {"stride": [1, 1], "padding": [1, 1]})
w.add_output("output", [1, 3, -1, -1])
w.save("demo.vva")

meta, tensors = read_vva("demo.vva")
```

The writer refuses values that overflow fp16 (store those tensors as `f32`) and
validates the graph rules above before writing.
