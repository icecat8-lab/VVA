#!/bin/sh
set -eu

usage() {
    echo "usage: compare-ncnn.sh <image.png|jpg> <model.vva> <ncnn-binary> <ncnn-model-dir> <ncnn-model-name> <scale> [tile]" >&2
    echo "example: compare-ncnn.sh photo.jpg maru-x4.vva ./realesrgan-ncnn-vulkan ./models realesrgan-x4plus 4 256" >&2
    exit 1
}

[ $# -ge 6 ] || usage
IMG="$1"; MODEL="$2"; NCNN="$3"; NMODELS="$4"; NNAME="$5"; SCALE="$6"; TILE="${7:-0}"
TILEARG=""
[ "$TILE" = "0" ] || TILEARG="--tile $TILE"

for tool in magick python3; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool is required (pkg install imagemagick python python-numpy)" >&2; exit 1; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

magick "$IMG" -alpha off "$WORK/in.png"
magick "$WORK/in.png" "$WORK/in.ppm"
SIZE="$(magick identify -format '%wx%h' "$WORK/in.png")"
echo "input: $SIZE, scale x$SCALE"

now() { python3 -c 'import time; print(time.time())'; }

echo "== VVA =="
vva-cli run "$MODEL" "$WORK/in.ppm" "$WORK/vva.ppm" $TILEARG > /dev/null
T0="$(now)"
vva-cli run "$MODEL" "$WORK/in.ppm" "$WORK/vva.ppm" $TILEARG
T1="$(now)"

echo "== ncnn =="
"$NCNN" -i "$WORK/in.png" -o "$WORK/ncnn.png" -m "$NMODELS" -n "$NNAME" -s "$SCALE" -t "$TILE" > /dev/null 2>&1 || true
T2="$(now)"
"$NCNN" -i "$WORK/in.png" -o "$WORK/ncnn.png" -m "$NMODELS" -n "$NNAME" -s "$SCALE" -t "$TILE"
T3="$(now)"
magick "$WORK/ncnn.png" -alpha off "$WORK/ncnn.ppm"

python3 - "$WORK/vva.ppm" "$WORK/ncnn.ppm" "$T0" "$T1" "$T2" "$T3" <<'PY'
import re
import sys

import numpy as np


def load(path):
    d = open(path, "rb").read()
    m = re.match(rb"P6\s+(\d+)\s+(\d+)\s+(\d+)\s", d)
    w, h = int(m.group(1)), int(m.group(2))
    return np.frombuffer(d[m.end():], dtype=np.uint8)[: w * h * 3].reshape(h, w, 3).astype(np.int32)


a, b = load(sys.argv[1]), load(sys.argv[2])
t0, t1, t2, t3 = (float(v) for v in sys.argv[3:7])
print("VVA time : %.1f s (second run, warm)" % (t1 - t0))
print("ncnn time: %.1f s (second run, warm)" % (t3 - t2))
if a.shape != b.shape:
    print("output sizes differ: VVA %s, ncnn %s; cannot compare pixels" % (a.shape, b.shape))
else:
    mse = float(np.mean((a - b) ** 2))
    print("difference between the two pictures: max %d levels, mean %.3f, PSNR %s" %
          (int(np.abs(a - b).max()), float(np.abs(a - b).mean()), "identical" if mse == 0 else "%.1f dB" % (10 * np.log10(255.0 ** 2 / mse))))
print("(the two programs use different weights and tile borders unless you point both at the same model, so a low PSNR "
      "between different models is expected; compare times, and look at both pictures)")
PY
