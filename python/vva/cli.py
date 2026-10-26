import argparse
import sys

from .converter import ConversionError, convert


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="vva-convert",
        description="Convert an x4 RRDBNet checkpoint (.pth or .npz) to a .vva model for the VVA engine.")
    parser.add_argument("--input", "-i", required=True, help="PyTorch checkpoint (.pth/.pt) or NumPy weights (.npz)")
    parser.add_argument("--output", "-o", default=None, help="output .vva file (default: next to the input)")
    parser.add_argument("--fp32", action="store_true", help="store all weights as float32 (bigger, no fp16 rounding)")
    parser.add_argument("--name", default=None, help="model name stored in the file")
    parser.add_argument("--author", default="", help="author stored in the file")
    parser.add_argument("--license", default="", help="license stored in the file")
    parser.add_argument("--lrelu-slope", type=float, default=0.2, help="LeakyReLU slope used in training (default 0.2)")
    parser.add_argument("--res-scale", type=float, default=0.2, help="residual scaling used in training (default 0.2)")
    parser.add_argument("--no-verify", action="store_true", help="skip the numerical check after writing")
    parser.add_argument("--allow-pickle", action="store_true",
                        help="allow full pickle loading of the .pth (only for files you trust)")
    args = parser.parse_args(argv)

    output = args.output
    if output is None:
        import os
        output = os.path.splitext(args.input)[0] + ".vva"
    try:
        convert(args.input, output, fp16=not args.fp32, name=args.name, author=args.author, license=args.license,
                lrelu_slope=args.lrelu_slope, res_scale=args.res_scale, verify=not args.no_verify,
                allow_pickle=args.allow_pickle)
    except ConversionError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
