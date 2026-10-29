import argparse
import sys

from .converter import ConversionError, convert


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="vva-convert",
        description="Convert an x4 RRDBNet or MobileNetV3-Small classifier checkpoint (.pth or .npz) to a .vva model "
                    "for the VVA engine.")
    parser.add_argument("--input", "-i", required=True, help="PyTorch checkpoint (.pth/.pt) or NumPy weights (.npz)")
    parser.add_argument("--output", "-o", default=None, help="output .vva file (default: next to the input)")
    parser.add_argument("--fp32", action="store_true", help="store all weights as float32 (bigger, no fp16 rounding)")
    parser.add_argument("--name", default=None, help="model name stored in the file")
    parser.add_argument("--author", default="", help="author stored in the file")
    parser.add_argument("--license", default="", help="license stored in the file")
    parser.add_argument("--lrelu-slope", type=float, default=0.2, help="LeakyReLU slope used in training (default 0.2)")
    parser.add_argument("--res-scale", type=float, default=0.2, help="residual scaling used in training (default 0.2)")
    parser.add_argument("--classes", default=None,
                        help="classifier only: comma separated class names in output order, e.g. photo,anime")
    parser.add_argument("--input-size", type=int, default=224,
                        help="classifier only: the image is resized to this square size (default 224)")
    parser.add_argument("--mean", default="0.485,0.456,0.406", help="classifier only: normalisation mean (R,G,B)")
    parser.add_argument("--std", default="0.229,0.224,0.225", help="classifier only: normalisation std (R,G,B)")
    parser.add_argument("--bn-eps", type=float, default=1e-3, help="classifier only: BatchNorm epsilon (default 0.001)")
    parser.add_argument("--no-verify", action="store_true", help="skip the numerical check after writing")
    parser.add_argument("--allow-pickle", action="store_true",
                        help="allow full pickle loading of the .pth (only for files you trust)")
    args = parser.parse_args(argv)

    output = args.output
    if output is None:
        import os
        output = os.path.splitext(args.input)[0] + ".vva"
    try:
        mean = [float(v) for v in args.mean.split(",")]
        std = [float(v) for v in args.std.split(",")]
        if len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError
    except ValueError:
        print("error: --mean and --std need three numbers (std > 0), e.g. 0.485,0.456,0.406", file=sys.stderr)
        return 1
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    try:
        convert(args.input, output, fp16=not args.fp32, name=args.name, author=args.author, license=args.license,
                lrelu_slope=args.lrelu_slope, res_scale=args.res_scale, verify=not args.no_verify,
                allow_pickle=args.allow_pickle, classes=classes, input_size=args.input_size, mean=mean, std=std,
                bn_eps=args.bn_eps)
    except ConversionError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
