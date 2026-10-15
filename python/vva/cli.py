import argparse
import os
from .converter import VVAExporter

def main():
    parser = argparse.ArgumentParser(description="VVA Model Exporter for Custom ESRGAN / RRDBNet")
    parser.add_argument("--input", "-i", required=True, help="Path to input PyTorch model (.pth)")
    parser.add_argument("--output", "-o", default=None, help="Path to output .vva binary file (optional)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found -> {input_path}")
        return

    if args.output is None:
        output_path = os.path.splitext(input_path)[0] + ".vva"
    else:
        output_path = args.output

    try:
        exporter = VVAExporter(input_path)
        exporter.export(output_path)
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    main()
