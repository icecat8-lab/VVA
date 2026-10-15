import argparse
from vva.converter.exporter import VVAExporter
import torch

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--model", type=str, required=True)
    convert_parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.command == "convert":
        model = torch.load(args.model, map_location='cpu')
        exporter = VVAExporter(model)
        exporter.export(args.output)

if __name__ == "__main__":
    main()
  
