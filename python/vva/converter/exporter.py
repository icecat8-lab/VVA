import struct
import json
import torch
import torch.nn as nn

class VVAExporter:
    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def export(self, output_vva_path: str):
        metadata = {"layers": []}
        weight_bytes = bytearray()
        offset = 0

        for name, param in self.model.named_parameters():
            tensor = param.data
            
            if "norm" in name.lower() or "out" in name.lower():
                precision = "FP32"
                data_bytes = tensor.to(torch.float32).numpy().tobytes()
            else:
                precision = "FP16"
                data_bytes = tensor.to(torch.float16).numpy().tobytes()

            length = len(data_bytes)
            metadata["layers"].append({
                "name": name,
                "shape": list(tensor.shape),
                "precision": precision,
                "offset": offset,
                "length": length
            })
            weight_bytes.extend(data_bytes)
            offset += length

        meta_json = json.dumps(metadata).encode('utf-8')
        with open(output_vva_path, 'wb') as f:
            f.write(b'VVA1')
            f.write(struct.pack('<I', len(meta_json)))
            f.write(meta_json)
            f.write(weight_bytes)
          
