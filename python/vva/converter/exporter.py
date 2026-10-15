import torch
import struct
import numpy as np
import os
import json

class VVAExporter:
    def __init__(self, data):
        if isinstance(data, dict):
            self.state_dict = data.get('params_ema', data.get('state_dict', data.get('params', data)))
        else:
            self.model = data.eval()
            self.state_dict = self.model.state_dict()

    def _infer_layer_type(self, name):
        name_lower = name.lower()
        if 'conv' in name_lower and 'weight' in name_lower:
            return "Conv2D"
        elif 'norm' in name_lower:
            return "InstanceNorm" if 'weight' in name_lower or 'bias' in name_lower else "Norm"
        elif 'bias' in name_lower:
            return "Bias"
        elif 'prelu' in name_lower or 'leaky' in name_lower:
            return "Activation"
        return "Unknown"

    def export(self, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        config_path = output_path.rsplit('.', 1)[0] + '.json'
        
        model_config = {
            "model_name": os.path.basename(output_path).split('.')[0],
            "total_layers": len(self.state_dict),
            "layers": []
        }
        
        with open(output_path, 'wb') as f:
            f.write(b'VVA1') 
            f.write(struct.pack('I', len(self.state_dict)))
            
            for name, tensor in self.state_dict.items():
                if 'num_batches_tracked' in name:
                    continue

                np_data = tensor.detach().cpu().numpy()
                shape = np_data.shape
                
                if 'norm' in name.lower() or 'bias' in name.lower():
                    np_data = np_data.astype(np.float32)
                    dtype_idx, dtype_str = 0, "FP32"
                else:
                    np_data = np_data.astype(np.float16)
                    dtype_idx, dtype_str = 1, "FP16"
                
                model_config["layers"].append({
                    "name": name,
                    "type": self._infer_layer_type(name),
                    "shape": list(shape),
                    "dtype": dtype_str
                })
                
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('I', dtype_idx))
                f.write(struct.pack('I', len(shape)))
                for dim in shape:
                    f.write(struct.pack('I', dim))
                f.write(np_data.tobytes())
                
        with open(config_path, 'w', encoding='utf-8') as json_file:
            json.dump(model_config, json_file, indent=4)
