import os
import json
import struct
import torch
import numpy as np

class VVAExporter:
    def __init__(self, pth_path: str):
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"Model file not found: {pth_path}")
        self.pth_path = pth_path
        checkpoint = torch.load(pth_path, map_location='cpu')
        self.state_dict = checkpoint.get('params_ema', checkpoint.get('state_dict', checkpoint.get('params', checkpoint)))

    def _auto_detect_config(self) -> int:
        body_indices = set()
        for key in self.state_dict.keys():
            if key.startswith('body.'):
                parts = key.split('.')
                try: body_indices.add(int(parts[1]))
                except ValueError: pass
        return max(body_indices) + 1 if body_indices else 6

    def export(self, output_path: str):
        nb = self._auto_detect_config()
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        config_path = output_path.rsplit('.', 1)[0] + '.json'
        
        model_config = {"model_name": os.path.basename(output_path).split('.')[0], "auto_detected_nb": nb, "total_layers": 0, "layers": []}
        valid_count = 0
        
        with open(output_path, 'wb') as f:
            f.write(b'VVA1')
            count_pos = f.tell()
            f.write(struct.pack('I', 0))

            for name, tensor in self.state_dict.items():
                if 'num_batches_tracked' in name: continue
                arr = tensor.detach().cpu().numpy()
                shape = list(arr.shape)

                if 'norm' in name.lower() or 'bias' in name.lower():
                    arr = arr.astype(np.float32)
                    dtype_idx, dtype_str = 0, "FP32"
                else:
                    arr = arr.astype(np.float16)
                    dtype_idx, dtype_str = 1, "FP16"

                model_config["layers"].append({"name": name, "shape": shape, "dtype": dtype_str})
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('I', dtype_idx))
                f.write(struct.pack('I', len(shape)))
                for dim in shape: f.write(struct.pack('I', dim))
                f.write(arr.tobytes())
                valid_count += 1

            f.seek(count_pos)
            f.write(struct.pack('I', valid_count))

        model_config["total_layers"] = valid_count
        with open(config_path, 'w', encoding='utf-8') as jf:
            json.dump(model_config, jf, indent=4)
        print(f"แปลงสำเร็จ! บันทึกที่: {output_path} (nb={nb}, {valid_count} เลเยอร์)")
