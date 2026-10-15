import torch
import struct
import numpy as np
import os

class VVAExporter:
    def __init__(self, data):
        """
        ตรวจสอบและโหลดน้ำหนัก (Weights) ของโมเดล
        รองรับทั้งไฟล์ที่เป็น State Dict (Dictionary) และไฟล์ Full Model
        """
        if isinstance(data, dict):
            
            self.state_dict = data.get('params_ema', data.get('state_dict', data.get('params', data)))
        else:
            
            self.model = data.eval()
            self.state_dict = self.model.state_dict()

    def export(self, output_path: str):
        """
        วิเคราะห์เลเยอร์ บีบอัด Precision และบันทึกเป็นฟอร์แมต .vva Binary
        """
        print(f"[VVA-CLI] เริ่มต้นการแปลง... พบเลเยอร์ทั้งหมด: {len(self.state_dict)}")
        
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            
            f.write(b'VVA1') 
            
            
            f.write(struct.pack('I', len(self.state_dict)))
            
            
            for name, tensor in self.state_dict.items():
                
                
                if 'num_batches_tracked' in name:
                    continue

                
                np_data = tensor.detach().cpu().numpy()
                

                
                if 'norm' in name.lower() or 'bias' in name.lower():
                    np_data = np_data.astype(np.float32)
                    dtype_idx = 0 
                else:
                    
                    np_data = np_data.astype(np.float16)
                    dtype_idx = 1  
                
                
                
                
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                
                
                f.write(struct.pack('I', dtype_idx))
                
                
                shape = np_data.shape
                f.write(struct.pack('I', len(shape)))
                for dim in shape:
                    f.write(struct.pack('I', dim))
                    
                
                f.write(np_data.tobytes())
                
        print(f"[VVA-CLI] แปลงโมเดลสำเร็จ! บันทึกไฟล์ไปที่: {output_path}")

