import torch.onnx
import onnx
from onnxsim import simplify
from model import attention_unet, depthwise_separable_conv
import os

models = os.listdir('best_model')

for model_name in models:
    path = os.path.join('best_model', model_name)
    if 'DS' in model_name:
        dsau = depthwise_separable_conv.DS_attention_unet()
        dsau.load_state_dict(torch.load(path))
        dsau.eval()
        dsau.to('cuda:0')
        dummy_input = torch.rand(1, 3, 184, 320).to('cuda:0')
        
        # Export model to ONNX format
        onnx_path = f'best_model/{model_name.split(".")[0]}.onnx'
        torch.onnx.export(dsau, dummy_input, onnx_path, verbose=True, opset_version=12)
        
        # Simplify and save with "_sim" suffix
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        assert check, f"Simplified ONNX model {onnx_path} could not be validated"
        simplified_path = f'best_model/{model_name.split(".")[0]}_sim.onnx'
        onnx.save(model_simp, simplified_path)
        onnx.checker.check_model(model_simp)

    else:
        au = attention_unet.attention_unet()
        au.load_state_dict(torch.load(path))
        au.eval()
        au.to('cuda:0')
        dummy_input = torch.rand(1, 3, 184, 320).to('cuda:0')
        
        # Export model to ONNX format
        onnx_path = f'best_model/{model_name.split(".")[0]}.onnx'
        torch.onnx.export(au, dummy_input, onnx_path, verbose=True, opset_version=12)
        
        # Simplify and save with "_sim" suffix
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        assert check, f"Simplified ONNX model {onnx_path} could not be validated"
        simplified_path = f'best_model/{model_name.split(".")[0]}_sim.onnx'
        onnx.save(model_simp, simplified_path)
        onnx.checker.check_model(model_simp)
