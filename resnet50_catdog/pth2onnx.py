import torch
import torchvision
import numpy as np
from onnxruntime.datasets import get_example
import onnxruntime

model = torch.load("models/model_pruned_0.5.pth")

model.eval()

x = torch.randn(1,3,224,224).cuda() 	
export_onnx_file = "models/model_pruned_0.5.onnx" 			
torch.onnx.export(model,x,export_onnx_file,verbose=True, input_names=["x"], output_names=["y"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

dummy_input = torch.randn(1, 3, 224, 224).cuda()
model.eval()
with torch.no_grad():
    torch_out = model(dummy_input)
print(torch_out)

example_model = get_example('/home/xywang/code/pruning/Torch-Pruning/resnet50_catdog/models/model_pruned_0.5.onnx')
sess = onnxruntime.InferenceSession(example_model)
onnx_out = sess.run(None, {sess.get_inputs()[0].name: to_numpy(dummy_input)})
print(onnx_out)

np.testing.assert_almost_equal(to_numpy(torch_out), onnx_out[0], decimal=3)