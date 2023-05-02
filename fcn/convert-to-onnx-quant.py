import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

# Set default QuantDescriptor to use histogram based calibration for activation
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

# Now new models will automatically have QuantConv2d layers instead of regular Conv2d
from pytorch_quantization import quant_modules
quant_modules.initialize()

from model_utils import FCN

in_sz =28*28

batch = torch.rand(1, in_sz).cuda()

model = FCN()
model.load_state_dict(torch.load("results/model_state.pt"))
model = model.cuda()
model=model.eval()

summary(model, input_data=batch) # make sure the model consist of QuantConv2d layers


y = model(batch).squeeze(0).softmax(0)
print("y=", y)

print("Converting to ONNX")
quant_nn.TensorQuantizer.use_fb_fake_quant = True

dummy_input = torch.randn(1, in_sz, device="cuda")
input_names = ["input0"]
output_names = ["output0"]
dynamic_axes = {"input0": {0: "batch"}, "output0": {0: "batch"}}

y = model(dummy_input)
torch.onnx.export(
    model,
    dummy_input,
    "fcn_quant.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)
print("The model was converted to ONNX")
