import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary

from tqdm import tqdm

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

img = read_image("cat.jpg")

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

model = resnet50(weights=weights)
model=model.eval()

summary(model, batch.shape) # make sure the model consist of QuantConv2d layers

model=model.cuda()
batch=batch.cuda()

prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
