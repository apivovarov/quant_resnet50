import torch
from torch import nn
from torchvision.io import read_image
from torchvision.models import resnet, ResNet, resnet50, ResNet50_Weights
from torchinfo import summary
from resnet_block import ResnetBlockLM

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

block = resnet.Bottleneck
layers = [3, 4, 6, 3]
weights = ResNet50_Weights.verify(weights)


model = ResnetBlockLM(block, layers, weights)

model=model.eval()

summary(model, batch.shape, depth=7)

model=model.cuda()
batch=batch.cuda()

prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

torch.save(model, "model2.pt")
torch.save(model.state_dict(), "model2_state_dict.pt")
