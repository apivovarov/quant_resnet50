import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary
from model_utils import FCN

in_sz =28*28

batch = torch.rand(1, in_sz)
batch = batch.cuda()

model = FCN()
model.load_state_dict(torch.load("results/model_state.pt"))
model = model.cuda()
model=model.eval()

summary(model, input_data=batch) # make sure the model consist of QuantConv2d layers


y = model(batch).squeeze(0).softmax(0)
print("y=", y)

print("Converting to ONNX")
dummy_input = torch.randn(1, in_sz, device="cuda")
input_names = ["input0"]
output_names = ["output0"]
dynamic_axes = {"input0": {0: "batch"}, "output0": {0: "batch"}}

y = model(dummy_input)
torch.onnx.export(
    model,
    dummy_input,
    "fcn.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)
print("The model was converted to ONNX")
