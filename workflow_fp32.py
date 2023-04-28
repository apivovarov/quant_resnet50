import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary

img = read_image("cat.jpg")

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

model = resnet50(weights=weights)
model=model.eval()

summary(model, batch.shape) # make sure the model consist of QuantConv2d layers

model=model.cuda()
batch=batch.cuda()
# Sets the model to inference mode - train(False)
model=model.eval()

prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

print("Converting to ONNX")
dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
input_names = ["input0"]
output_names = ["output0"]
dynamic_axes = {"input0": {0: "batch"}, "output0": {0: "batch"}}

y = model(dummy_input)
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)
print("The model was converted to ONNX")
