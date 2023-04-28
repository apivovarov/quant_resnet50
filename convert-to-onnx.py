from pytorch_quantization import nn as quant_nn

quant_nn.TensorQuantizer.use_fb_fake_quant = True

dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
input_names = ["input0"]
output_names = ["output0"]
dynamic_axes = {"input0": {0: "batch"}, "output0": {0: "batch"}}

# Sets the model to inference mode - train(False)
model = model.eval()
y = model(dummy_input)
torch.onnx.export(
    model,
    dummy_input,
    "quant_resnet50.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)
