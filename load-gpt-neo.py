import torch
from transformers import pipeline
from transformers import convert_graph_to_onnx, GPTNeoForCausalLM, AutoConfig, AutoModelForCausalLM
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

model_name='EleutherAI/gpt-neo-125M'
#model_name='EleutherAI/gpt-neo-1.3B'
#model_name='EleutherAI/gpt-neo-2.7B'

device = torch.device("cuda:0")

if 1:
  model=AutoModelForCausalLM.from_pretrained(model_name).cuda()
  input_tokens = torch.LongTensor([[27079, 318]]).cuda()
  summary(model, input_data=input_tokens, depth=7, device=device)
  exit()


if 1:
  generator = pipeline('text-generation', model=model_name, device=device)
  print("Pipeline Loaded")

  res = generator("Germany is", do_sample=True, min_length=20)
  print(res)


  input_tokens = torch.LongTensor([[27079, 318]])
  summary(generator.model, input_data=input_tokens, depth=7, device=device)

  res = generator("Germany is", do_sample=True, min_length=20)
  print(res)
