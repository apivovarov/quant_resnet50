trtexec \
--int8 \
--verbose \
--onnx=quant_resnet50.onnx \
--saveEngine=quant_resnet50.trt \
--minShapes=input0:1x3x224x224 \
--optShapes=input0:8x3x224x224 \
--maxShapes=input0:16x3x224x224
