trtexec \
--int8 \
--fp16 \
--verbose \
--onnx=quant_resnet50_test.onnx \
--saveEngine=resnet50_fp16_int8_test.trt \
--minShapes=input0:1x3x224x224 \
--optShapes=input0:8x3x224x224 \
--maxShapes=input0:16x3x224x224
