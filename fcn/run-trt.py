import numpy as np
BATCH_SIZE = 8
batch = np.ones(shape=(BATCH_SIZE, 28*28), dtype="float32") / 100

import tensorrt as trt
from net_utils import cuda_error_check
from cuda import cuda

cuda_error_check(cuda.cuInit(0))
cuDevice = cuda_error_check(cuda.cuDeviceGet(0))
cuCtx = cuda_error_check(cuda.cuCtxCreate(0, cuDevice))
import sys
import numpy as np
trt_logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(trt_logger)

fpath="fcn_fp16_int8.trt"
fpath="fcn_fp16.trt"
fpath="fcn_fp32.trt"

with open(fpath, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
context.set_input_shape("input0", (BATCH_SIZE, 28*28))

print("Engine Info:")
for i, binding in enumerate(engine):
    shape = [engine.max_batch_size, *engine.get_binding_shape(binding)]
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    volume = abs(trt.volume(engine.get_binding_shape(binding)))
    if engine.binding_is_input(binding):
        desc = "input"
    else:
        desc = "output"
    print(f"{i} type:    {desc}\n  binding: {binding} \n  data:    {np.dtype(dtype).name}\n  shape:   {shape} => {volume} \n")

USE_FP16 = False
target_dtype = np.float16 if USE_FP16 else np.float32

output = np.empty([BATCH_SIZE, 1000], dtype=target_dtype)

# allocate device memory
d_input = cuda_error_check(cuda.cuMemAlloc(1 * batch.nbytes))
d_output = cuda_error_check(cuda.cuMemAlloc(1 * output.nbytes))
bindings = [int(d_input), int(d_output)]
stream = cuda_error_check(cuda.cuStreamCreate(0))


def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda_error_check(
        cuda.cuMemcpyHtoDAsync(d_input, batch, batch.nbytes, stream)
    )
    # execute model
    context.execute_async_v2(bindings, stream, None)
    # transfer predictions back
    cuda_error_check(
        cuda.cuMemcpyDtoHAsync(output, d_output, output.nbytes, stream)
    )
    # synchronize threads
    cuda_error_check(
        cuda.cuStreamSynchronize(stream)
    )

predict(batch)

print("output:", output)

# Warmup
for i in range(100):
  predict(batch)

# Measure Latency
import time
TT=[]
N = 3000
for i in range(N):
  t0=time.time()
  predict(batch)
  t1=time.time()
  TT.append((t1-t0)*1000/BATCH_SIZE)

print("AVG time (ms):",np.mean(TT))
print("P50 time (ms):",np.percentile(TT, 50))
print("P95 time (ms):",np.percentile(TT, 95))
