#!/bin/bash

model_path=$1
trt_path=$2 

model_name=$(basename $model_path)

mkdir -p $trt_path

docker rm -f triton
docker run -i -d --name triton --gpus=all nvcr.io/nvidia/tensorrt:23.05-py3 

docker cp $model_path triton:/workspace/$model_name

docker exec triton ./tensorrt/bin/trtexec \
  --onnx=$model_name \
  --minShapes=input:1x243x17x3 \
  --optShapes=input:4x243x17x3 \
  --maxShapes=input:8x243x17x3 \
  --fp16 \
  --workspace=20480 \
  --saveEngine=model.engine \
  --timingCacheFile=timing.cache \
  --useSpinWait

docker cp triton:/workspace/model.engine "$2/model.plan"
docker rm -f triton