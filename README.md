### 准备工作（暂无）


### 第一步：导出onnx
1. 按照transformers官方维护的onnx导出工具：optimum。[参考教程](https://huggingface.co/docs/transformers/serialization)
```bash
pip install optimum[exporters]
```

2. 查看帮助
```bash
optimum-cli export onnx --help
```

2. 一键导出命令：
- 纯CPU导出fp32格式(如果内存够大，比如128G以上，可以去掉`--no-post-process`)
```bash
optimum-cli export onnx \
  --model tigerbot-7b-sft \
  --framework pt \
  --opset 18 \
  --task text-generation-with-past \
  --sequence_length 1024 \
  --batch_size 1 \
  --no-post-process \
  output/onnx_output/tigerbot-7b-sft-fp32/
```
- 纯GPU导出fp16格式，需要较大显存，预计最低24G以上，如果显存不够，可以降低seq_len长度。
```bash
optimum-cli export onnx \
  --model tigerbot-7b-sft \
  --framework pt \
  --device cuda \
  --fp16 \
  --opset 18 \
  --task text-generation-with-past \
  --sequence_length 1024 \
  --batch_size 1 \
  --no-post-process \
  output/onnx_output/tigerbot-7b-sft-fp16/
```
- 导出后有两个onnx文件，对应forward函数的两种情况。


### 待完成
- [x] onnx导出
- [ ] onnx对齐(包含CPU/CUDA推理+数据精度对比)
- [ ] onnx转TensorRT(fp16/int8)
- [ ] TensorRT对齐
- [ ] 推理Demo
- [ ] TensorRT+FastTransformer双重加速