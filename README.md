### 准备工作
1. 一个64G以上内存的电脑。
2. 配置好cuda/cudnn/TensorRT环境，安装好pytorch，transformers框架。
- cuda: 11.8
- cudnn: 8.9.1
- TensorRT: 8.6.1
- pytorch: 2.0.1
3. 安装polygraphy工具。
```bash
pip install nvidia-pyindex
pip install polygraphy
pip install colored
```
4. 大显存的英伟达显卡，至少24G以上


### 第一步：导出onnx
1. 按照transformers官方维护的onnx导出工具：optimum。[参考教程](https://huggingface.co/docs/transformers/serialization)
```bash
pip install onnxruntime-gpu
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
  --atol 5e-4 \
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
  --atol 5e-4 \
  --task text-generation-with-past \
  --sequence_length 1024 \
  --batch_size 1 \
  --no-post-process \
  output/onnx_output/tigerbot-7b-sft-fp16/
```
- 导出后有两个onnx文件，对应forward函数的两种情况。
3. 检查onnx输入输出以及onnx兼容性
- 查看非past_key版onnx的输入输出
```bash
polygraphy inspect model output/onnx_output/tigerbot-7b-sft-fp32/decoder_model.onnx
```
- 查看past_key版onnx的输入输出
```bash
polygraphy inspect model output/onnx_output/tigerbot-7b-sft-fp32/decoder_with_past_model.onnx
```
- 验证非past_key版模型能否转TensorRT
```bash
polygraphy inspect capability output/onnx_output/tigerbot-7b-sft-fp32/decoder_model.onnx 
```
- 验证past_key版模型能否转TensorRT
```bash
polygraphy inspect capability output/onnx_output/tigerbot-7b-sft-fp32/decoder_with_past_model.onnx
```

### 第二步 检查onnx
##### 对于CPU
1. 从原版那里输出一个fp32样本出来。
```bash
python3 onnx_check/export_example.py
```
2. 验证CPU下的onnx是否ok
```bash
python3 onnx_check/run_onnx_cpu.py
```


### 待完成
- [x] onnx导出
- [x] onnx对齐(包含CPU/CUDA推理+数据精度对比)
- [ ] onnx转TensorRT(fp16/int8)
- [ ] TensorRT对齐
- [ ] 推理Demo
- [ ] TensorRT+FastTransformer双重加速