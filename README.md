
# Unsloth 微调
unsloth 是一个专门为提升微调速度和降低显存使用而设计的库。它通过优化的 Triton 内核和算法，
声称可以实现比原生 Hugging Face 快2倍的训练速度和减少70%的显存占用，同时保持与 Hugging Face 生态的完全兼容。
如果您追求极致的训练效率， unsloth 是一个强大的加速工具，尤其是在资源有限的情况下,推荐使用unsloth微调模型。
官方文档 : https://docs.unsloth.ai


# 一、llama.cpp 如何部署 GGUF 格式的模型（用于 llama.cpp）

## 1、准备 GGUF 文件
如果你用 Transformers 的 `save_pretrained_gguf("q4_k_m")`，会在本地得到  
`my-model-q4_k_m.gguf`  
或者从 Hugging Face 下载别人已经转好的 `.gguf`（文件名里会带量化标识）。

## 2、编译 llama.cpp（仅需一次）
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j   # Linux/macOS
```
### Windows 用 cmake 或 直接拿 release 里的 llama.exe 
编译完会出现
```bash
./main     # 命令行聊天
./server   # 启动 OpenAI-API 兼容的 HTTP 服务
```

## 3、命令行直接跑
# 简单聊天
```bash
./main -m my-model-q4_k_m.gguf -p "Once upon a time" -n 400 -c 2048

# 带对话模板（Llama-3 为例）
./main -m my-model-q4_k_m.gguf \
       -p "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
       -n 400 -c 2048 --temp 0.7
```
参数说明
-m   GGUF 文件路径
-c   上下文长度（默认 512，可调到 4096/8192）
-n   要生成的最大 token 数
--temp 采样温度

## 4、启动 API 服务
```bash
./server -m my-model-q4_k_m.gguf -c 4096 --host 0.0.0.0 --port 8080
然后就能用任何 OpenAI 客户端访问
POST http://localhost:8080/v1/chat/completions
```
# vLLM 模型 部署模型
## 1. 安装 vLLM
## 2. 验证安装是否成功
## 3. 启动 OpenAI 兼容 API 服务
   ```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/model \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --trust-remote-code \
  --served-model-name my_model
```
