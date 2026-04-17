# CUDA E2E 测试记录

## 测试环境

| 项目 | 值 |
|------|-----|
| 机器 | GPU 服务器 |
| GPU | NVIDIA A800-SXM4-80GB (单卡) |
| PyTorch | — |
| vLLM | v0.11.0 |
| 运行日期 | 2026-04-17 |

## 测试目标

在 CUDA GPU 上运行 `speculative_token_tree` 的 baseline 和 tree 模式，
获得 GPU 基线数据，与 NPU 实现进行对比。

## 测试配置

| 项目 | 值 |
|------|-----|
| Target 模型 | Meta-Llama-3.1-8B-Instruct (来自 modelscope) |
| Draft 模型 | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B |
| max_model_len | 2048 |
| gpu_memory_utilization | 0.85 |
| temperature | 0.0 |
| max_tokens | 100 |
| 测试请求数 | 10 |
| 测试 prompts | 10 个固定 prompt（见脚本） |

### Tree 配置

| 名称 | num_speculative_tokens | speculative_token_tree |
|------|----------------------|----------------------|
| tree_n2 | 2 | `[(0,), (0, 0)]` |
| tree_n3 | 3 | `[(0,), (0, 0), (0, 1)]` |

## 复现步骤

```bash
# 1. 切换分支
git checkout feature/tree-spec-decode-cuda-verify

# 2. 模型路径（已下载到 /data-ssd/lizhijun/models/）
#    Target: /data-ssd/lizhijun/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
#    Draft:  /data-ssd/lizhijun/models/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

# 3. 如果需要下载模型（通过 modelscope）：
pip install modelscope
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir /data-ssd/lizhijun/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
huggingface-cli download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --local-dir /data-ssd/lizhijun/models/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

# 4. 跑测试（脚本方式，需要指定空闲 GPU）
#    修改脚本中的 CUDA_VISIBLE_DEVICES=N 指定空闲卡
python param-selection/tree_spec_decode_cuda_e2e.py --mode both --models-dir /data-ssd/lizhijun/models
```

### 手动启动方式（更可控）

```bash
# 启动 baseline
export CUDA_VISIBLE_DEVICES=3  # 指定空闲卡
python -m vllm.entrypoints.openai.api_server \
    --model /data-ssd/lizhijun/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --port 8010 --max-model-len 2048 --gpu-memory-utilization 0.85 &

# 启动 tree_n2
export CUDA_VISIBLE_DEVICES=3
python -m vllm.entrypoints.openai.api_server \
    --model /data-ssd/lizhijun/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --port 8010 --max-model-len 2048 --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"eagle3","model":"/data-ssd/lizhijun/models/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":2,"speculative_token_tree":"[(0,), (0, 0)]"}' &

# 用 Python 发请求（避免 curl proxy 问题）
python /tmp/run_test.py 8010 "test_name"
```

## 踩坑记录

### 1. 端口冲突

首次跑测试时端口 8000 被占用，改为 8010/8011。

### 2. 显存不足

多卡环境下 `CUDA_VISIBLE_DEVICES` 需要通过 shell 脚本传递给子进程，
直接在 `conda run` 或命令行 export 可能不生效给 EngineCore 子进程。

解决方案：写一个 shell 启动脚本，在脚本内 `export CUDA_VISIBLE_DEVICES=N`。

### 3. curl proxy 问题

环境配了 `http_proxy=127.0.0.1:7870`，curl 请求 localhost 会被代理拦截。
解决方案：用 Python `urllib.request` 发请求，或 `curl --noproxy '*'`。

### 4. 模型下载

Meta-Llama-3.1-8B-Instruct 是 gated repo，需要 HF token。
通过 modelscope 下载到 `/data-ssd/lizhijun/models/LLM-Research/` 路径。

### 5. vLLM 启动超时

`torch.compile` + CUDA graph capture 约需 2-3 分钟。
脚本默认 120s 超时不够，改为 300s。

### 6. 服务器清理

每次测试完必须 kill vLLM server 和 EngineCore 子进程，
否则残留进程占用显存导致后续测试失败：

```bash
ps aux | grep "api_server\|VLLM::EngineCore" | grep -v grep | awk '{print $2}' | xargs kill -9
```

## 测试结果

| 模式 | 吞吐量 (tokens/s) | vs Baseline |
|------|-------------------|-------------|
| Baseline | 86.62 | — |
| Tree n=2 | 89.36 | +3.2% |
| Tree n=3 | 99.82 | +15.2% |

## 结论

- CUDA GPU 上 tree speculative decoding 带来了可测量的加速（+3.2% ~ +15.2%）
- Tree n=3 比 n=2 加速更明显（+15.2% vs +3.2%），说明更深的 tree 允许更多并行
- NPU 上 tree 加速效果更显著（+22.0% / +18.4%），可能因为 NPU baseline 速度本身较低，tree 并行化的收益更大
- 两个平台趋势一致：tree speculative decoding 有效提升吞吐量
