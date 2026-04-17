# CUDA E2E 测试记录

## 测试环境

| 项目 | 值 |
|------|-----|
| 机器 | GPU 服务器 |
| GPU | NVIDIA A800-SXM4-80GB (单卡) |
| PyTorch | — |
| vLLM | dev (from source, commit `9e138cb01`, 环境 `vllm17`) |
| 运行日期 | 2026-04-17 |

**注意**: 第一轮测试用 vLLM v0.11.0 (pip install)，因版本与 NPU 端不一致，重新用 dev 版本跑了一轮。

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

# 3. 使用 vllm17 环境（dev 版本，从 source build）
#    vLLM 源码: /data/lizhijun/work/fd-vllm/vllm
VLLM_PYTHON="/data/lizhijun/anaconda3/envs/vllm17/bin/python"

# 4. 启动 baseline
export CUDA_VISIBLE_DEVICES=3
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model /data-ssd/lizhijun/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --port 8010 --max-model-len 2048 --gpu-memory-utilization 0.85 &

# 5. 启动 tree_n2
export CUDA_VISIBLE_DEVICES=3
$VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model /data-ssd/lizhijun/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --port 8010 --max-model-len 2048 --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"eagle3","model":"/data-ssd/lizhijun/models/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":2,"speculative_token_tree":"[(0,), (0, 0)]"}' &

# 6. 用 Python 发请求（避免 curl proxy 问题）
$VLLM_PYTHON /tmp/run_test_vllm17.py 8010 "test_name"
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

### vLLM dev 版本（与 NPU 同版本，公平对比）

| 模式 | 吞吐量 (tokens/s) | vs Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|-------------|-------------------|-------------------|
| Baseline | 86.64 | — | N/A | N/A |
| Tree n=2 | 100.14 | +15.6% | 1.48 | 0.351, 0.125 |
| Tree n=3 | 98.07 | +13.2% | 1.50 | 0.318, 0.119, 0.064 |

所有模式 Async scheduling: enabled

### vLLM v0.11.0（第一轮测试，仅供参考）

| 模式 | 吞吐量 (tokens/s) | vs Baseline |
|------|-------------------|-------------|
| Baseline | 86.62 | — |
| Tree n=2 | 89.36 | +3.2% |
| Tree n=3 | 99.82 | +15.2% |

## 结论

- CUDA GPU 上 tree speculative decoding 带来了可测量的加速（+13.2% ~ +15.6%）
- 使用 dev 版本后，tree n=2 和 n=3 的加速比更接近（+15.6% vs +13.2%），趋势与 NPU 一致
- CUDA baseline 吞吐量（86.64）约为 NPU（54.68）的 1.58x
- CUDA Accept Length（1.48~1.50）与 NPU（1.46~1.50）接近，说明两个平台的 draft model 行为一致
- NPU 优化后 tree n=2 达到 +14.3%，与 CUDA 的 +15.6% 仅差 1.3pp
- 两个平台趋势一致：tree speculative decoding 有效提升吞吐量

## NPU vs CUDA 最终对比（同一 vLLM dev 版本）

| 指标 | CUDA (A800) | NPU (910B2C, 优化后) | 差距 |
|------|------------|---------------------|------|
| Tree n=2 加速 | +15.6% | **+14.3%** | 1.3pp |
| Tree n=3 加速 | +13.2% | **+10.2%** | 3.0pp |
| Mean Accept Length | 1.48~1.50 | 1.46~1.50 | 一致 |
| Position 0 接受率 | 0.32~0.35 | 0.35 | 一致 |
| Position 1 接受率 | 0.12~0.13 | 0.11 | 一致 |
