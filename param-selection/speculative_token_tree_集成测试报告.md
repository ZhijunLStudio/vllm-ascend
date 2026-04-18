# speculative_token_tree CUDA vs NPU 对比验证

## 概述

本文档记录 `speculative_token_tree` 在 CUDA GPU 和 Ascend NPU 上的对比测试结果，用于验证 NPU 实现的正确性和性能。

**目标**: 使用相同的 model pair 和配置，在 GPU 和 NPU 上分别运行，对比吞吐量和 speculative decoding metrics。

---

## 一、测试环境

### CUDA 端（A800 80GB）

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA A800-SXM4-80GB (单卡) |
| vLLM | **dev** (from source, commit `9e138cb01`, 环境 `vllm17`) |
| PyTorch | — |
| 运行日期 | 2026-04-17 |

### NPU 端（910B2C）

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910B2C |
| vLLM | **dev** (from source, fork: [ZhijunLStudio/vllm](https://github.com/ZhijunLStudio/vllm), commit `0f3ce4c74`, branch `main`) |
| vllm-ascend | branch `feature/tree-spec-decode-cuda-verify`, commit `54d494d2` |
| VLLM_VERSION | 0.19.0 (环境变量) |
| torch_npu | — |
| 运行日期 | 2026-04-17 |

**重要**: CUDA 端和 NPU 端均使用 dev 版本 vLLM（同一 fork，相近 commit），可直接对比。

---

## 二、测试配置

| 项目 | 值 |
|------|-----|
| Target 模型 | Meta-Llama-3.1-8B-Instruct |
| Draft 模型 | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B |
| max_model_len | 2048 |
| temperature | 0.0 |
| 测试请求数 | 10 |
| Test prompts | 10 个固定 prompt（见 `tree_spec_decode_cuda_e2e.py`） |

### Tree 配置

| 名称 | num_speculative_tokens | speculative_token_tree |
|------|----------------------|----------------------|
| tree_n2 | 2 | `[(0,), (0, 0)]` |
| tree_n3 | 3 | `[(0,), (0, 0), (0, 1)]` |

---

## 三、测试结果

### 3.1 CUDA 结果（A800 80GB，vLLM dev，Async scheduling enabled）

| 模式 | 吞吐量 (tokens/s) | vs Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|-------------|-------------------|-------------------|
| Baseline | 86.64 | — | N/A | N/A |
| Tree n=2 | 100.14 | **+15.6%** | 1.48 | 0.351, 0.125 |
| Tree n=3 | 98.07 | **+13.2%** | 1.50 | 0.318, 0.119, 0.064 |

### 3.2 NPU 结果（910B2C，vLLM dev，优化后）

**注意**: vllm-ascend 在 speculative decoding 活跃时会**强制禁用 async scheduling**（`platform.py:886-893`），而 baseline 默认开启 async scheduling。这导致 tree 模式的基线实际上比 baseline 更低。

**方式 1: 与默认 baseline 对比（async enabled）**

| 模式 | 吞吐量 (tokens/s) | vs Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|-------------|-------------------|-------------------|
| Baseline (async=on) | 56.60 | — | N/A | N/A |
| Tree n=2 (async=off) | 62.49 | +10.4% | 1.46 | 0.349, 0.109 |
| Tree n=3 (async=off) | 60.23 | +6.4% | 1.50 | 0.318, 0.100, 0.042 |

**方式 2: 相同 async 配置对比（都禁用 async scheduling）**

| 模式 | 吞吐量 (tokens/s) | vs No-Async Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|---------------------|-------------------|-------------------|
| Baseline (async=off) | 54.68 | — | N/A | N/A |
| Tree n=2 (async=off) | 62.49 | **+14.3%** | 1.46 | 0.349, 0.109 |
| Tree n=3 (async=off) | 60.23 | **+10.2%** | 1.50 | 0.318, 0.100, 0.042 |

### 3.3 NPU 历史结果对比

| 版本 | Tree n=2 加速 | Tree n=3 加速 | 备注 |
|------|-------------|-------------|------|
| 优化前 (commit `016ef078`) | +12.4% | +10.6% | 初始实现，`.tolist()` 强制同步 |
| 优化后 (commit `54d494d2`) | **+14.3%** | **+10.2%** | 消除同步 + ACL Graph 支持 |

---

## 四、对比分析

### 4.1 CUDA vs NPU 最终对比

| 指标 | CUDA (dev, A800) | NPU (dev, 910B2C, 优化后) | 差距 |
|------|-----------------|-------------------------|------|
| Baseline 吞吐量 | 86.64 | 54.68 | CUDA 为 NPU 的 1.58x |
| Tree n=2 吞吐量 | 100.14 | 62.49 | |
| Tree n=3 吞吐量 | 98.07 | 60.23 | |
| Tree n=2 vs Baseline | +15.6% | **+14.3%** | 差距 1.3pp |
| Tree n=3 vs Baseline | +13.2% | **+10.2%** | 差距 3.0pp |
| Mean Accept Length | 1.48~1.50 | 1.46~1.50 | 基本一致 |
| Position 0 接受率 | 0.32~0.35 | 0.35 | 基本一致 |
| Position 1 接受率 | 0.12~0.13 | 0.11 | 基本一致 |
| Async scheduling | enabled | baseline=on, tree=off | |

### 4.2 关键发现

1. **Tree n=2 已接近 CUDA 水平**: NPU +14.3% vs CUDA +15.6%，差距仅 1.3pp
2. **Accept Length 完全一致**: 两个平台的 Mean Accept Length 都在 1.46~1.50
3. **Per-position 接受率一致**: Position 0 约 0.32~0.35，Position 1 约 0.11~0.13
4. **n=2 优于 n=3**: 两个平台都是 n=2 加速比更高，符合预期
5. **优化有效**: 消除 `.tolist()` 同步后，tree n=2 从 +12.4% 提升到 +14.3%

### 4.3 NPU 实现正确性验证

以下证据表明 NPU 实现是正确的：

1. **所有模式都有正向加速**: baseline 54.68 → tree n=2 62.49 (+14.3%), tree n=3 60.23 (+10.2%)
2. **SpecDecoding metrics 正常输出**: Mean acceptance length、Per-position acceptance rate 均正常
3. **Acceptance rate 与 CUDA 一致**: Position 0: 0.35 vs CUDA 0.32~0.35, Position 1: 0.11 vs CUDA 0.12~0.13
4. **Accept Length 与 CUDA 一致**: NPU 1.46~1.50, CUDA 1.48~1.50
5. **单元测试全通过**: 11/11 tests pass
6. **代码与原生 vLLM 逐行对齐**: `propose_tree`、`_propose_tree`、`build_for_drafting` 等核心逻辑一致

---

## 五、性能优化记录

### 5.1 优化 1: 消除 CPU-GPU 同步

**文件**: `vllm_ascend/attention/backends/tree_attn.py` — `_forward_decode` 方法

**问题**: 每次 tree level forward 都执行两个 `.tolist()` 调用，强制 CPU-GPU 同步：
```python
# Before（同步调用）:
actual_seq_lengths = decode_meta.query_start_loc[1:].tolist()
actual_seq_lengths_kv = decode_meta.seq_lens.tolist()
```

**修复**: 改为 tensor 操作，避免 CPU 同步：
```python
# After（无同步）:
actual_seq_lengths = decode_meta.query_start_loc[1:].to(torch.int64)
actual_seq_lengths_kv = decode_meta.seq_lens.to(torch.int64)
```

**依据**: 标准 decode 路径（`attention_v1.py:860`）已经用 tensor 传参给 `npu_fused_infer_attention_score`，kernel 完全支持。

**收益**: Tree n=2 从 +12.4% 提升到 +14.3%（+1.9pp）

### 5.2 优化 2: ACL Graph 支持

**文件**: `vllm_ascend/attention/backends/tree_attn.py` — `_forward_decode` 方法

**问题**: 当前 `_forward_decode` 直接调用 `npu_fused_infer_attention_score`，没有使用 ACL Graph 优化。而标准 decode 路径（`attention_v1.py` 的 `full_graph_fia`）使用了 `ExternalEvent` + `graph_task_group_begin/end` 来支持 ACL Graph。

**修复**: 当 `_EXTRA_CTX.is_draft_model` 为 True 时（即在 `propose_tree` 中），使用：
- `_npu_fused_infer_attention_score_get_max_workspace` 预计算 workspace
- `.out()` 变体避免额外内存分配

与标准 decode 的 `full_graph_fia` 模式对齐。非 draft model 场景保持原有逻辑不变。

**收益**: 与优化 1 共同贡献，tree n=2 总提升 +1.9pp

### 5.3 待进一步优化方向

1. **Async scheduling**: vllm-ascend 强制禁用 async scheduling（`platform.py:886-893`），如果能支持，可进一步提升 NPU tree 模式性能
2. **更优的 tree 结构**: 当前测试的 n=2 和 n=3 差距不大，可尝试更复杂的 tree 结构
3. **batch_size > 1**: 当前测试均为单请求串行，多请求并发场景可能有不同表现

---

## 六、NPU 端到端测试方法

### 6.1 环境准备

```bash
# Python 路径
PYTHON="/usr/local/python3.11.14/bin/python3"

# 环境变量
export VLLM_VERSION=0.19.0
export ASCEND_RT_VISIBLE_DEVICES=0
```

### 6.2 测试步骤

```bash
# 1. 启动 baseline（async=off，公平对比基线）
$PYTHON -m vllm.entrypoints.openai.api_server \
  --model /data/models/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --max-model-len 2048 --no-async-scheduling

# 2. 启动 tree n=2
$PYTHON -m vllm.entrypoints.openai.api_server \
  --model /data/models/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --max-model-len 2048 \
  --speculative-config '{"method":"eagle3","model":"/data/models/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":2,"speculative_token_tree":"[(0,), (0, 0)]"}'

# 3. 启动 tree n=3
$PYTHON -m vllm.entrypoints.openai.api_server \
  --model /data/models/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --max-model-len 2048 \
  --speculative-config '{"method":"eagle3","model":"/data/models/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":3,"speculative_token_tree":"[(0,), (0, 0), (0, 1)]"}'
```

### 6.3 发送测试请求

```python
import urllib.request, json, time

TEST_PROMPTS = [
    "The theory of relativity was proposed by Albert Einstein. It fundamentally changed our understanding of space, time, and gravity. The key ideas include",
    "Hello, how are you today? Please tell me a joke.",
    "Explain quantum computing in simple terms.",
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
    "Write a short story about a robot learning to paint.",
    "Machine learning is a subset of artificial intelligence that focuses on",
    "The solar system consists of eight planets orbiting around the Sun. The innermost planet is",
    "Python is a popular programming language because",
    "The history of the internet dates back to the 1960s when",
    "Climate change is one of the most pressing issues facing humanity today. The primary cause is",
]

port = 8000
url = f'http://localhost:{port}/v1/completions'
total_tokens = 0
start = time.time()

for i, prompt in enumerate(TEST_PROMPTS):
    data = json.dumps({
        'model': '/data/models/Meta-Llama-3.1-8B-Instruct',
        'prompt': prompt,
        'max_tokens': 100,
        'temperature': 0.0
    }).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    ct = resp['usage']['completion_tokens']
    total_tokens += ct
    t = time.time() - t0
    print(f'Request {i+1}: {ct} tokens in {t:.2f}s ({ct/t:.1f} tokens/s)')

total_time = time.time() - start
print(f'\nTotal: {total_tokens} tokens in {total_time:.2f}s = {total_tokens/total_time:.2f} tokens/s')
```

### 6.4 采集 SpecDecoding metrics

```bash
grep -i "specdecod\|acceptance\|accept length\|mean accept" /tmp/npu_server.log
```

### 6.5 清理残留进程

```bash
ps aux | grep "api_server\|VLLM::EngineCore" | grep -v grep | awk '{print $2}' | xargs kill -9
```

---

## 七、NPU 实现详情

### 7.1 已对齐原生 vLLM 的功能

| 功能 | 状态 | 文件 |
|------|------|------|
| Backend 名称 `"TREE_ATTN"` | ✅ | `vllm_ascend/attention/backends/tree_attn.py` |
| Root token 行为 `[:, 0] = 0` | ✅ | `tree_attn.py` |
| 祖先关系计算 `_is_ancestor` | ✅ | `tree_attn.py` |
| Mask 缓存优化 | ✅ | `tree_attn.py` |
| `propose_tree` 调用 | ✅ | `eagle_proposer.py` |
| `_propose_tree` 方法 | ✅ | `eagle_proposer.py` |
| `slot_mapping` 传递 | ✅ | `ascend_forward_context.py` |
| `num_tokens_across_dp` 传递 | ✅ | `eagle_proposer.py` |
| ACL Graph per-level dispatch | ✅ | `eagle_proposer.py` |
| `exceeds_max_model_len` 处理 | ✅ | `eagle_proposer.py` |
| SpecDecoding metrics 输出 | ✅ | Server log |
| ACL Graph attention workspace | ✅ | `tree_attn.py` (优化后) |
| 无同步 actual_seq_lengths | ✅ | `tree_attn.py` (优化后) |

### 7.2 NPU 独有优化

| 优化 | 说明 |
|------|------|
| Mask 缓存 | 预计算所有 slice mask，避免运行时重复计算 |
| int8 mask | GPU 用 float32 (-inf/0)，NPU 用 int8 (1/0)，pad 到 2048x2048 |
| `.out()` 变体 | ACL Graph 模式下使用 `.out()` 避免额外内存分配 |

### 7.3 Bug 修复记录

| 编号 | 严重度 | 问题 | 修复 |
|------|--------|------|------|
| 1 | P0 | `propose_tree` 从未被调用 | `_propose` 添加 `AscendTreeAttentionMetadata` 类型检查 |
| 2 | P0 | Backend 名称 `"ASCEND_TREE_ATTN"` 不匹配 | 改为 `"TREE_ATTN"` |
| 3 | P0 | Root token 行为 `[0, :] = 0` 语义反了 | 改为 `[:, 0] = 0` |
| 4 | P1 | `slot_mapping` 未传递 | `set_ascend_forward_context` 添加参数 |
| 5 | P1 | ACL Graph per-level 缺失 | `propose_tree` 添加 dispatch |
| 6 | P2 | `.tolist()` 强制 CPU 同步 | 改为 `.to(torch.int64)` |
| 7 | P2 | Attention 未使用 ACL Graph | 添加 workspace 预计算 + `.out()` 支持 |

---

## 八、单元测试

NPU 端单元测试全部通过（11/11）：

```
tests/ut/attention/test_tree_attn_npu.py (11 tests)
```

---

## 九、修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `vllm_ascend/attention/backends/__init__.py` | 新建 | backends 包初始化 |
| `vllm_ascend/attention/backends/tree_attn.py` | 新建 | AscendTreeAttentionBackend 核心实现 + 优化 |
| `vllm_ascend/attention/attention_v1.py` | 修改 | 注册 TREE_ATTN backend |
| `vllm_ascend/attention/attention_mask.py` | 修改 | mask 转换工具函数 |
| `vllm_ascend/spec_decode/eagle_proposer.py` | 修改 | propose_tree + _propose_tree + ACL Graph |
| `vllm_ascend/ascend_forward_context.py` | 修改 | 添加 slot_mapping 参数 |
| `tests/ut/attention/test_tree_attn_npu.py` | 新建 | 单元测试 (11 个) |
| `param-selection/tree_spec_decode_cuda_e2e.py` | 新建 | CUDA E2E 复现脚本 |
| `param-selection/test_tree_attn_npu.py` | 新建 | NPU kernel 验证脚本 |
