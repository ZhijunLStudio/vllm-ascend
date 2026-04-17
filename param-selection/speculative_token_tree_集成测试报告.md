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
| vLLM | **v0.11.0** (pip install) |
| PyTorch | — |
| 运行日期 | 2026-04-17 |

### NPU 端（910B2C）

| 项目 | 值 |
|------|-----|
| NPU | Ascend 910B2C |
| vLLM | **dev** (from source, fork: [ZhijunLStudio/vllm](https://github.com/ZhijunLStudio/vllm), commit `0f3ce4c74`, branch `main`) |
| vllm-ascend | branch `feature/tree-spec-decode-cuda-verify`, commit `016ef078` |
| VLLM_VERSION | 0.19.0 (环境变量) |
| torch_npu | — |
| 运行日期 | 2026-04-17 |

**重要**: CUDA 端使用 pip 安装的 v0.11.0，NPU 端使用从 source build 的 dev 版本。两个版本的 EAGLE tree 实现可能有差异，**直接对比绝对吞吐量和加速比需要谨慎**。

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

### 3.1 CUDA 结果（A800 80GB，vLLM v0.11.0）

| 模式 | 吞吐量 (tokens/s) | vs Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|-------------|-------------------|-------------------|
| Baseline | 86.62 | — | N/A | N/A |
| Tree n=2 | 89.36 | +3.2% | **未采集** | **未采集** |
| Tree n=3 | 99.82 | +15.2% | **未采集** | **未采集** |

### 3.2 NPU 结果（910B2C，vLLM dev，2026-04-17）

**注意**: vllm-ascend 在 speculative decoding 活跃时会**强制禁用 async scheduling**（`platform.py:886-893`），而 baseline 默认开启 async scheduling。这导致 tree 模式的基线实际上比 baseline 更低。

**方式 1: 与默认 baseline 对比（async enabled）**

| 模式 | 吞吐量 (tokens/s) | vs Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|-------------|-------------------|-------------------|
| Baseline (async=on) | 56.60 | — | N/A | N/A |
| Tree n=2 (async=off) | 61.82 | +9.3% | 1.43 | 0.32, 0.11 |
| Tree n=3 (async=off) | 60.85 | +7.5% | 1.50 | 0.33, 0.12, 0.05 |

**方式 2: 相同 async 配置对比（都禁用 async scheduling）**

| 模式 | 吞吐量 (tokens/s) | vs No-Async Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|---------------------|-------------------|-------------------|
| Baseline (async=off) | 55.02 | — | N/A | N/A |
| Tree n=2 (async=off) | 61.82 | **+12.4%** | 1.43 | 0.32, 0.11 |
| Tree n=3 (async=off) | 60.85 | **+10.6%** | 1.50 | 0.33, 0.12, 0.05 |

---

## 四、对比分析

### 4.1 CUDA vs NPU 对比（需谨慎解读）

| 指标 | CUDA (v0.11.0) | NPU (dev) | 备注 |
|------|----------------|-----------|------|
| Baseline 吞吐量 | 86.62 | 56.60 | NPU 基线低 35%，硬件差异 |
| Tree n=2 吞吐量 | 89.36 | 61.82 | |
| Tree n=3 吞吐量 | 99.82 | 60.85 | |
| Tree n=2 vs Baseline | +3.2% | +12.4% (公平对比) | 版本不同，不可直接比较 |
| Tree n=3 vs Baseline | +15.2% | +10.6% (公平对比) | 版本不同，不可直接比较 |
| Position 0 接受率 | **未采集** | 0.33 | |
| Position 1 接受率 | **未采集** | 0.12 | |
| Mean Accept Length | **未采集** | 1.43~1.50 | |

### 4.2 当前测试的局限性

1. **vLLM 版本不一致**: CUDA 用 v0.11.0 (pip)，NPU 用 dev (source build)。EAGLE tree 实现可能有差异。
2. **CUDA 缺少 acceptance rate**: 无法判断两个平台的 draft model 行为是否一致。
3. **Async scheduling 差异**: NPU tree 模式被强制禁用 async scheduling，CUDA v0.11.0 可能根本没有此功能。
4. **n=2 vs n=3 趋势不一致**: CUDA n=3 远优于 n=2（+15.2% vs +3.2%），NPU 两者接近。原因可能是版本差异或 NPU kernel 对大 mask 的开销。

### 4.3 NPU 实现正确性验证

尽管绝对性能对比受限于版本差异，以下证据表明 NPU 实现是正确的：

1. **所有模式都有正向加速**: baseline 56.60 → tree n=2 61.82 (+12.4%), tree n=3 60.85 (+10.6%)
2. **SpecDecoding metrics 正常输出**: Mean acceptance length、Per-position acceptance rate 均正常
3. **Acceptance rate 稳定**: Position 0: ~0.33, Position 1: ~0.12，多次测试一致
4. **单元测试全通过**: 11/11 tests pass
5. **代码与原生 vLLM 逐行对齐**: `propose_tree`、`_propose_tree`、`build_for_drafting` 等核心逻辑一致

---

## 五、需要重新做的实验

### 5.1 CUDA 端需要提供的数据

在 **相同版本的 vLLM**（建议用 dev 版本，从 source build）上重新测试：

```bash
# 1. 使用与 NPU 相同的 vllm fork
git clone https://github.com/ZhijunLStudio/vllm.git
cd vllm
git checkout main  # commit 0f3ce4c74
pip install -e .

# 2. 下载模型
python param-selection/tree_spec_decode_cuda_e2e.py --download --models-dir /path/to/models

# 3. 运行测试
python param-selection/tree_spec_decode_cuda_e2e.py --mode both --models-dir /path/to/models
```

**必须采集的数据**:

| 数据 | 说明 |
|------|------|
| Baseline 吞吐量 | tokens/s，async scheduling 开启 |
| Tree n=2 吞吐量 | tokens/s |
| Tree n=3 吞吐量 | tokens/s |
| Tree n=2 Mean Accept Length | server log 中 `SpecDecoding metrics` 行 |
| Tree n=3 Mean Accept Length | server log 中 `SpecDecoding metrics` 行 |
| Tree n=2 Per-position acceptance rate | server log 中 `SpecDecoding metrics` 行 |
| Tree n=3 Per-position acceptance rate | server log 中 `SpecDecoding metrics` 行 |
| Async scheduling 状态 | server log 中 `Asynchronous scheduling is ...` 行 |

### 5.2 NPU 端可选的补充实验

1. 测试更多 tree 结构（如 `[(0,), (0, 0), (0, 1), (0, 2)]`，n=4）
2. 测试不同的 `max_model_len`（1024, 4096）
3. 测试 batch_size > 1 的场景

---

## 六、NPU 实现详情

### 6.1 已对齐原生 vLLM 的功能

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

### 6.2 NPU 独有优化

| 优化 | 说明 |
|------|------|
| Mask 缓存 | 预计算所有 slice mask，避免运行时重复计算 |
| int8 mask | GPU 用 float32 (-inf/0)，NPU 用 int8 (1/0)，pad 到 2048x2048 |

### 6.3 Bug 修复记录

| 编号 | 严重度 | 问题 | 修复 |
|------|--------|------|------|
| 1 | P0 | `propose_tree` 从未被调用 | `_propose` 添加 `AscendTreeAttentionMetadata` 类型检查 |
| 2 | P0 | Backend 名称 `"ASCEND_TREE_ATTN"` 不匹配 | 改为 `"TREE_ATTN"` |
| 3 | P0 | Root token 行为 `[0, :] = 0` 语义反了 | 改为 `[:, 0] = 0` |
| 4 | P1 | `slot_mapping` 未传递 | `set_ascend_forward_context` 添加参数 |
| 5 | P1 | ACL Graph per-level 缺失 | `propose_tree` 添加 dispatch |

---

## 七、单元测试

NPU 端单元测试全部通过（11/11）：

```
tests/ut/attention/test_tree_attn_npu.py (11 tests)
```

---

## 八、修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `vllm_ascend/attention/backends/__init__.py` | 新建 | backends 包初始化 |
| `vllm_ascend/attention/backends/tree_attn.py` | 新建 | AscendTreeAttentionBackend 核心实现 |
| `vllm_ascend/attention/attention_v1.py` | 修改 | 注册 TREE_ATTN backend |
| `vllm_ascend/attention/attention_mask.py` | 修改 | mask 转换工具函数 |
| `vllm_ascend/spec_decode/eagle_proposer.py` | 修改 | propose_tree + _propose_tree + ACL Graph |
| `vllm_ascend/ascend_forward_context.py` | 修改 | 添加 slot_mapping 参数 |
| `tests/ut/attention/test_tree_attn_npu.py` | 新建 | 单元测试 (11 个) |
| `param-selection/tree_spec_decode_cuda_e2e.py` | 新建 | CUDA E2E 复现脚本 |
| `param-selection/test_tree_attn_npu.py` | 新建 | NPU kernel 验证脚本 |
