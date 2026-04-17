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
| vllm-ascend | branch `feature/tree-spec-decode-cuda-verify`, commit `016ef078` |
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

### 4.1 CUDA vs NPU 对比

| 指标 | CUDA (dev, A800) | NPU (dev, 910B2C) | 分析 |
|------|-----------------|-------------|------|
| Baseline 吞吐量 | 86.64 | 56.60 | CUDA baseline 约为 NPU 的 1.53x |
| Tree n=2 吞吐量 | 100.14 | 61.82 | |
| Tree n=3 吞吐量 | 98.07 | 60.85 | |
| Tree n=2 vs Baseline | +15.6% | +12.4% (公平对比) | 趋势一致 |
| Tree n=3 vs Baseline | +13.2% | +10.6% (公平对比) | 趋势一致 |
| Mean Accept Length | 1.48~1.50 | 1.43~1.50 | 两个平台 draft model 行为一致 |
| Position 0 接受率 | 0.35 / 0.32 | 0.33 | |
| Position 1 接受率 | 0.13 / 0.12 | 0.12 | |
| Async scheduling | enabled (两个模式都开启) | baseline=on, tree=off | |

### 4.2 关键发现

1. **趋势一致**: CUDA 和 NPU 上 tree speculative decoding 都带来 10~16% 的加速
2. **Accept Length 接近**: 两个平台的 Mean Accept Length 都在 1.43~1.50，draft model 行为一致
3. **Per-position 接受率一致**: Position 0 约 0.32~0.35，Position 1 约 0.12~0.13
4. **n=2 vs n=3**: 两个平台上 n=2 和 n=3 的加速比都接近，n=2 略优于 n=3
5. **NPU 实现正确性**: 综合以上数据，NPU 实现与 CUDA 端行为高度一致

### 4.3 NPU 实现正确性验证

以下证据表明 NPU 实现是正确的：

1. **所有模式都有正向加速**: baseline 56.60 → tree n=2 61.82 (+12.4%), tree n=3 60.85 (+10.6%)
2. **SpecDecoding metrics 正常输出**: Mean acceptance length、Per-position acceptance rate 均正常
3. **Acceptance rate 与 CUDA 一致**: Position 0: ~0.33 vs CUDA ~0.32~0.35, Position 1: ~0.12 vs CUDA ~0.12~0.13
4. **Accept Length 与 CUDA 一致**: NPU 1.43~1.50, CUDA 1.48~1.50
5. **单元测试全通过**: 11/11 tests pass
6. **代码与原生 vLLM 逐行对齐**: `propose_tree`、`_propose_tree`、`build_for_drafting` 等核心逻辑一致

---

## 五、CUDA 端测试详情

### 5.1 测试环境

- vLLM: dev 版本，从 `/data/lizhijun/work/fd-vllm/vllm` source build
- Conda 环境: `vllm17`
- GPU: 单卡 A800 (CUDA_VISIBLE_DEVICES=3)

### 5.2 可选的补充实验

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
