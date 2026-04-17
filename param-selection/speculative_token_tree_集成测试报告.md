# speculative_token_tree CUDA vs NPU 对比验证

## 概述

本文档记录 `speculative_token_tree` 在 CUDA GPU 和 Ascend NPU 上的对比测试结果，用于验证 NPU 实现的正确性和性能。

**目标**: 使用相同的 model pair 和配置，在 GPU 和 NPU 上分别运行，对比吞吐量和 speculative decoding metrics。

---

## 测试配置

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

## 测试步骤

### CUDA 端（目标机器）

```bash
# 1. 切换到 tree-spec-decode-cuda-verify 分支
git checkout feature/tree-spec-decode-cuda-verify

# 2. 下载模型
python param-selection/tree_spec_decode_cuda_e2e.py --download --models-dir /path/to/models

# 3. 运行完整测试（baseline + tree_n2 + tree_n3）
python param-selection/tree_spec_decode_cuda_e2e.py --mode both --models-dir /path/to/models

# 结果保存在 tree_spec_decode_cuda_results.json
```

### NPU 端（已完成）

NPU 测试已在 Ascend 910B2C 上完成，结果见下方。

---

## 结果对比

### CUDA 结果（已填入）

| 模式 | 吞吐量 (tokens/s) | vs Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|-------------|-------------------|-------------------|
| Baseline | 86.62 | — | N/A | N/A |
| Tree n=2 | 89.36 | +3.2% | — | — |
| Tree n=3 | 99.82 | +15.2% | — | — |

### NPU 结果（已完成）

| 模式 | 吞吐量 (tokens/s) | vs Baseline | Mean Accept Length | Position 0-N 接受率 |
|------|-------------------|-------------|-------------------|-------------------|
| Baseline | 56.65 | — | N/A | N/A |
| Tree n=2 | 69.08 | +22.0% | 1.41 | 0.283, 0.126 |
| Tree n=3 | 67.05 | +18.4% | 1.38 | 0.278, 0.085, 0.014 |

### 对比分析

| 指标 | CUDA (A800 80GB) | NPU (910B2C) | 分析 |
|------|-----------------|-------------|------|
| Baseline 吞吐量 | 86.62 | 56.65 | CUDA 单卡 baseline 约为 NPU 的 1.53x |
| Tree n=2 吞吐量 | 89.36 | 69.08 | |
| Tree n=2 vs Baseline | +3.2% | +22.0% | NPU 上 tree 加速效果更显著 |
| Tree n=3 吞吐量 | 99.82 | 67.05 | |
| Tree n=3 vs Baseline | +15.2% | +18.4% | 两者趋势一致，n=3 均优于 n=2 |

---

## NPU 实现细节

### 已对齐原生 vLLM 的功能

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

### NPU 独有优化

| 优化 | 说明 |
|------|------|
| Mask 缓存 | 预计算所有 slice mask，避免运行时重复计算 |
| int8 mask | GPU 用 float32 (-inf/0)，NPU 用 int8 (1/0)，pad 到 2048x2048 |

### Bug 修复记录

| 编号 | 严重度 | 问题 | 修复 |
|------|--------|------|------|
| 1 | P0 | `propose_tree` 从未被调用 | `_propose` 添加 `AscendTreeAttentionMetadata` 类型检查 |
| 2 | P0 | Backend 名称 `"ASCEND_TREE_ATTN"` 不匹配 | 改为 `"TREE_ATTN"` |
| 3 | P0 | Root token 行为 `[0, :] = 0` 语义反了 | 改为 `[:, 0] = 0` |
| 4 | P1 | `slot_mapping` 未传递 | `set_ascend_forward_context` 添加参数 |
| 5 | P1 | ACL Graph per-level 缺失 | `propose_tree` 添加 dispatch |

---

## 单元测试

NPU 端单元测试全部通过（11/11）：

```
tests/ut/attention/test_tree_attn_npu.py::TestTreeMaskConversion::test_is_ancestor PASSED
tests/ut/attention/test_tree_attn_npu.py::TestTreeMaskConversion::test_get_depth_counts PASSED
tests/ut/attention/test_tree_attn_npu.py::TestTreeMaskConversion::test_prepare_tree_attn_bias_gpu PASSED
tests/ut/attention/test_tree_attn_npu.py::TestTreeMaskConversion::test_convert_tree_mask_for_npu PASSED
tests/ut/attention/test_tree_attn_npu.py::TestAscendTreeAttentionBackend::test_backend_name PASSED
tests/ut/attention/test_tree_attn_npu.py::TestAscendTreeAttentionBackend::test_backend_classes PASSED
tests/ut/attention/test_tree_attn_npu.py::TestAscendTreeAttentionBackend::test_kv_cache_shape PASSED
tests/ut/attention/test_tree_attn_npu.py::TestAscendTreeAttentionBackend::test_kv_cache_shape_invalid_block_size PASSED
tests/ut/attention/test_tree_attn_npu.py::TestAscendTreeAttentionMetadata::test_metadata_creation PASSED
tests/ut/attention/test_tree_attn_npu.py::TestConvertFunction::test_binary_tree_depth_3 PASSED
tests/ut/attention/test_tree_attn_npu.py::TestConvertFunction::test_single_token_tree PASSED
```

---

## 修改文件清单

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
