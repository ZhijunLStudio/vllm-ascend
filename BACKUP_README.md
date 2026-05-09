# vLLM Ascend 适配赛 — 机器备份记录

> 备份日期：2026-05-09  
> 机器到期前备份：代码、文档、脚本（不含模型权重和编译产物）

## 环境版本

| 组件 | 版本 |
|------|------|
| NPU | Ascend 910B2C |
| CANN | 8.5.1 |
| Driver | 25.5.1 |
| Python | 3.11.6 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0 |
| torchair | 0.1 |
| triton-ascend | 3.2.0 |
| vllm | main (0f3ce4c74), 0.19.0 dev |
| vllm-ascend | main (gef94377), 0.19.0 dev |
| OS | Linux x86_64, kernel 5.15.0 |

## 四个参数总结

### 参数1：speculative_token_tree
- **分支**：`feature/speculative-token-tree-npu`（本地）/ `ZhijunLStudio:feature/speculative-token-tree-npu`（GitHub）
- **PR**：https://github.com/vllm-project/vllm-ascend/pull/8408 (CLOSED)
- **做了什么**：在 Ascend NPU 上实现 EAGLE 树结构推测解码。注册 `TREE_ATTN` backend（686行），替代 GPU FlashAttention 为 `npu_fused_infer_attention_score`；扩展 EagleProposer 支持多层级树 draft；GPU→NPU mask 格式转换（float32→int8，pad 2048×2048）
- **性能**：NPU +14.3% (tree n=2) vs CUDA +15.6%，差距 1.3pp
- **改动文件**（7个）：tree_attn.py, eagle_proposer.py, attention_v1.py, attention_mask.py, ascend_forward_context.py, backends/__init__.py, test_tree_attn_npu.py
- **报告**：`/root/work/speculative_token_tree_implementation_report.md`

### 参数2：cpu_offload_params
- **分支**：`feature/cpu-offload-params`（本地，已推送 GitHub `ZhijunLStudio/vllm-ascend`）
- **PR**：未提交（待后续）
- **做了什么**：适配 prefetch 后端到 NPU，替换 `torch.cuda.Stream/Event` 为 `torch.npu.*`；三级 factory patch 确保 NPUPrefetchOffloader 生效；ACL Graph 集成（sync_prev_onload / join_after_forward）；UVA 降级为 NoopOffloader
- **性能**：eager 541→78 tok/s；graph 475→70 tok/s；所有 output_match=true
- **改动文件**（11个）：npu_prefetch.py, patch_offloader.py, acl_graph.py, platform.py, model_runner_v1.py, model_runner_310p.py, worker.py 等
- **报告**：`/root/work/vllm-ascend/param-selection/cpu-offload-params-implementation-report.md`
- **测试脚本**：`param-selection/run_cuda_offload_tests.py`（统一 CUDA+NPU 测试框架，支持 --mode eager|graph|both）

### 参数3：compilation-config mode
- **分支**：`feature/pr-compilation-config-v2`（本地）/ `ZhijunLStudio:feature/pr-compilation-config-v2`（GitHub）
- **PR**：https://github.com/vllm-project/vllm-ascend/pull/8229 (OPEN)
- **做了什么**：支持全部四种 CompilationMode。STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE 不再静默降级，透传给 vLLM 上层；get_compile_backend() → "npu"
- **改动文件**（4个）：platform.py, compiler_interface.py, utils.py, test_platform.py
- **报告**：在 05-compilation-config 合集报告中

### 参数4：compilation-config dynamic_shapes_config
- **分支**：同上（`feature/pr-compilation-config-v2`）
- **PR**：同上 #8229
- **做了什么**：dynamic_shapes_config 不再静默忽略。VLLM_COMPILE 模式：UNBACKED→BACKED 回退(warning)，evaluate_guards→False(warning)；其他模式透传
- **改动文件**：同上
- **报告**：在 05-compilation-config 合集报告中

## 分支关系图

```
feature/speculative-token-tree-npu   (4 commits) → PR #8408 → 报告03
feature/cpu-offload-params           (7 commits) → 未提PR   → 报告04
feature/pr-compilation-config-v2     (1 commit)  → PR #8229 → 报告05(mode+dynshape)
feature/full-compilation-config-support (5 commits) → PR #8229的完整版(含测试+SKILL)
feature/tree-spec-decode-cuda-verify (6 commits) → tree优化的CUDA验证分支
feature/compilation-config-support   (0 commits) → 空分支(同main)
```

## 重要文件清单

### 项目文档
| 文件 | 大小 | 说明 |
|------|------|------|
| `/root/work/项目申请书.md` | 13KB | 参赛项目申请书 |
| `/root/work/compilation-config-技术复现指南.md` | 15KB | compilation-config 适配技术指南 |
| `/root/work/speculative_token_tree_implementation_report.md` | 25KB | tree attention 技术报告 |
| `/root/work/cpu_offload_params_implementation_report.md` | 27KB | offload 技术报告 |
| `/root/work/PR_DESCRIPTION.md` | 2KB | offload PR 描述 |
| `/root/work/PR_DESCRIPTION_TREE_ATTN.md` | 4KB | tree PR 描述 |

### 测试脚本（已归档到竞赛仓库 skills/ 中）
| 原始位置 | 大小 | 说明 |
|---------|------|------|
| `param-selection/run_cuda_offload_tests.py` | 15KB | offload 跨平台测试（CUDA+NPU） |
| `run_compilation_tests.sh` | 309行 | compile NPU 批量测试 |
| `run_compilation_tests_cuda.sh` | 307行 | compile CUDA 对照测试 |
| `compilation_config_batch_test.py` | 650行 | compile 批量测试 Python 版 |
| `tests/ut/attention/test_tree_attn_npu.py` | 298行 | tree UT |

### 已推送到竞赛仓库的内容
- GitLink fork：`https://gitlink.org.cn/ZhijunLStudio/vllm-ascend-competition`
- PR #1：已提交，含 source_adaptations 03/04/05 + 3个 skills
- GitHub vllm-ascend PR：#8229 (OPEN), #8408 (CLOSED)

## 已推送的仓库

| 仓库 | URL |
|------|-----|
| vllm-ascend (GitHub) | https://github.com/ZhijunLStudio/vllm-ascend |
| vllm-ascend-competition (GitLink) | https://gitlink.org.cn/ZhijunLStudio/vllm-ascend-competition |
