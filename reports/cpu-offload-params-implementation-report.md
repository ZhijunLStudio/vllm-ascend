# cpu-offload-params Ascend NPU 实现技术报告

## 一、项目概述

在 Ascend NPU 上实现 `--cpu-offload-params` prefetch 后端，使 NPU 支持 CPU 参数卸载（将部分层参数放在 CPU 内存中，按需异步预取到 NPU HBM），功能和输出对齐原生 vLLM。

**背景**：原生 vLLM 已完整支持 prefetch 后端，但所有 `torch.cuda.Stream/Event` API 需替换为 `torch.npu.*`；CUDA Graph 集成点需改为 ACL Graph；UVA（统一虚拟寻址）NPU 不支持。

**设计理念**：用吞吐换显存。参数卸载后吞吐降到 ~0.15x，但释放的显存可给 KV cache，支持更大 batch 或更长序列。

## 二、测试模型

| 角色 | 模型 | 说明 |
|------|------|------|
| 测试模型 | Meta-Llama-3.1-8B-Instruct | 32 层 Transformer，约 8B |

## 三、Offload 配置

| 参数 | 含义 |
|------|------|
| `--offload-group-size N` | 每 N 层为一组 |
| `--offload-num-in-group M` | 每组最后 M 层被 offload |
| `--offload-prefetch-step S` | 预取步长（同时预取 S 组） |
| `--offload-backend` | 后端：prefetch / uva / auto |

### 测试配置矩阵（32 层 Llama-3.1-8B）

| 配置名 | group | num | step | offload 层数 | 占比 |
|--------|-------|-----|------|-------------|------|
| baseline | 0 | 0 | 0 | 0 | 0% |
| g8n2s1 | 8 | 2 | 1 | 16 | 50% |
| g8n2s2 | 8 | 2 | 2 | 16 | 50% |
| g4n1s1 | 4 | 1 | 1 | 8 | 25% |
| g4n1s2 | 4 | 1 | 2 | 8 | 25% |

## 四、实现架构

### 4.1 文件结构

```
vllm_ascend/
├── offloader/
│   ├── __init__.py                    # 包初始化
│   └── npu_prefetch.py                # 核心（~280行）
├── patch/worker/
│   ├── __init__.py                    # 导入 patch
│   └── patch_offloader.py             # 工厂 patch（~90行）
├── compilation/acl_graph.py           # ACL Graph 集成
├── platform.py                        # is_uva_available()=False
├── worker/model_runner_v1.py          # post_init() 调用
└── worker/worker.py                   # 提前导入 patch
tests/
├── ut/offloader/test_npu_prefetch_offloader.py  # 13 个 UT
└── e2e/singlecard/test_cpu_offload_params.py    # 7 个 E2E
```

### 4.2 NPU 与 GPU 核心差异

| 层面 | GPU | NPU |
|------|-----|-----|
| Stream / Event | `torch.cuda.*` | `torch.npu.*` |
| 流捕获检测 | `torch.cuda.is_current_stream_capturing()` | `torch.npu.is_current_stream_capturing()` |
| Graph | CUDA Graph | ACL Graph |
| UVA 支持 | ✅ | ❌（降级 NoopOffloader） |

### 4.3 核心组件

**NPUPrefetchOffloader**（`npu_prefetch.py`）：与 GPU 版结构一致，`torch.cuda.*` 全部替换为 `torch.npu.*`。
- `wrap_modules`：按 group 选层 + forward hook 注入 wait/start prefetch
- `post_init`：分配 StaticBufferPool，为每个参数分配 buffer slot
- `sync_prev_onload` / `join_after_forward`：ACL Graph 同步接口

**_NPUModuleOffloader**：管理单层参数 offload。核心方法 `start_onload_to_static` 用 event-based stream forking 实现 ACL Graph 兼容。

**ACL Graph 集成**（`acl_graph.py`）：三个集成点 — capture 前 sync_prev_onload、forward 后 join_after_forward、replay 前 sync_prev_onload。

### 4.4 关键实现细节

**Event-based Stream Forking**：copy 在独立 copy_stream 上执行，通过 event 和主 stream 同步，不会被录进 graph。

**Capture-aware Wait**：capture 模式用 `wait_event`（graph 友好），eager 模式优先用 event 再 fallback 到 `wait_stream`。

**多层级 Patch**：patch `offloader_base` / `offloader_pkg` / `gpu_model_runner` 三级 `create_offloader`，确保所有 import 路径生效。

**Import 时序控制**：`worker.py` 中 `patch_offloader` 必须在 `model_runner_v1` 之前导入，确保 `gpu_model_runner` 捕获的是 patched 版本。

**post_init() 集成**：`NPUModelRunner.load_model()` 末尾补上 `get_offloader().post_init()`。

**UVA 降级**：`is_uva_available()` 返回 False，UVA 请求降级 NoopOffloader 并 warning。

## 五、正确性验证

### 5.1 单元测试（13 个，全部通过）

| 测试类 | 测试方法 | 内容 |
|--------|---------|------|
| TestNPUPrefetchOffloaderInit | test_creates_npu_stream / test_default_params / test_offload_params | 初始化 |
| TestWrapModules | test_layer_selection_group_4_num_1 / _2_num_1 / _4_num_2 | 层选择逻辑 |
| TestWrapModules | test_offload_params_whitelist / test_returns_all_modules / test_wrap_modules_called_twice_raises | 参数过滤 |
| TestSyncMethods | test_sync_prev_onload / test_join_after_forward / test_post_init | 空 offloader 安全 |
| TestCreateOffloaderPatch | 5 个 test: prefetch→NPU / uva→Noop / auto+group→NPU / auto+gb→Noop / empty→Noop | 工厂路由 |
| TestStaticBufferPoolNPU | test_allocate_on_npu / test_buffer_slot_reuse | Buffer 分配 |
| TestNPUModuleOffloader | test_creates_npu_event / test_param_offloaders_created / test_cpu_storage | Module offloader |

### 5.2 E2E 测试（7 个，全部通过）

| 测试 | 配置 | 结果 |
|------|------|------|
| test_prefetch_offload_eager [g4n1s1] | eager | ✅ |
| test_prefetch_offload_eager [g8n2s1] | eager | ✅ |
| test_prefetch_offload_eager [g4n1s2] | eager | ✅ |
| test_prefetch_offload_acl_graph [g4n1s1] | graph | ✅ |
| test_prefetch_offload_acl_graph [g8n2s1] | graph | ✅ |
| test_prefetch_correctness_eager | eager | ✅ |
| test_prefetch_correctness_graph | graph | ✅ |
| test_uva_backend_graceful_fallback | eager | ✅ |

### 5.3 输出一致性

所有配置 `output_match = true`（temperature=0.0）：

| 配置 | Eager | Graph |
|------|-------|-------|
| g8n2s1 | ✅ | ✅ |
| g8n2s2 | ✅ | ✅ |
| g4n1s1 | ✅ | ✅ |
| g4n1s2 | ✅ | ✅ |

## 六、性能测试

### 6.1 测试环境

| 项目 | CUDA | NPU |
|------|------|-----|
| 设备 | A800-SXM4-80GB | Ascend 910B2C |
| 模型 | Meta-Llama-3.1-8B | 同左 |
| 配置 | max_model_len=512, 10 prompts, temp=0.0, max_tokens=50 |

### 6.2 NPU 性能结果

#### Eager 模式

| 配置 | Tokens/s | Latency | vs Baseline |
|------|----------|---------|-------------|
| baseline | **541.0** | 0.772s | 1.00x |
| g8n2s1 | 78.5 | 6.205s | 0.15x |
| g8n2s2 | 81.3 | 6.024s | 0.15x |
| g4n1s1 | 78.5 | 6.204s | 0.15x |
| g4n1s2 | 81.3 | 6.024s | 0.15x |

#### ACL Graph 模式

| 配置 | Tokens/s | Latency | vs Baseline |
|------|----------|---------|-------------|
| baseline | **475.4** | 0.876s | 1.00x |
| g8n2s1 | 69.6 | 6.915s | 0.15x |
| g8n2s2 | 69.8 | 6.883s | 0.15x |
| g4n1s1 | 69.9 | 6.877s | 0.15x |
| g4n1s2 | 69.7 | 6.880s | 0.15x |

所有配置 output_match=true。

### 6.3 CUDA 性能结果

#### Eager 模式

| 配置 | Tokens/s | vs Baseline |
|------|----------|-------------|
| baseline | **727.4** | 1.00x |
| g8n2s1 | 67.1 | 0.09x |
| g8n2s2 | 69.8 | 0.10x |
| g4n1s1 | 68.8 | 0.09x |
| g4n1s2 | 70.2 | 0.10x |

#### CUDA Graph 模式

| 配置 | Tokens/s | vs Baseline |
|------|----------|-------------|
| baseline | **867.4** | 1.00x |
| g8n2s1 | 59.4 | 0.07x |
| g8n2s2 | 66.3 | 0.08x |
| g4n1s1 | 61.5 | 0.07x |
| g4n1s2 | 68.3 | 0.08x |

所有配置 output_match=true。

### 6.4 CUDA vs NPU 最终对比

| 指标 | CUDA (A800) | NPU (910B2C) | 备注 |
|------|------------|-------------|------|
| Eager baseline | 727.4 tok/s | 541.0 tok/s | CUDA 1.34x |
| Offload eager | 67-70 tok/s | 78-81 tok/s | NPU 略高 |
| Graph baseline | 867.4 tok/s | 475.4 tok/s | CUDA 1.82x |
| Offload graph | 59-68 tok/s | ~70 tok/s | 接近 |
| 输出一致性 | ✅ | ✅ | temp=0.0 完全一致 |

### 6.5 性能分析

1. Offload 后瓶颈在 CPU↔设备带宽，吞吐从数百 tok/s 降到 ~70 tok/s
2. NPU offload 吞吐略高（78-81 vs 67-70），因 NPU baseline 较低
3. 不同配置差异小：g8n2(50%) 和 g4n1(25%) 吞吐接近，瓶颈在最慢层
4. step=1 vs step=2 差异有限

## 七、CUDA Graph 兼容性分析

CUDA Graph + prefetch offload **已实测通过**。上游 `CUDAGraphWrapper` 有三处 offloader 集成点（sync_prev_onload / join_after_forward），prefetch 代码有 `is_current_stream_capturing()` 感知。NPU 端 ACL Graph 同样兼容，三层机制：event-based stream forking + capture-aware wait + join_after_forward。

此前报告"CUDA Graph 崩溃"结论有误，实际是显存 OOM，非代码兼容性问题。

## 八、Bug 修复记录

| 编号 | 严重度 | 问题 | 修复 |
|------|--------|------|------|
| 1 | P0 | create_offloader patch 未覆盖 gpu_model_runner 的 from-import 引用 | 三级 patch |
| 2 | P0 | NPUModelRunner.load_model() 缺少 post_init() | 末尾补上调用 |
| 3 | P1 | patch_offloader 导入时序晚于 model_runner_v1 | worker.py 提前导入 |
| 4 | P1 | npu_prefetch.py 文件损坏 | 重写 |

## 九、已对齐原生 vLLM 的功能清单

| 功能 | 文件 |
|------|------|
| prefetch → NPUPrefetchOffloader | `patch_offloader.py` |
| UVA → NoopOffloader 降级 | `patch_offloader.py` |
| auto 后端路由 | `patch_offloader.py` |
| `is_uva_available() = False` | `platform.py` |
| post_init() 集成 | `model_runner_v1.py` |
| ACL Graph sync_prev_onload / join_after_forward / replay sync | `acl_graph.py` |
| Event-based stream forking | `npu_prefetch.py` |
| Capture-aware wait | `npu_prefetch.py` |
| Forward hook (wait/start prefetch) | `npu_prefetch.py` |
| StaticBufferPool on NPU | 复用上游 |
| Eager + Graph 输出一致性 | E2E 测试 ✅ |

## 十、评测方法

### NPU

```bash
cd /root/work/vllm-ascend && git checkout feature/cpu-offload-params
export VLLM_VERSION=0.19.0 ASCEND_RT_VISIBLE_DEVICES=0
python param-selection/run_cuda_offload_tests.py \
  --model /data/models/Meta-Llama-3.1-8B-Instruct --mode both
```

### CUDA

```bash
git checkout feature/cpu-offload-params
export CUDA_VISIBLE_DEVICES=0
python param-selection/run_cuda_offload_tests.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct --mode both
```

## 十一、修改文件清单

| 文件 | 操作 |
|------|------|
| `vllm_ascend/offloader/__init__.py` | 新建 |
| `vllm_ascend/offloader/npu_prefetch.py` | 新建 |
| `vllm_ascend/patch/worker/patch_offloader.py` | 新建 |
| `tests/ut/offloader/test_npu_prefetch_offloader.py` | 新建 |
| `tests/e2e/singlecard/test_cpu_offload_params.py` | 新建 |
| `param-selection/run_cuda_offload_tests.py` | 新建 |
| `vllm_ascend/patch/worker/__init__.py` | 修改 |
| `vllm_ascend/compilation/acl_graph.py` | 修改 |
| `vllm_ascend/platform.py` | 修改 |
| `vllm_ascend/worker/model_runner_v1.py` | 修改 |
| `vllm_ascend/worker/worker.py` | 修改 |
| `vllm_ascend/_310p/model_runner_310p.py` | 修改 |

## 十二、结论

- NPU prefetch offload 完全对齐 GPU：prefetch 后端、UVA 降级、auto 路由、ACL Graph 集成
- Eager 和 ACL Graph 模式输出完全一致（temperature=0.0）
- 22 个 UT + 7 个 E2E 全部通过
- Offload 后吞吐 ~0.15x，是 offload 设计的本质 trade-off
- CUDA 和 NPU offload 吞吐接近（60-80 tok/s），瓶颈为 CPU↔设备带宽
