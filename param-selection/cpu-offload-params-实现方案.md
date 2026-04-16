# cpu-offload-params 实现方案

## 背景

vLLM 上游提供了完整的 V2 UVA/Prefetch weight offloader 框架，支持通过 `--cpu-offload-params` 等 CLI 参数控制大模型权重在 HBM 和 CPU 之间的分组卸载与智能预取。

当前 Ascend 平台：
- 仅有独立的 `vllm_ascend/ops/weight_prefetch.py`（45行），手动调用 `torch_npu` 预取
- `vllm_ascend/patch/worker/patch_v2/patch_uva.py`（85行）做了 UVA buffer 的基础 patch
- `model_runner_v1.py` 中 `offload_config.uva.cpu_offload_gb == 0` 被 assert，offloader 从未被调用
- **完全没有接入 vLLM 的 OffloadConfig 参数体系**

## GPU 端参考实现

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/config/offload.py` | 139行 | OffloadConfig, UVAOffloadConfig, PrefetchOffloadConfig |
| `vllm/model_executor/offloader/__init__.py` | 22行 | create_offloader 工厂函数 |
| `vllm/model_executor/offloader/uva.py` | 105行 | UVAOffloadingManager - CUDA 流 + 内存管理 |
| `vllm/model_executor/offloader/prefetch.py` | 128行 | PrefetchOffloadingManager - 预取调度 |
| `vllm/v1/worker/gpu/buffer_utils.py` | 60行 | UvaBuffer - CUDA caching allocator |

## 实现方案

### 第一步：NPU Offloader 框架

新建 `vllm_ascend/model_executor/offloader/` 目录：

```
vllm_ascend/model_executor/offloader/
├── __init__.py          # 注册 NPU offloader 到 create_offloader
├── uva.py               # NPUUVAOffloadingManager
└── prefetch.py          # NPUPrefetchOffloadingManager
```

**关键替换：**
- `torch.cuda.Stream` → `torch.npu.Stream`
- `torch.cuda.memory_allocated` → `torch.npu.memory.memory_allocated`
- `torch.cuda.empty_cache` → `torch.npu.empty_cache`
- CUDA pinned memory → NPU pinned memory
- CUDA caching allocator → NPU 等价实现

### 第二步：NPU UvaBuffer

新建 `vllm_ascend/worker/buffer_utils.py`：
- 替换 GPU 端的 `torch.cuda.caching_allocator_alloc` 为 NPU 等价
- 复用已有 `patch_uva.py` 中的 NPU UVA 逻辑

### 第三步：接入 model_runner_v1

修改 `vllm_ascend/worker/model_runner_v1.py`：
- 移除 `assert offload_config.uva.cpu_offload_gb == 0`
- 在 `__init__` 中调用 `create_offloader` 创建 NPU offloader
- 在 forward 流程中集成 offload/prefetch/release 逻辑
- 和现有 `weight_prefetch.py` 统一

### 第四步：配置验证

修改 `vllm_ascend/platform.py` 的 `check_and_update_config`：
- 验证 OffloadConfig 参数在 NPU 上的合法性
- 设置 NPU 特有的默认值

### 第五步：测试

- 单元测试：offloader 创建、内存分配、流管理
- E2E 测试：大模型（70B）在 offload 模式下的推理正确性和性能

## 性能展示方案

**测试场景：**
- 模型：Llama-3-70B（或同级别大模型）
- 基线：不启用 offload（OOM 或 batch=1）
- 对比：启用 `--cpu-offload-params '{"num_layers": 40}'`
- 指标：最大 batch size、吞吐量（tokens/s）、首 token 延迟

**预期效果：**
- 原本 OOM 的模型能跑起来
- batch size 可以扩大
- 性能指标（30%权重）直接拿高分
