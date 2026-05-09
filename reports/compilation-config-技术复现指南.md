# vLLM Ascend `--compilation-config` 参数适配 — 技术复现指南

## 一、任务概述

为 vLLM Ascend 适配以下两个 vLLM 启动参数的子参数：

- **`--compilation-config mode`**（CompilationMode 枚举）
- **`--compilation-config dynamic_shapes_config`**（DynamicShapesConfig 配置对象）

目标：让用户设置的这两个参数在 Ascend NPU 平台上**正确生效或给出明确的错误提示**，消除当前"静默忽略"的问题。

---

## 二、代码架构与调用链

### 2.1 核心文件索引

| 文件 | 角色 |
|---|---|
| `vllm/vllm/config/compilation.py` | 上游定义 `CompilationMode`、`DynamicShapesConfig`、`CompilationConfig`、`CUDAGraphMode` |
| `vllm/vllm/compilation/wrapper.py` | 上游编译包装器，使用 `dynamic_shapes_config` 控制 guard 行为 |
| `vllm/vllm/compilation/backends.py` | 上游 `VllmBackend`，`VLLM_COMPILE` 模式的核心后端 |
| `vllm/vllm/compilation/compiler_interface.py` | 上游 `CompilerInterface` 基类 |
| `vllm_ascend/platform.py` | Ascend 平台入口，`check_and_update_config` 处理所有配置 |
| `vllm_ascend/compilation/compiler_interface.py` | Ascend 自定义编译器 `AscendCompiler` |
| `vllm_ascend/compilation/acl_graph.py` | Ascend ACL Graph 运行时包装器 `ACLGraphWrapper` |
| `vllm_ascend/compilation/graph_fusion_pass_manager.py` | Ascend 图融合 pass 管理器 |

### 2.2 完整调用链

```
用户设置 --compilation-config mode=STOCK_TORCH_COMPILE dynamic_shapes_config="{...}"
    │
    ▼
vllm/config/compilation.py  CompilationConfig 解析参数
    │
    ▼
vllm_ascend/platform.py  NPUPlatform.check_and_update_config()
    │  ├── 当前行为：检查 mode 值，非 NONE/VLLM_COMPILE 则 warning + cudagraph_mode=NONE
    │  ├── 当前行为：完全忽略 dynamic_shapes_config
    │  ├── 目标行为：验证 mode 合法性 → 正确传递或明确报错
    │  └── 目标行为：读取 dynamic_shapes_config → 验证兼容性
    │
    ▼
vllm/compilation/backends.py  VllmBackend.__call__()
    │  ├── split_graph() 分段
    │  ├── PiecewiseCompileInterpreter 编译子模块
    │  └── 包装为 VllmSerializableFunction
    │
    ▼
vllm/compilation/wrapper.py  CompiledGraphWrapper.__call__()
    │  ├── 读取 dynamic_shapes_config.evaluate_guards
    │  ├── 读取 dynamic_shapes_config.type
    │  ├── 根据 type 处理 guard（drop/keep）
    │  └── 调用 torch.compile(dynamic=False, ...)
    │
    ▼
vllm_ascend/compilation/compiler_interface.py  AscendCompiler.compile()
    │  ├── fusion_pass_compile：GraphFusionPassManager + compile_fx + aot_autograd
    │  └── npugraph_ex_compile：torchair ACL Graph
    │
    ▼
vllm_ascend/compilation/acl_graph.py  ACLGraphWrapper.__call__()
    │  ├── 按 BatchDescriptor 捕获/回放 ACL Graph
    │  └── 运行期处理动态 shape（多图捕获）
```

---

## 三、`--compilation-config mode` 详细分析

### 3.1 上游定义

**文件**: `vllm/vllm/config/compilation.py`

```python
class CompilationMode(enum.IntEnum):
    NONE = 0                # 纯 eager，不使用 torch.compile
    STOCK_TORCH_COMPILE = 1 # 标准 torch.compile 流水线
    DYNAMO_TRACE_ONCE = 2   # 单次 Dynamo trace，移除 guards
    VLLM_COMPILE = 3        # vLLM 自定义编译后端（分段编译、shape 特化）
```

### 3.2 当前 Ascend 行为

**文件**: `vllm_ascend/platform.py`，方法 `check_and_update_config`

关键代码段（约 296-308 行）：

```python
# enforce_eager 处理：强制 NONE
if enforce_eager:
    compilation_config.mode = CompilationMode.NONE

# mode 兼容性检查：非 NONE/VLLM_COMPILE 则降级
if compilation_config.mode not in [CompilationMode.NONE, CompilationMode.VLLM_COMPILE]:
    logger.warning("NPU does not support %s compilation mode. Setting CUDAGraphMode to NONE",
                   compilation_config.mode)
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE  # ← 静默降级！
```

关键代码段（约 350-401 行，CUDAGraphMode 分发）：

```python
# CUDAGraphMode.NONE → CompilationMode.NONE
if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
    compilation_config.mode = CompilationMode.NONE

# CUDAGraphMode.PIECEWISE → 要求 VLLM_COMPILE
elif compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
    assert compilation_config.mode == CompilationMode.VLLM_COMPILE
    compilation_config.use_inductor = False
    compilation_config.splitting_ops.extend(["vllm::mla_forward"])

# FULL_DECODE_ONLY / FULL → use_inductor = False
elif compilation_config.cudagraph_mode in [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL]:
    compilation_config.use_inductor = False
    compilation_config.splitting_ops = []
```

### 3.3 问题总结

| 用户设置 | 当前行为 | 问题 |
|---|---|---|
| `mode=NONE` | 正常工作 | 无问题 |
| `mode=VLLM_COMPILE` | 正常工作（配合 cudagraph_mode） | 无问题 |
| `mode=STOCK_TORCH_COMPILE` | warning + cudagraph_mode=NONE → eager | **静默降级** |
| `mode=DYNAMO_TRACE_ONCE` | warning + cudagraph_mode=NONE → eager | **静默降级** |

### 3.4 适配目标

1. 对 `STOCK_TORCH_COMPILE` 和 `DYNAMO_TRACE_ONCE`，**不再静默降级**，改为抛出明确的 `ValueError`，告知用户当前 Ascend 平台仅支持 `NONE` 和 `VLLM_COMPILE`，并推荐替代方案
2. 探索 `STOCK_TORCH_COMPILE` 在 Ascend 上的可行性（是否可将 Inductor 替换为 AscendCompiler）

### 3.5 适配点

**主要修改文件**: `vllm_ascend/platform.py`，方法 `check_and_update_config`

**修改位置**: 约 304-308 行，将 warning 改为 raise ValueError：

```python
# 当前代码
if compilation_config.mode not in [CompilationMode.NONE, CompilationMode.VLLM_COMPILE]:
    logger.warning("NPU does not support %s compilation mode...", ...)
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE

# 修改为
if compilation_config.mode not in [CompilationMode.NONE, CompilationMode.VLLM_COMPILE]:
    raise ValueError(
        f"Ascend NPU currently does not support {compilation_config.mode} compilation mode. "
        f"Supported modes: NONE (eager), VLLM_COMPILE (ACL Graph). "
        f"Please set --compilation-config mode=VLLM_COMPILE or mode=NONE."
    )
```

---

## 四、`--compilation-config dynamic_shapes_config` 详细分析

### 4.1 上游定义

**文件**: `vllm/vllm/config/compilation.py`（295-356 行）

```python
class DynamicShapesType(str, enum.Enum):
    BACKED = "backed"                    # 默认，PyTorch 标准行为，有 guards
    UNBACKED = "unbacked"                # 无 guards，可能抛 data-dependent 错误
    BACKED_SIZE_OBLIVIOUS = "backed_size_oblivious"  # 实验性

class DynamicShapesConfig:
    type: DynamicShapesType = DynamicShapesType.BACKED
    evaluate_guards: bool = False         # 调试模式，检测 Dynamo shape 特化
    assume_32_bit_indexing: bool = False  # 32 位索引假设
```

### 4.2 上游使用方式

**文件**: `vllm/vllm/compilation/wrapper.py`（约 106-160 行）

```python
# 读取配置
evaluate_guards = self.vllm_config.compilation_config.dynamic_shapes_config.evaluate_guards
ds_type = self.vllm_config.compilation_config.dynamic_shapes_config.type

# 根据 type 处理 guards
if ds_type == DynamicShapesType.UNBACKED:
    assert not evaluate_guards
    assert os.environ.get("VLLM_USE_BYTECODE_HOOK", "1") == "0"

# guard 过滤逻辑
if mode != CompilationMode.STOCK_TORCH_COMPILE:
    # 非标准模式下，drop 所有 guards
    guard_filter_fn = torch.compiler.skip_all_guards_unsafe
elif evaluate_guards:
    # 调试模式下，保留 SHAPE_ENV guards
    guard_filter_fn = lambda guards: [g for g in guards if g.guard_types[0] == "SHAPE_ENV"]

# 调用 torch.compile
torch.compile(..., dynamic=False, ...)
```

### 4.3 当前 Ascend 行为

**完全未处理。** 搜索整个 `vllm_ascend` 代码库：

- `AscendCompiler.compile()`（`compiler_interface.py`）：**不读取** `dynamic_shapes_config`
- `ACLGraphWrapper.__call__()`（`acl_graph.py`）：**不读取** `dynamic_shapes_config`
- `check_and_update_config`（`platform.py`）：**不读取** `dynamic_shapes_config`

Ascend 的动态 shape 处理完全依赖 **`ACLGraphWrapper` 的运行时多图捕获**：
- 为每个 `BatchDescriptor`（包含 batch size 信息）捕获独立的 ACL Graph
- 新的 batch size → 捕获新图
- 已知的 batch size → 回放已捕获的图

这与上游的编译期 `DynamicShapesConfig` 是**两套独立的机制**。

### 4.4 核心差距

| 维度 | 上游 vLLM (CUDA) | vLLM Ascend |
|---|---|---|
| 动态 shape 策略 | 编译期配置（`DynamicShapesConfig`） | 无配置，运行时硬编码 |
| guard 处理 | 根据 `type` 选择 drop/keep/evaluate | 不处理 guard |
| `torch.compile` 参数 | `dynamic=False` + guard_filter_fn | 不经过 `wrapper.py` 的编译路径 |
| shape 特化 | `BACKED` 下可特化 0/1 | 不感知特化策略 |
| 符号 shape | 通过 `mark_dynamic` 标记维度 | 不使用 `mark_dynamic` |

### 4.5 适配目标

1. **最小目标**：在 `check_and_update_config` 中读取 `dynamic_shapes_config`，对不兼容的配置给出明确 warning 或报错，不再静默忽略
2. **进阶目标**：将 `dynamic_shapes_config.type` 的配置传递给 `AscendCompiler`，影响 `torchair` 或 `fusion_pass_compile` 的行为
3. **理想目标**：让 `BACKED` 类型的 symbolic shape 机制与 `ACLGraphWrapper` 的多图捕获结合，减少图捕获数量

### 4.6 适配点

#### 适配点 1：`platform.py` — 配置验证

在 `check_and_update_config` 中添加：

```python
dynamic_shapes_config = vllm_config.compilation_config.dynamic_shapes_config
if dynamic_shapes_config.type == DynamicShapesType.UNBACKED:
    logger.warning(
        "UNBACKED dynamic shapes type may not be fully supported on Ascend NPU. "
        "ACLGraphWrapper handles dynamic shapes at runtime via multi-graph capture. "
        "Falling back to BACKED."
    )
    dynamic_shapes_config.type = DynamicShapesType.BACKED

if dynamic_shapes_config.evaluate_guards:
    logger.warning(
        "evaluate_guards=True requires torch.compile guard infrastructure "
        "which is not available in Ascend's ACLGraphWrapper path. "
        "Setting to False."
    )
    dynamic_shapes_config.evaluate_guards = False

if dynamic_shapes_config.assume_32_bit_indexing:
    # 需要验证 Ascend NPU 是否支持 32 位索引
    logger.info("assume_32_bit_indexing is enabled. Verify Ascend NPU compatibility.")
```

#### 适配点 2：`compiler_interface.py` — 传递配置

在 `AscendCompiler.compile()` 中读取 `dynamic_shapes_config`：

```python
def compile(self, graph, example_inputs, compiler_config, compile_range):
    dynamic_shapes_config = self.vllm_config.compilation_config.dynamic_shapes_config

    # 根据 type 影响编译行为
    if dynamic_shapes_config.type == DynamicShapesType.BACKED:
        # BACKED: 允许 torchair 或 compile_fx 进行 shape 特化
        pass
    elif dynamic_shapes_config.type == DynamicShapesType.BACKED_SIZE_OBLIVIOUS:
        # 实验性：对 torchair 的影响待验证
        pass

    # ... 原有逻辑
```

#### 适配点 3：`acl_graph.py` — 运行期配置

在 `ACLGraphWrapper` 中考虑 `dynamic_shapes_config` 的影响：

```python
def __call__(self, *args, **kwargs):
    dynamic_shapes_config = self.vllm_config.compilation_config.dynamic_shapes_config

    # 如果 type == BACKED_SIZE_OBLIVIOUS，可以尝试用 symbolic shape 
    # 合并相近的 batch size 图，减少捕获数量
    # 具体策略需要根据 torchair 的 symbolic shape 支持情况确定
```

---

## 五、风险点与缓解措施

### 5.1 `STOCK_TORCH_COMPILE` 适配风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| Inductor 的某些 pass 在 Ascend 上没有等价实现，无法完整替代 | 高 | 优先实现"明确报错"，适配为可选项 |
| `torchair` 不支持 `dynamic=False` 的行为 | 中 | 在 NPU 环境上实测验证 |
| 改为 `raise ValueError` 可能破坏现有依赖静默降级的测试 | 中 | 先搜索所有 `STOCK_TORCH_COMPILE` 相关测试用例 |

### 5.2 `dynamic_shapes_config` 适配风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| `torchair` 不支持 symbolic shape 推导 | 高 | 先做"配置验证+报错"，symbolic shape 为探索性工作 |
| `BACKED_SIZE_OBLIVIOUS` 在 Ascend 上的行为未知 | 高 | 仅做 warning 处理，不强行适配 |
| 修改 `ACLGraphWrapper` 可能影响现有图捕获逻辑 | 中 | 添加配置检查，不影响现有行为 |
| `assume_32_bit_indexing` 在 Ascend 上不成立 | 低 | 仅做 warning，不修改核心逻辑 |

### 5.3 通用风险

| 风险 | 缓解 |
|---|---|
| NPU 算力不足，无法做端到端验证 | 使用 `COMPILE_CUSTOM_KERNELS=0` 先在 CPU 上跑 UT |
| 上游 vLLM 版本变更导致 API 变化 | 锁定 vLLM v0.11.0 tag |
| PR review 周期长 | 提前与社区沟通，在 SIG-Ascend 频道说明方案 |

---

## 六、验证方案

### 6.1 单元测试

```bash
# 在 vllm-ascend 目录下
pytest -sv tests/ut/compilation/  # 如果有现有编译相关测试

# 新建测试文件
tests/ut/compilation/test_compilation_mode.py
tests/ut/compilation/test_dynamic_shapes_config.py
```

### 6.2 验证场景

#### `mode` 验证

| 场景 | 预期行为 |
|---|---|
| `mode=NONE` | 正常 eager 运行 |
| `mode=VLLM_COMPILE` + `cudagraph_mode=PIECEWISE` | ACL Graph 分段编译 |
| `mode=STOCK_TORCH_COMPILE` | 抛出 `ValueError`，提示使用 `VLLM_COMPILE` |
| `mode=DYNAMO_TRACE_ONCE` | 抛出 `ValueError`，提示使用 `VLLM_COMPILE` |
| `mode=VLLM_COMPILE` + `cudagraph_mode=NONE` | 正常 eager（mode 被覆盖为 NONE） |

#### `dynamic_shapes_config` 验证

| 场景 | 预期行为 |
|---|---|
| 默认 `BACKED` | 正常工作（现有行为不变） |
| `UNBACKED` | warning，回退到 `BACKED` |
| `evaluate_guards=True` | warning，强制设为 `False` |
| `assume_32_bit_indexing=True` | info log，验证兼容性 |

### 6.3 端到端测试

```bash
# 需要 NPU 环境
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --compilation-config '{"mode": "VLLM_COMPILE"}' \
  --additional-config '{"ascend_compilation_config": {}}'

# 测试 mode 报错
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --compilation-config '{"mode": "STOCK_TORCH_COMPILE"}'
# 应抛出 ValueError
```

---

## 七、开发时间建议

| 阶段 | 周期 | 内容 |
|---|---|---|
| Phase 1 | 1-2 周 | 读源码，跑通现有编译流程，写验证场景 |
| Phase 2 | 1 周 | 实现 `mode` 的明确报错 + 单元测试 |
| Phase 3 | 1-2 周 | 实现 `dynamic_shapes_config` 配置验证 + 传递 |
| Phase 4 | 1 周 | 端到端验证 + benchmark + PR 提交 |

**关键决策点**: Phase 1 结束时，根据 `torchair` 对 symbolic shape 的支持情况，决定 `dynamic_shapes_config` 的进阶目标（symbolic shape 集成）是否可行。

---

## 八、参考资源

- vLLM 编译流水线设计文档：`vllm/docs/design/torch_compile.md`
- vLLM 动态 shape 测试：`vllm/tests/compile/test_dynamic_shapes_compilation.py`
- Ascend ACL Graph 文档：`https://www.hiascend.com/document/detail/zh/Pytorch/`
- torchair 编译后端：Ascend `torchair` 包，`CompilerConfig` API
