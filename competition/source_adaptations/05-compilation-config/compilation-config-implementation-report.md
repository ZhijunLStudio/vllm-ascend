# compilation-config Ascend NPU 实现技术报告


## 一、项目概述

### 1.1 目标

为 Ascend NPU 适配 `--compilation-config` 的两个子参数：
- **`mode`** — CompilationMode 枚举（NONE / VLLM_COMPILE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE），控制编译策略
- **`dynamic_shapes_config`** — DynamicShapesConfig 配置对象（type / evaluate_guards / assume_32_bit_indexing），控制动态 shape 处理

消除当前"静默忽略用户配置"的问题，让每个用户设置都能正确生效或给出明确提示。

### 1.2 背景

原生 vLLM 的 `--compilation-config` 支持四种 CompilationMode：
- `NONE`：纯 eager 模式，不编译
- `VLLM_COMPILE`：vLLM 自定义编译后端（分段编译、shape 特化），在 Ascend 上对应 ACL Graph
- `STOCK_TORCH_COMPILE`：标准 `torch.compile` 流水线
- `DYNAMO_TRACE_ONCE`：单次 Dynamo trace，移除 guards

在适配前，Ascend 平台仅支持 `NONE` 和 `VLLM_COMPILE` 两种模式，`STOCK_TORCH_COMPILE` 和 `DYNAMO_TRACE_ONCE` 会被**静默降级**为 eager，`dynamic_shapes_config` 则**完全被忽略**。

---

## 二、测试模型

| 角色 | 模型 | 说明 |
|------|------|------|
| NPU 端 | **Qwen/Qwen2.5-0.5B-Instruct** | 轻量模型，适合快速迭代测试 |
| CUDA 对照 | **Qwen/Qwen2.5-0.5B-Instruct** | 相同模型，确保对比公平 |

---

## 三、配置矩阵

测试覆盖 15 个场景：

**mode 参数（4 个）**：
| 测试 | mode | 预期行为 |
|------|------|---------|
| T1 | NONE | 纯 eager |
| T2 | VLLM_COMPILE | ACL Graph 分段编译 |
| T3 | STOCK_TORCH_COMPILE | `torch.compile(backend='npu')` |
| T4 | DYNAMO_TRACE_ONCE | 单次 Dynamo trace，backend='npu' |

**dynamic_shapes_config 参数（6 个）**：
| 测试 | type | evaluate_guards | 预期行为 |
|------|------|----------------|---------|
| T5 | BACKED | False | 正常工作 |
| T6 | UNBACKED (VLLM_COMPILE) | False | Warning + 回退到 BACKED |
| T7 | BACKED_SIZE_OBLIVIOUS | False | 正常工作 |
| T8 | BACKED | True (VLLM_COMPILE) | Warning + 设为 False |
| T9 | BACKED | True (DYNAMO) | 正常工作（透传） |
| T10 | — | assume_32_bit_indexing=True | Info log |

**组合测试（4 个）**：
| 测试 | mode | dynamic_shapes | 说明 |
|------|------|---------------|------|
| T11 | VLLM_COMPILE | UNBACKED | ACL Graph 回退 |
| T12 | VLLM_COMPILE | evaluate_guards=True | ACL Graph 回退 |
| T13 | DYNAMO_TRACE_ONCE | BACKED_SIZE_OBLIVIOUS | 正常透传 |
| T14 | STOCK_TORCH_COMPILE | UNBACKED | 正常透传 |

**错误处理（1 个）**：
| 测试 | 场景 | 预期 |
|------|------|------|
| T15 | mode=INVALID | ValidationError |

---

## 四、实现架构

### 4.1 文件结构

```
vllm_ascend/
├── platform.py                        # 核心：check_and_update_config（+200行）
├── compilation/
│   └── compiler_interface.py          # AscendCompiler 修正
├── utils.py                           # get_compile_backend 返回 "npu"
tests/ut/
└── test_platform.py                   # 35+ 单元测试
```

### 4.2 核心改动

#### 改动 1：支持所有四种 CompilationMode（platform.py）

**之前行为**：`STOCK_TORCH_COMPILE` 和 `DYNAMO_TRACE_ONCE` 被静默降级为 eager：
```python
# 旧代码
if compilation_config.mode not in [NONE, VLLM_COMPILE]:
    logger.warning("NPU does not support %s ...", ...)
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE  # ← 静默降级
```

**现在行为**：
- `VLLM_COMPILE`：使用 ACL Graph 分段编译（`cudagraph_mode=PIECEWISE`，`use_inductor=False`）
- `STOCK_TORCH_COMPILE` / `DYNAMO_TRACE_ONCE`：直接传递给 vLLM 上层，`cudagraph_mode=NONE`，backend 设为 `"npu"`
- `NONE`：纯 eager

#### 改动 2：dynamic_shapes_config 智能处理（platform.py）

**VLLM_COMPILE 模式**（ACL Graph 静态捕获）：
- `UNBACKED` → 回退到 `BACKED` + warning（ACL Graph 不支持 unbacked shapes）
- `evaluate_guards=True` → 设为 `False` + warning（ACL Graph 路径不支持 guard 评估）

**其他模式**（NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE）：
- 直接透传，不修改
- 由 vLLM 上层 `TorchCompileWithNoGuardsWrapper` 处理

#### 改动 3：get_compile_backend → "npu"（utils.py）

```python
def get_compile_backend() -> str:
    return "npu"  # torch_npu 提供的 Inductor 后端
```

`simple_compile_backend` 也同步改为 `"npu"`。

### 4.3 四种模式在 NPU 上的机制

| mode | 底层机制 | cudagraph_mode | backend |
|------|---------|---------------|---------|
| NONE | 纯 eager | NONE | — |
| VLLM_COMPILE | VllmBackend → AscendCompiler → TorchAIR ACL Graph | PIECEWISE | "eager" (EagerAdaptor) |
| STOCK_TORCH_COMPILE | torch.compile(backend="npu") | NONE | "npu" (Inductor) |
| DYNAMO_TRACE_ONCE | Single Dynamo trace, backend="npu" | NONE | "npu" (Inductor) |

### 4.4 关键设计决策

#### 决策 1：平台层只做兼容性检查，不做运行时实现

`dynamic_shapes_config` 的运行时行为完全由 vLLM 上层的 `TorchCompileWithNoGuardsWrapper`（`wrapper.py`）和 `decorators.py` 处理。CUDA 平台也不处理它。Ascend 平台层只负责 VLLM_COMPILE 模式的兼容性检查——因为 ACL Graph 是静态图，与 UNBACKED 语义冲突。

#### 决策 2：UNBACKED 回退而非报错

当用户在 VLLM_COMPILE 模式下设置 UNBACKED 时，主动回退到 BACKED 并给出 warning 和替代方案建议，比直接报错更友好：
```
WARNING: UNBACKED dynamic shapes type is incompatible with ACL Graph capture.
Falling back to BACKED. Consider using STOCK_TORCH_COMPILE or DYNAMO_TRACE_ONCE
for UNBACKED support.
```

#### 决策 3：编译模式与 cudagraph_mode 的映射关系

四个编译模式的 cudagraph_mode 默认值不同：
- VLLM_COMPILE → PIECEWISE（使用 ACL Graph）
- 其他三种 → NONE（使用 torch.compile 或 eager）

这个映射关系不能一刀切处理。

---

## 五、正确性验证

### 5.1 单元测试（35+ 个）

测试文件：`tests/ut/test_platform.py`

| 测试类 | 测试方法 | 验证内容 |
|--------|---------|---------|
| TestCompilationConfig | test_all_compilation_modes | 四种 mode 正确设置 |
| TestCompilationConfig | test_dynamic_shapes_for_stock_modes | pass-through vs fallback 逻辑 |
| TestCompilationConfig | test_enforce_eager_mode | enforce_eager 强制 NONE |
| TestCompilationConfig | test_vllm_compile_unbacked_fallback | UNBACKED → BACKED 回退 |
| TestCompilationConfig | test_vllm_compile_eval_guards_fallback | evaluate_guards → False |
| TestCompilationConfig | test_compile_backend_is_npu | backend 返回 "npu" |
| TestPlatform | test_is_uva_available | NPU 不支持 UVA |
| TestPlatform | test_manual_seed_all | NPU manual_seed_all |
| ... | ... | ... |

### 5.2 NPU E2E 测试（15/15 通过）

使用 `run_compilation_tests.sh` 批量测试，所有 15 个测试用例通过：

| 测试 | 结果 | 关键日志验证 |
|------|------|------------|
| T1 (NONE) | ✅ | `Cudagraph mode ... is not compatible ... Overriding to NONE` |
| T2 (VLLM_COMPILE) | ✅ | `PIECEWISE compilation enabled on NPU` + `Replaying aclgraph` |
| T3 (STOCK_TORCH_COMPILE) | ✅ | `STOCK_TORCH_COMPILE compilation mode enabled` |
| T4 (DYNAMO_TRACE_ONCE) | ✅ | `DYNAMO_TRACE_ONCE compilation mode enabled` |
| T11 (VLLM_COMPILE+UNBACKED) | ✅ | `WARNING ... UNBACKED ... Falling back to BACKED` |
| T12 (VLLM_COMPILE+eval_guards) | ✅ | `WARNING ... evaluate_guards=True ... Setting to False` |
| T13 (DYNAMO+BSO) | ✅ | 正常透传 |
| T14 (STOCK+UNBACKED) | ✅ | 正常透传 |
| T15 (INVALID) | ✅ | `ValidationError: Invalid compilation mode: INVALID` |

### 5.3 CUDA 对照测试

相同测试用例在 CUDA A800 上运行作为对照，验证 Ascend 行为与 CUDA 一致（除 ACL Graph 特有的 UNBACKED/eval_guards 回退）。

---

## 六、Bug 修复记录

| 编号 | 严重度 | 问题 | 修复方式 |
|------|--------|------|---------|
| 1 | P0 | `STOCK_TORCH_COMPILE` 和 `DYNAMO_TRACE_ONCE` 被静默降级，用户配置不生效 | 改为透传给 vLLM 上层，backend 设为 "npu" |
| 2 | P0 | `dynamic_shapes_config` 完全未处理，用户配置被忽略 | 添加读取和验证逻辑 |
| 3 | P1 | `get_compile_backend()` 返回 `None`，Inductor 路径不可用 | 改为返回 `"npu"` |
| 4 | P1 | `NPUPlatform` 缺少 `manual_seed_all` 方法 | 添加 `torch.npu.manual_seed_all(seed)` |

---

## 七、已对齐原生 vLLM 的功能清单

| 功能 | 状态 | 文件 |
|------|------|------|
| mode=NONE 正常工作 | ✅ | `platform.py` |
| mode=VLLM_COMPILE (ACL Graph) | ✅ | `platform.py` |
| mode=STOCK_TORCH_COMPILE | ✅ (不再静默降级) | `platform.py` |
| mode=DYNAMO_TRACE_ONCE | ✅ (不再静默降级) | `platform.py` |
| dynamic_shapes_config 读取 | ✅ | `platform.py` |
| UNBACKED → BACKED 回退 (VLLM_COMPILE) | ✅ | `platform.py` |
| evaluate_guards 回退 (VLLM_COMPILE) | ✅ | `platform.py` |
| 其他模式 pass-through | ✅ | `platform.py` |
| get_compile_backend → "npu" | ✅ | `utils.py` |
| simple_compile_backend → "npu" | ✅ | `utils.py` |
| manual_seed_all | ✅ | `platform.py` |
| 单元测试覆盖 | ✅ | `tests/ut/test_platform.py` |

---

## 八、评测方法

### 8.1 NPU 端

```bash
cd /root/work/vllm-ascend
git checkout feature/pr-compilation-config-v2

export VLLM_VERSION=0.9.0
./run_compilation_tests.sh
# 日志: ./test_logs/compilation_<时间戳>/
```

### 8.2 CUDA 端

```bash
# 拷贝 run_compilation_tests_cuda.sh 到 CUDA 机子
./run_compilation_tests_cuda.sh
```

### 8.3 单元测试

```bash
pytest tests/ut/test_platform.py -v
# Expected: 35+ passed
```

---

## 九、修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `vllm_ascend/platform.py` | 修改 | check_and_update_config（+200行） |
| `vllm_ascend/compilation/compiler_interface.py` | 修改 | AscendCompiler 路径修正 |
| `vllm_ascend/utils.py` | 修改 | get_compile_backend → "npu" |
| `tests/ut/test_platform.py` | 修改 | 35+ 单元测试 |

---

## 十、结论

### 功能结论

- 四种 CompilationMode **全部支持**：NONE、VLLM_COMPILE、STOCK_TORCH_COMPILE、DYNAMO_TRACE_ONCE
- `dynamic_shapes_config` 不再静默忽略：UNBACKED 在 ACL Graph 模式下智能回退，其他模式正常透传
- 所有 15 个 E2E 测试 + 35+ 个单元测试全部通过
- 与 CUDA 平台行为一致（除 ACL Graph 特有的兼容性回退）

### 工程总结

核心适配是**将 Ascend 平台从"限制用户选择"转变为"尊重并引导用户选择"**：
1. `STOCK_TORCH_COMPILE`/`DYNAMO_TRACE_ONCE` 不再降级，正常使用 torch_npu 的 Inductor 后端
2. ACL Graph 与 `UNBACKED` 冲突时有明确的 warning 和替代方案
3. 其他模式通过 pass-through，不改动用户配置，由 vLLM 上层统一处理
