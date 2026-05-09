# compilation-config Ascend NPU 实现技术报告

## 一、项目概述

为 Ascend NPU 适配 `--compilation-config` 的两个子参数：

- **`mode`**：CompilationMode 枚举（NONE / VLLM_COMPILE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE）
- **`dynamic_shapes_config`**：DynamicShapesConfig（type / evaluate_guards / assume_32_bit_indexing）

消除"静默忽略用户配置"的问题，每个用户设置都能正确生效或给出明确提示。

**背景**：适配前 Ascend 仅支持 NONE 和 VLLM_COMPILE，STOCK_TORCH_COMPILE 和 DYNAMO_TRACE_ONCE 被静默降级为 eager，dynamic_shapes_config 完全未处理。

## 二、测试模型

| 角色 | 模型 |
|------|------|
| NPU | Qwen/Qwen2.5-0.5B-Instruct |
| CUDA 对照 | 同上 |

## 三、配置矩阵

15 个测试场景覆盖所有组合：

**mode 参数（4个）**：NONE（纯eager）/ VLLM_COMPILE（ACL Graph）/ STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE

**dynamic_shapes_config（6个）**：BACKED / UNBACKED(VLLM_COMPILE回退) / BACKED_SIZE_OBLIVIOUS / evaluate_guards=True / assume_32_bit_indexing

**组合测试（4个）**：VLLM_COMPILE+UNBACKED / VLLM_COMPILE+eval_guards / DYNAMO+BSO / STOCK+UNBACKED

**错误处理（1个）**：mode=INVALID → ValidationError

## 四、实现架构

### 4.1 文件结构

```
vllm_ascend/
├── platform.py                        # check_and_update_config（+200行）
├── compilation/compiler_interface.py  # AscendCompiler 修正
├── utils.py                           # get_compile_backend → "npu"
tests/ut/test_platform.py              # 35+ 单元测试
```

### 4.2 四种模式在 NPU 上的机制

| mode | 底层机制 | cudagraph_mode | backend |
|------|---------|---------------|---------|
| NONE | 纯 eager | NONE | — |
| VLLM_COMPILE | AscendCompiler → TorchAIR ACL Graph | PIECEWISE | eager(EagerAdaptor) |
| STOCK_TORCH_COMPILE | torch.compile(backend="npu") | NONE | npu(Inductor) |
| DYNAMO_TRACE_ONCE | 单次Dynamo trace, backend="npu" | NONE | npu(Inductor) |

### 4.3 核心改动

**改动 1：支持全部四种 CompilationMode**

之前 STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE 被静默降级。现在透传给 vLLM 上层，backend 设为 "npu"。

**改动 2：dynamic_shapes_config 智能处理**

- VLLM_COMPILE 模式：UNBACKED→BACKED 回退(warning)，evaluate_guards=True→False(warning)
- 其他模式：直接透传，由 vLLM 上层 `TorchCompileWithNoGuardsWrapper` 处理
- 与 CUDA 平台行为一致

**改动 3：get_compile_backend → "npu"**

torch_npu 提供基于 Inductor 的 NPU 编译后端。VLLM_COMPILE 模式 backend 覆写为 "eager" 使 make_compiler() 使用 EagerAdaptor。

### 4.4 关键设计决策

- 平台层只做兼容性检查，不做运行时实现：dynamic_shapes_config 运行时由 vLLM 上层 wrapper.py 处理
- UNBACKED 回退而非报错：给 warning + 替代方案建议，比直接报错更友好
- 透传策略：CUDA 平台也不处理 dynamic_shapes_config，Ascend 保持一致

## 五、正确性验证

### 5.1 单元测试（35+ 个，全部通过）

测试文件：`tests/ut/test_platform.py`

| 测试 | 验证内容 |
|------|---------|
| test_all_compilation_modes | 四种 mode 正确设置 |
| test_dynamic_shapes_for_stock_modes | pass-through vs fallback |
| test_enforce_eager_mode | enforce_eager 强制 NONE |
| test_vllm_compile_unbacked_fallback | UNBACKED → BACKED |
| test_vllm_compile_eval_guards_fallback | evaluate_guards → False |
| test_compile_backend_is_npu | backend 返回 "npu" |

### 5.2 E2E 测试（15/15 通过）

| 测试 | 关键日志验证 |
|------|------------|
| T1: NONE | `Overriding to NONE` |
| T2: VLLM_COMPILE | `PIECEWISE compilation enabled` + `Replaying aclgraph` |
| T3: STOCK_TORCH_COMPILE | `STOCK_TORCH_COMPILE compilation mode enabled` |
| T4: DYNAMO_TRACE_ONCE | `DYNAMO_TRACE_ONCE compilation mode enabled` |
| T11: VLLM+UNBACKED | `WARNING ... UNBACKED ... Falling back to BACKED` |
| T12: VLLM+eval_guards | `WARNING ... evaluate_guards=True ... Setting to False` |
| T13: DYNAMO+BSO | 正常透传 |
| T14: STOCK+UNBACKED | 正常透传 |
| T15: mode=INVALID | `ValidationError: Invalid compilation mode` |

## 六、Bug 修复记录

| 编号 | 严重度 | 问题 | 修复 |
|------|--------|------|------|
| 1 | P0 | STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE 被静默降级 | 透传给 vLLM 上层，backend="npu" |
| 2 | P0 | dynamic_shapes_config 完全未处理 | 添加读取和验证逻辑 |
| 3 | P1 | get_compile_backend() 返回 None | 改为 "npu" |
| 4 | P1 | NPUPlatform 缺少 manual_seed_all | 添加 torch.npu.manual_seed_all(seed) |

## 七、已对齐原生 vLLM 的功能清单

| 功能 | 文件 |
|------|------|
| mode=NONE 正常 | `platform.py` |
| mode=VLLM_COMPILE (ACL Graph) | `platform.py` |
| mode=STOCK_TORCH_COMPILE | `platform.py` |
| mode=DYNAMO_TRACE_ONCE | `platform.py` |
| dynamic_shapes_config 读取和验证 | `platform.py` |
| UNBACKED→BACKED 回退 (VLLM_COMPILE) | `platform.py` |
| evaluate_guards 回退 (VLLM_COMPILE) | `platform.py` |
| get_compile_backend→"npu" | `utils.py` |
| simple_compile_backend→"npu" | `utils.py` |
| manual_seed_all | `platform.py` |

## 八、评测方法

### NPU

```bash
cd /root/work/vllm-ascend && git checkout feature/pr-compilation-config-v2
export VLLM_VERSION=0.9.0
./run_compilation_tests.sh
```

### 单元测试

```bash
pytest tests/ut/test_platform.py -v
# Expected: 35+ passed
```

## 九、修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `vllm_ascend/platform.py` | 修改 | check_and_update_config（+200行） |
| `vllm_ascend/compilation/compiler_interface.py` | 修改 | AscendCompiler 路径修正 |
| `vllm_ascend/utils.py` | 修改 | get_compile_backend → "npu" |
| `tests/ut/test_platform.py` | 修改 | 35+ 单元测试 |

## 十、结论

- 四种 CompilationMode 全部支持，不再静默降级
- dynamic_shapes_config 不再忽略，ACL Graph 模式智能回退，其他模式正常透传
- 15 个 E2E + 35+ UT 全部通过
- 与 CUDA 平台行为一致
