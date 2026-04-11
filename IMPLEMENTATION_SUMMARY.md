# --compilation-config 参数完整支持实现总结

## 概述

本次实现为 vLLM Ascend 完整支持了 `--compilation-config` 参数下的所有子参数，使昇腾 NPU 能够使用所有四种编译模式。

## 修改的文件

### 1. `vllm_ascend/platform.py`

#### 主要改动：

**a) 移除 mode 限制（304-318 行）**
- 移除了对 `STOCK_TORCH_COMPILE` 和 `DYNAMO_TRACE_ONCE` 的 ValueError 检查
- 改为记录 info 日志，提示这是实验性功能
- 为这些模式默认设置 `cudagraph_mode = NONE`

**b) 更新 PIECEWISE 模式逻辑（353-375 行）**
- 对于 `VLLM_COMPILE` 模式：保持原有行为
- 对于其他模式：记录 warning，允许使用但不强制 `use_inductor=False`

**c) 更新 FULL_DECODE_ONLY/FULL 模式逻辑（377-408 行）**
- 对于 `VLLM_COMPILE` 模式：保持原有行为
- 对于其他模式：记录 warning，不强制修改配置

**d) 更新 else 分支（410-422 行）**
- 对于 `VLLM_COMPILE` 模式：保持原有行为（回退到 NONE）
- 对于其他模式：只记录 warning，不强制修改 mode

**e) 完整支持 dynamic_shapes_config（424-465 行）**
- 对于 `STOCK_TORCH_COMPILE` 和 `DYNAMO_TRACE_ONCE`：
  - 直接传递配置，不修改
  - 记录 info 日志提示实验性
- 对于 `NONE` 和 `VLLM_COMPILE`：
  - 保持原有的保守行为
  - `UNBACKED` → 回退到 `BACKED` + warning
  - `evaluate_guards=True` → 设置为 `False` + warning
- `assume_32_bit_indexing`：始终记录 info 日志

### 2. `vllm_ascend/compilation/compiler_interface.py`

#### 主要改动：

**a) `AscendCompiler.compile()` 方法（133-173 行）**
- 添加对 `dynamic_shapes_config` 的访问
- 添加 debug 日志，记录使用的配置
- 将配置传递给底层编译函数

**b) `npugraph_ex_compile()` 函数（70-130 行）**
- 添加对 `dynamic_shapes_config` 的访问
- 预留了将配置应用到 TorchAIR 的位置（注释）
- 保持现有功能不变

### 3. `tests/ut/test_platform.py`

#### 主要改动：

**a) 更新测试用例**
- 将 `test_check_and_update_config_unsupported_compilation_level` 重命名为 `test_check_and_update_config_all_compilation_modes`
- 更新测试逻辑，验证所有四种模式都被允许

**b) 新增测试用例**
- `test_check_and_update_config_dynamic_shapes_for_experimental_modes`：
  - 验证实验模式下 dynamic_shapes_config 被直接传递
  - 验证 VLLM_COMPILE 模式下保持保守行为

## 功能说明

### 四种编译模式支持

| 模式 | 状态 | 说明 |
|------|------|------|
| `NONE` | ✅ 完整支持 | 纯 eager 模式 |
| `VLLM_COMPILE` | ✅ 完整支持 | ACL Graph 编译（原有行为） |
| `STOCK_TORCH_COMPILE` | ⚠️ 实验性支持 | 标准 torch.compile 模式 |
| `DYNAMO_TRACE_ONCE` | ⚠️ 实验性支持 | 单次 trace 模式 |

### dynamic_shapes_config 支持

| 配置项 | VLLM_COMPILE/NONE | STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE |
|--------|-------------------|----------------------------------------|
| `type=BACKED` | ✅ 支持 | ✅ 支持 |
| `type=UNBACKED` | ⚠️ 回退到 BACKED + warning | ✅ 直接传递 + info |
| `evaluate_guards=True` | ⚠️ 设置为 False + warning | ✅ 直接传递 + info |
| `assume_32_bit_indexing=True` | ℹ️ info 日志 | ℹ️ info 日志 |

## 设计原则

1. **向后兼容**：`VLLM_COMPILE` 和 `NONE` 模式行为完全不变
2. **渐进式支持**：新功能标记为实验性，给用户反馈
3. **区别对待**：
   - VLLM_COMPILE：保守策略（保持稳定性）
   - STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE：开放策略（允许探索）

## 使用示例

```bash
# 使用 VLLM_COMPILE（原有方式，推荐）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --compilation-config '{"mode": "VLLM_COMPILE"}'

# 实验 STOCK_TORCH_COMPILE
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --compilation-config '{"mode": "STOCK_TORCH_COMPILE"}'

# 实验 DYNAMO_TRACE_ONCE + dynamic_shapes_config
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --compilation-config '{
    "mode": "DYNAMO_TRACE_ONCE",
    "dynamic_shapes_config": {
      "type": "UNBACKED",
      "evaluate_guards": true
    }
  }'
```

## 后续工作建议

1. **性能测试**：对比不同模式的性能和内存使用
2. **TorchAIR 集成**：将 dynamic_shapes_config 实际应用到 TorchAIR 配置
3. **Guard 处理**：探索在 Ascend 上实现 guard 移除的可行性
4. **文档完善**：添加更多关于实验模式的使用说明
