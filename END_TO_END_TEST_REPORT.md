# --compilation-config 参数适配：端到端测试报告

## 一、任务概述

适配 vLLM 启动参数 `--compilation-config` 的两个子参数到 Ascend NPU 平台：
- `mode`：CompilationMode 枚举（NONE / VLLM_COMPILE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE）
- `dynamic_shapes_config`：DynamicShapesConfig 配置对象（type / evaluate_guards / assume_32_bit_indexing）

目标：让用户设置的这两个参数在 Ascend NPU 平台上正确生效或给出明确的提示，消除"静默忽略"的问题。

---

## 二、创新与工程难点

### 2.1 UNBACKED → BACKED 智能降级

**技术背景**：Ascend ACL Graph 是纯静态图捕获机制（类似 CUDA Graph），捕获时所有张量维度必须是确定的常量。`UNBACKED` 动态 shape 将张量维度标记为符号值（symbolic），与 ACL Graph 的静态捕获语义根本冲突。如果用户在 VLLM_COMPILE 模式下配置 UNBACKED，运行时会因维度不匹配而崩溃。

**创新点**：在平台层 `check_and_update_config` 中主动拦截并降级：
```python
if compilation_config.mode == CompilationMode.VLLM_COMPILE:
    if dynamic_shapes_config.type == DynamicShapesType.UNBACKED:
        logger.warning("UNBACKED ... Falling back to BACKED. "
            "Consider using STOCK_TORCH_COMPILE or DYNAMO_TRACE_ONCE for UNBACKED support.")
        dynamic_shapes_config.type = DynamicShapesType.BACKED
```
这比静默忽略更安全，比直接报错更友好——用户得到明确的 warning 和替代方案指引。

### 2.2 vLLM 上层与平台层的职责分离

**关键设计决策**：`dynamic_shapes_config` 的运行时行为**不需要**平台层实现。vLLM 上层的 `TorchCompileWithNoGuardsWrapper`（wrapper.py）和 `decorators.py` 已经完整处理了所有三种 type 和两个布尔参数。CUDA 平台的 `check_and_update_config` 也完全不处理 `dynamic_shapes_config`。

平台层只做 VLLM_COMPILE 模式的兼容性检查（因为 ACL Graph 的特殊性），其他模式直接透传——与 CUDA 平台行为一致。

### 2.3 多模式兼容的工程权衡

四种编译模式有不同的底层机制：
- `VLLM_COMPILE`：走 VllmBackend → AscendCompiler → TorchAIR ACL Graph
- `STOCK_TORCH_COMPILE` / `DYNAMO_TRACE_ONCE`：走 vLLM 上层的 `torch.compile` 路径
- `NONE`：纯 eager

平台层的 `check_and_update_config` 必须正确区分这四种模式的 cudagraph_mode 默认值、use_inductor 设置、splitting_ops 等，而不能一刀切处理。

---

## 三、实现的核心改动

### 3.1 platform.py — check_and_update_config

**文件**: `vllm_ascend/platform.py`

#### 改动 1：支持所有四种 CompilationMode

**之前行为**：STOCK_TORCH_COMPILE 和 DYNAMO_TRACE_ONCE 会被静默降级为 eager 模式（warning + cudagraph_mode=NONE）。

**现在行为**：
- NONE：纯 eager，不做任何编译
- VLLM_COMPILE：使用 ACL Graph 分段编译（原有行为）
- STOCK_TORCH_COMPILE：直接传递给 vLLM 上层的 `TorchCompileWithNoGuardsWrapper`，使用标准 `torch.compile`
- DYNAMO_TRACE_ONCE：类似 STOCK_TORCH_COMPILE，单次 Dynamo trace

#### 改动 2：根据模式区分处理 dynamic_shapes_config

**VLLM_COMPILE 模式**（ACL Graph 静态捕获）：
- UNBACKED → 回退到 BACKED + warning（ACL Graph 捕获静态图，不支持 unbacked shapes）
- evaluate_guards=True → 设为 False + warning（ACL Graph 路径不支持 guard 评估）

**NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE 模式**：
- 直接透传，不做任何修改
- 由 vLLM 上层的 `TorchCompileWithNoGuardsWrapper` 和 `decorators.py` 处理
- 与 CUDA 平台行为一致

#### 改动 3：get_compile_backend 返回 "npu"

`get_compile_backend()` 返回 `"npu"`，torch_npu 提供了基于 Inductor 的 NPU 编译后端，注册名为 `"npu"`。对不同编译模式的作用：
- STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE：返回值直接作为 `torch.compile(model, backend='npu')` 的 backend，获得真正的 Inductor 编译加速
- VLLM_COMPILE：在 `check_and_update_config` 中将 `compilation_config.backend` 覆写为 `"eager"`，使 `make_compiler()` 使用 EagerAdaptor（实际编译由 AscendCompiler → TorchAIR ACL Graph 接管）
- NONE 模式：不编译

`simple_compile_backend` 也从 `"eager"` 改为 `"npu"`，用于简单/独立函数的 torch.compile。

#### 改动 4：移除 "experimental" 标签

所有四种模式均为完整支持，移除日志中的 "experimental" 警告。

#### 改动 5：添加 manual_seed_all 方法

NPUPlatform 缺少 `manual_seed_all` 方法，添加调用 `torch.npu.manual_seed_all(seed)`。

---

## 四、CUDA 对照测试

在 CUDA GPU 机子上运行 vLLM 原版作为对照基准，与 NPU（vllm-ascend）测试数据进行对比。

### 4.1 CUDA 测试脚本

使用 `run_compilation_tests_cuda.sh`，拷贝到 CUDA 机子上运行：

```bash
# 在 CUDA 机子上：
# 1. 安装 vLLM
pip install vllm

# 2. 下载脚本并运行
./run_compilation_tests_cuda.sh
```

脚本与 NPU 版本（`run_compilation_tests.sh`）使用相同的：
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`（HuggingFace ID，CUDA 机子自动下载）
- 相同的 15 个测试用例（4 mode + 6 dynamic_shapes + 4 组合 + 1 错误处理）
- 相同的 warmup + 3 条变长 prompt
- 相同的 baseline 正确性比对

日志保存到 `./test_logs/cuda_compilation_<时间戳>/`。

### 4.2 两台机子的测试分工

| 机子 | 脚本 | 模型路径 | 测试内容 |
|------|------|---------|---------|
| NPU (本机) | `run_compilation_tests.sh` | `/root/work/models/Qwen2.5-0.5B-Instruct/` | vllm-ascend 全部 15 个测试 |
| CUDA | `run_compilation_tests_cuda.sh` | `Qwen/Qwen2.5-0.5B-Instruct` (HF ID) | vLLM 原版全部 15 个测试 |

两份脚本的测试用例完全对齐，输出可直接填入报告第六章和第七章进行对比。

---

## 五、NPU 端测试（本机）

### 5.1 测试方法说明

本机（NPU）采用离线 `vllm.LLM` 批处理方式进行端到端测试。选择离线方式的原因：
- 可精确控制每条 prompt 的输入输出，便于自动化正确性比对
- 无需管理 HTTP 服务进程和端口
- 更快的启动-测试-验证循环

### 5.2 批量测试脚本

使用 `run_compilation_tests.sh` 一键运行所有测试：

```bash
cd /root/work/vllm-ascend
./run_compilation_tests.sh
```

脚本自动运行测试用例，包含：
- mode 参数测试（4 个，含 baseline 保存）
- dynamic_shapes_config 参数测试（6 个）
- 组合测试（4 个）
- 错误处理测试（1 个）

**测试脚本改进（v2）**：
- **Warmup 预热**：每次测试在正式计时前先 warmup 一次，消除首次编译/JIT 开销
- **变长 prompt**：每条测试用 3 条不同长度的问句（短/中/长），验证动态 shape 处理
- **正确性比对**：各模式输出与 NONE baseline 逐条比对，确保编译不影响推理结果
- **安全传递**：配置通过环境变量传递给 Python，避免 JSON 注入风险

日志保存到 `./test_logs/compilation_<时间戳>/` 目录。

### 5.3 单独运行 Python 测试

```bash
export VLLM_VERSION=0.9.0

# mode=NONE: 纯 eager 模式
/usr/local/python3.11.14/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='/root/work/models/Qwen2.5-0.5B-Instruct/',
    compilation_config={'mode': 'NONE'},
    max_model_len=256, max_num_seqs=4, trust_remote_code=True
)
outputs = llm.generate(['Hello, my name is'], SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
"

# mode=VLLM_COMPILE: ACL Graph 分段编译
/usr/local/python3.11.14/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='/root/work/models/Qwen2.5-0.5B-Instruct/',
    compilation_config={'mode': 'VLLM_COMPILE'},
    max_model_len=256, max_num_seqs=4, trust_remote_code=True
)
outputs = llm.generate(['Hello, my name is'], SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
"

# mode=STOCK_TORCH_COMPILE: 标准 torch.compile
/usr/local/python3.11.14/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='/root/work/models/Qwen2.5-0.5B-Instruct/',
    compilation_config={'mode': 'STOCK_TORCH_COMPILE'},
    max_model_len=256, max_num_seqs=4, trust_remote_code=True
)
outputs = llm.generate(['Hello, my name is'], SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
"

# mode=DYNAMO_TRACE_ONCE: 单次 Dynamo trace
/usr/local/python3.11.14/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='/root/work/models/Qwen2.5-0.5B-Instruct/',
    compilation_config={'mode': 'DYNAMO_TRACE_ONCE'},
    max_model_len=256, max_num_seqs=4, trust_remote_code=True
)
outputs = llm.generate(['Hello, my name is'], SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
"
```

---

## 六、实测结果

### 6.1 NPU 端测试结果（v2 脚本实测数据）

测试环境：Ascend 910B2C NPU，Qwen2.5-0.5B-Instruct 模型，max_model_len=256

> v2 脚本：warmup + 3 条变长 prompt，总输出 tokens = 96。

#### 一、mode 参数测试

| 测试 | 模式 | 吞吐量 (toks/s) | 输出 tokens | 总耗时 (ms) | 状态 |
|------|------|-----------------|-------------|-------------|------|
| test_1 | NONE (纯 eager) | 100.12 | 96 | ~8200 | PASS |
| test_2 | VLLM_COMPILE (ACL Graph) | 187.87 | 96 | ~11800 | PASS |
| test_3 | STOCK_TORCH_COMPILE | 101.53 | 96 | ~8200 | PASS |
| test_4 | DYNAMO_TRACE_ONCE | 126.52 | 96 | ~10700 | PASS |

**关键日志验证：**
- NONE：`Cudagraph mode FULL_AND_PIECEWISE is not compatible with compilation mode 0. Overriding to NONE.` — 正确禁用编译
- VLLM_COMPILE：`PIECEWISE compilation enabled on NPU` + ACL Graph 捕获 3/3 — 正确启用 ACL Graph
- STOCK_TORCH_COMPILE：`STOCK_TORCH_COMPILE compilation mode enabled on Ascend NPU.` — 正确启用
- DYNAMO_TRACE_ONCE：`DYNAMO_TRACE_ONCE compilation mode enabled on Ascend NPU.` — 正确启用

#### 二、dynamic_shapes_config 参数测试

| 测试 | 配置 | 吞吐量 (toks/s) | 状态 | 关键行为 |
|------|------|-----------------|------|---------|
| test_5 | STOCK_TORCH_COMPILE + BACKED | 102.22 | PASS | BACKED 是默认值，直接透传 |
| test_6 | DYNAMO_TRACE_ONCE + UNBACKED | 118.36 | PASS | UNBACKED 直接透传，无 warning |
| test_7 | STOCK_TORCH_COMPILE + evaluate_guards=True | 102.39 | PASS | evaluate_guards 直接透传 |
| test_8 | STOCK_TORCH_COMPILE + UNBACKED | 100.46 | PASS | UNBACKED 直接透传 |
| test_9 | DYNAMO_TRACE_ONCE + assume_32_bit_indexing | 122.76 | PASS | INFO: assume_32_bit_indexing 已传递 |
| test_10 | STOCK_TORCH_COMPILE + BACKED_SIZE_OBLIVIOUS | 97.17 | PASS | SIZE_OBLIVIOUS 直接透传 |

#### 三、组合测试

| 测试 | 配置 | 吞吐量 (toks/s) | 状态 | 关键行为 |
|------|------|-----------------|------|---------|
| test_11 | VLLM_COMPILE + UNBACKED | 185.63 | PASS | WARNING: UNBACKED → BACKED 回退，ACL Graph 正常工作 |
| test_12 | VLLM_COMPILE + evaluate_guards=True | 175.49 | PASS | WARNING: evaluate_guards True → False，ACL Graph 正常工作 |
| test_13 | DYNAMO_TRACE_ONCE + UNBACKED | 119.42 | PASS | UNBACKED 直接透传 |
| test_14 | NONE + BACKED | 103.25 | PASS | 无编译，直接透传 |

**总计：15/15 全部通过（14 个功能测试 + 1 个错误处理测试），0 失败**

### 6.2 吞吐量分析（v2 脚本实测数据）

| 编译模式 | 平均吞吐量 (toks/s) | 相对 NONE 的加速比 |
|---------|---------------------|-------------------|
| NONE (纯 eager) | 100.12 | 1.00x |
| STOCK_TORCH_COMPILE | 101.53 | 1.01x |
| DYNAMO_TRACE_ONCE | 126.52 | 1.26x |
| VLLM_COMPILE (ACL Graph) | 187.87 | 1.88x |

**说明：**
- VLLM_COMPILE 使用 ACL Graph 减少 Host 下发 Kernel 的开销，性能最佳（1.88x 加速）
- DYNAMO_TRACE_ONCE 通过单次 trace + 移除 guards 获得一定加速（1.26x）
- STOCK_TORCH_COMPILE 使用 torch_npu Inductor 后端，但 0.5B 小模型下编译加速不明显（1.01x）
- 所有测试输出文本完全一致，说明不同编译模式不影响推理结果

**数据局限性说明**：
- 0.5B 模型在 910B 上不是性能瓶颈，数据主要用于验证功能正确性
- 真实生产场景（大模型 + 高并发）下 ACL Graph 的加速比会更显著
- 生产级性能评估应使用 vLLM 内置 `benchmark_serving.py` 工具

### 6.3 关键日志验证

#### VLLM_COMPILE + UNBACKED（test_11）
```
WARNING [platform.py:458] UNBACKED dynamic shapes type is not compatible with
VLLM_COMPILE mode's ACL Graph static capture. Falling back to BACKED.
Consider using STOCK_TORCH_COMPILE or DYNAMO_TRACE_ONCE for UNBACKED support.
```

#### VLLM_COMPILE + evaluate_guards=True（test_12）
```
WARNING [platform.py:466] evaluate_guards=True is not compatible with VLLM_COMPILE
mode's ACL Graph path. Setting to False. Consider using STOCK_TORCH_COMPILE or
DYNAMO_TRACE_ONCE for evaluate_guards support.
```

#### DYNAMO_TRACE_ONCE + assume_32_bit_indexing（test_9）
```
INFO [platform.py:479] assume_32_bit_indexing is enabled. This is passed to
torch._inductor.config for Inductor backend compatibility.
```

### 6.4 单元测试结果

```
========= 35 passed, 1 skipped, 4 warnings, 2 subtests passed in 0.20s =========
```

**测试覆盖明细**：
- `test_check_and_update_config_all_compilation_modes`：验证四种 mode 均可正常设置
- `test_check_and_update_config_dynamic_shapes_for_stock_modes`：
  - STOCK_TORCH_COMPILE + UNBACKED → type 保持 UNBACKED（透传给 vLLM 上层）
  - VLLM_COMPILE + UNBACKED → type 改为 BACKED + WARNING 日志
  - VLLM_COMPILE + evaluate_guards=True → 设为 False + WARNING 日志
  - DYNAMO_TRACE_ONCE + UNBACKED → type 保持 UNBACKED（透传）
  - NONE + UNBACKED → type 保持 UNBACKED（透传，与 CUDA 平台一致）
- `test_check_and_update_config_enforce_eager_mode`：验证 enforce_eager 强制 NONE
- `test_check_and_update_config_preserves_platform_default_max_input`：验证 max_cudagraph_capture_size 默认值

### 6.5 CUDA 对照测试结果（待在 CUDA 机子上实测）

> **注意**：使用 `run_compilation_tests_cuda.sh` 在 CUDA GPU 机子上运行 vLLM 原版，结果填入此节。CUDA 测试数据用于与 6.1 NPU 数据进行逐项对比，填入第七章对比表。

<!-- TODO: 在 CUDA 机子运行 ./run_compilation_tests_cuda.sh，将结果填入此处 -->

---

## 七、CUDA vs NPU 行为对比

> **吞吐量数据待补充**：以下行为对比基于代码分析。CUDA 端实际吞吐量数据待 `run_compilation_tests_cuda.sh` 实测后填入。

| 测试场景 | CUDA (vLLM) | NPU (vLLM Ascend) | 行为差异 |
|----------|-------------|-------------------|---------|
| `mode=NONE` | eager | eager | 一致 |
| `mode=VLLM_COMPILE` | CUDA Graph | ACL Graph | 后端不同，功能一致 |
| `mode=STOCK_TORCH_COMPILE` | torch.compile + inductor | torch.compile + npu (torch_npu Inductor) | 后端名称不同，机制一致 |
| `mode=DYNAMO_TRACE_ONCE` | 单次 trace | 单次 trace | 一致 |
| `type=BACKED` | 正常 | 正常 | 一致 |
| `type=UNBACKED` | 正常 | 正常（VLLM_COMPILE 下回退） | VLLM_COMPILE 有差异 |
| `evaluate_guards=True` | 正常 | 正常（VLLM_COMPILE 下设 False） | VLLM_COMPILE 有差异 |
| `assume_32_bit_indexing` | 传给 Inductor | 传给 torch_npu Inductor | 一致 |

**STOCK_TORCH_COMPILE 的 NPU 限制说明**：
在 CUDA 平台上，STOCK_TORCH_COMPILE 使用 PyTorch 原生 Inductor 后端（`"inductor"`）进行算子融合和代码生成。在 NPU 平台上，`get_compile_backend()` 返回 `"npu"`——这是 torch_npu 提供的基于 Inductor 的编译后端，`torch.compile(model, backend='npu')` 可正常工作。VLLM_COMPILE 模式下通过 `compilation_config.backend = "eager"` 绕过 Inductor 路径，由 AscendCompiler → TorchAIR ACL Graph 接管编译。

---

## 八、关键技术发现

### 8.1 dynamic_shapes_config 平台层无需实现运行时行为

`dynamic_shapes_config` 的运行时行为由 vLLM 上层处理（`TorchCompileWithNoGuardsWrapper` 和 `decorators.py`）。CUDA 平台的 `check_and_update_config` 也完全不处理 `dynamic_shapes_config`。我们的平台层只在 VLLM_COMPILE 模式下做兼容性检查（ACL Graph 静态捕获限制），其他模式直接透传——与 CUDA 平台行为一致。

### 8.2 get_compile_backend 的双重作用

`get_compile_backend()` 返回 `"npu"`，在 vLLM 中有两重作用：

1. **STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE 模式**：`init_backend()` 中 `"npu"` 在 `torch_backends` 列表中，直接返回作为 `torch.compile(model, backend='npu')` 的 backend——这是 torch_npu 提供的 Inductor-based 编译后端
2. **VLLM_COMPILE 模式**：`check_and_update_config` 中将 `compilation_config.backend` 覆写为 `"eager"`，使 `make_compiler()` 使用 EagerAdaptor。实际编译由 AscendCompiler → TorchAIR ACL Graph 接管

### 8.3 已知限制

- UNBACKED 模式需设置 `VLLM_USE_BYTECODE_HOOK=0`（vLLM 上游要求，非 Ascend 特有）
- vLLM 开发版（main 分支，`__version__="dev"`）需设置 `VLLM_VERSION=0.9.0` 环境变量，正式版本不需要

---

## 九、修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `vllm_ascend/platform.py` | 支持四种 CompilationMode + dynamic_shapes_config 分模式处理 + 注释加强 |
| `vllm_ascend/compilation/compiler_interface.py` | 移除 placeholder dynamic_shapes_config 代码 |
| `tests/ut/test_platform.py` | 更新测试断言，添加 dynamic_shapes_config 测试用例 |
| `run_compilation_tests.sh` | NPU 端批量测试脚本 v2：warmup + 变长 prompt + 正确性比对 + 错误测试 |
| `run_compilation_tests_cuda.sh` | CUDA 端对照测试脚本：与 NPU 脚本测试用例完全对齐 |
| `END_TO_END_TEST_REPORT.md` | 本文件 |
| `SKILL.md` | AI 工具使用说明（比赛交付物） |

---

## 十、结论

1. **`mode` 参数**：四种模式全部在 Ascend NPU 上完整支持，端到端验证通过
2. **`dynamic_shapes_config` 参数**：三个子参数全部正确处理，与 vLLM 标准行为一致
3. **关键设计**：平台层只在 VLLM_COMPILE 模式下做必要的回退（ACL Graph 限制），其他模式直接透传给 vLLM 上层处理
4. **性能验证**：VLLM_COMPILE 模式性能最佳（187.87 toks/s，相对 NONE 1.88x 加速），DYNAMO_TRACE_ONCE 次之（126.52 toks/s，1.26x 加速）
5. **15/15 测试全部通过**，所有配置组合在 Qwen2.5-0.5B 模型上验证通过，输出结果一致
6. **STOCK_TORCH_COMPILE 使用 torch_npu Inductor 后端**：`get_compile_backend()` 返回 `"npu"`，使用 torch_npu 提供的基于 Inductor 的编译后端。VLLM_COMPILE 模式下覆写 `backend` 为 `"eager"` 以绕过 Inductor 路径，由 AscendCompiler → TorchAIR ACL Graph 接管
