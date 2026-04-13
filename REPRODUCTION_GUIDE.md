# 复现报告：--compilation-config 参数适配

## 一、环境准备

### NPU 机子（本机）

- 硬件：Ascend 910B2C
- 软件：CANN 8.5.1, PyTorch 2.9.0, torch-npu 2.9.0, vLLM (dev)
- 模型路径：`/root/work/models/Qwen2.5-0.5B-Instruct/`

```bash
cd /root/work/vllm-ascend
export VLLM_VERSION=0.9.0  # dev 版本 workaround，正式版不需要
```

### CUDA 机子（对照基准）

- 硬件：NVIDIA GPU
- 软件：vLLM（pip install vllm）
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`（HuggingFace，自动下载）

## 二、复现步骤

### 步骤 1：运行单元测试（NPU 机子）

```bash
cd /root/work/vllm-ascend
export VLLM_VERSION=0.9.0
pytest tests/ut/test_platform.py -v
```

预期结果：35 passed, 1 skipped, ~0.20s

关键测试用例：
- `test_check_and_update_config_all_compilation_modes`：四种 CompilationMode 均可正常设置
- `test_check_and_update_config_dynamic_shapes_for_stock_modes`：
  - STOCK_TORCH_COMPILE + UNBACKED → type 保持 UNBACKED（透传）
  - VLLM_COMPILE + UNBACKED → type 改为 BACKED + WARNING 日志
  - VLLM_COMPILE + evaluate_guards=True → 设为 False + WARNING 日志
  - DYNAMO_TRACE_ONCE + UNBACKED → type 保持 UNBACKED（透传）
  - NONE + UNBACKED → type 保持 UNBACKED（透传，与 CUDA 平台一致）

### 步骤 2：运行端到端批量测试（NPU 机子）

```bash
cd /root/work/vllm-ascend
export VLLM_VERSION=0.9.0
./run_compilation_tests.sh
```

脚本自动运行 15 个测试用例（4 mode + 6 dynamic_shapes + 4 组合 + 1 错误处理），每个测试含 warmup + 3 条变长 prompt，日志保存到 `./test_logs/compilation_<时间戳>/`。

预期结果：15/15 全部通过

### 步骤 3：运行 CUDA 对照测试（CUDA 机子）

将 `run_compilation_tests_cuda.sh` 拷贝到 CUDA 机子上运行：

```bash
./run_compilation_tests_cuda.sh
```

相同 15 个测试用例，相同 warmup + 3 条 prompt，日志保存到 `./test_logs/cuda_compilation_<时间戳>/`。

### 步骤 4：对比结果

将两台机子的测试数据填入 `END_TO_END_TEST_REPORT.md` 第七章对比表。

## 三、关键验证点

### 3.1 四种 CompilationMode 行为验证

| 模式 | NPU 行为 | 验证方法 |
|------|---------|---------|
| NONE | 纯 eager，不编译 | 日志含 `cudagraph_mode NONE` |
| VLLM_COMPILE | ACL Graph 分段编译 | 日志含 `PIECEWISE compilation enabled on NPU` + `Replaying aclgraph` |
| STOCK_TORCH_COMPILE | torch.compile + npu 后端(torch_npu Inductor) | 日志含 `STOCK_TORCH_COMPILE compilation mode enabled` |
| DYNAMO_TRACE_ONCE | 单次 Dynamo trace | 日志含 `DYNAMO_TRACE_ONCE compilation mode enabled` |

### 3.2 dynamic_shapes_config 行为验证

| 配置 | VLLM_COMPILE 模式 | 其他模式 |
|------|-------------------|---------|
| UNBACKED | 自动回退到 BACKED + WARNING | 直接透传，无修改 |
| evaluate_guards=True | 自动设为 False + WARNING | 直接透传，无修改 |
| assume_32_bit_indexing | INFO 日志（NPU torch_npu Inductor 后端生效） | INFO 日志 |

### 3.3 输出正确性验证

所有 15 个测试的输出文本应与 NONE baseline 完全一致，说明不同编译模式不影响推理结果。

## 四、已知限制

1. **STOCK_TORCH_COMPILE 使用 torch_npu Inductor 后端**：`get_compile_backend()` 返回 `"npu"`，使用 torch_npu 提供的基于 Inductor 的编译后端。VLLM_COMPILE 模式下覆写 backend 为 `"eager"` 以绕过 Inductor 路径，由 AscendCompiler → TorchAIR ACL Graph 接管。
2. **VLLM_COMPILE 不支持 UNBACKED**：ACL Graph 是纯静态图捕获，与 UNBACKED 的符号维度语义冲突。自动降级到 BACKED。
3. **VLLM_VERSION workaround**：vLLM 开发版（main 分支）的 `__version__` 返回 "dev"，需要设置 `VLLM_VERSION=0.9.0`。正式发布版本不需要。
