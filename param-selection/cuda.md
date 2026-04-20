# cpu-offload-params CUDA 端测试指南

## 一、测试目标

在 CUDA GPU 上使用与 NPU 相同的模型和配置，验证 `--cpu-offload-params` prefetch 后端的功能正确性和性能基线，作为 NPU 实现的对比参照。

测试覆盖 **eager 模式** 和 **CUDA Graph (PIECEWISE) 模式**。

## 二、测试环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA A800-SXM4-80GB 或同级（需 >=40GB 显存） |
| Python | 3.11+ |
| vLLM | dev 版本（与 NPU 端同版本 commit） |
| 模型 | Meta-Llama-3.1-8B-Instruct |

## 三、环境准备

```bash
cd /path/to/vllm-ascend
git checkout feature/cpu-offload-params

MODEL_PATH="/path/to/Meta-Llama-3.1-8B-Instruct"
export CUDA_VISIBLE_DEVICES=0
```

## 四、运行测试

```bash
# 完整测试 (eager + CUDA Graph)
python param-selection/run_cuda_offload_tests.py --model $MODEL_PATH --mode both

# 只跑 eager 模式
python param-selection/run_cuda_offload_tests.py --model $MODEL_PATH --mode eager

# 只跑 CUDA Graph 模式
python param-selection/run_cuda_offload_tests.py --model $MODEL_PATH --mode graph

# 跳过单元测试，只跑功能和性能
python param-selection/run_cuda_offload_tests.py --model $MODEL_PATH --mode both --skip-unit
```

## 五、测试项说明

### 5.1 单元测试 (6 个)

| 测试 | 说明 |
|------|------|
| `test_is_uva_available` | CUDA 应返回 True |
| `test_create_offloader_prefetch` | prefetch -> PrefetchOffloader |
| `test_create_offloader_uva` | UVA -> UVAOffloader |
| `test_create_offloader_noop` | 空配置 -> NoopOffloader |
| `test_static_buffer_pool` | CUDA 设备上分配 buffer |
| `test_layer_selection` | group=8, num=2 -> 8 层 offloaded |

### 5.2 功能测试

每个 offload 配置在 eager 和 graph 模式下各测一次，验证：
- 模型正常加载
- 生成正常完成
- **输出与 baseline 一致**（temperature=0.0）

| 配置 | 说明 |
|------|------|
| baseline | 无 offload |
| prefetch_g8n2s1 | 32层中 offload 16层 (50%) |
| prefetch_g8n2s2 | 32层中 offload 16层, step=2 |
| prefetch_g4n1s1 | 32层中 offload 8层 (25%) |
| prefetch_g4n1s2 | 32层中 offload 8层, step=2 |

### 5.3 性能基准

每个配置 x 每个模式，采集：
- **Warmup 时间**: 首次推理延迟（含 graph capture）
- **单请求延迟**: 5 个 prompt 的平均延迟
- **批量吞吐**: 10 个 prompt 串行的 tokens/s
- **vs Baseline**: 相对无 offload 的加速/减速比

## 六、结果文件

测试完成后生成 `cuda_offload_results.json`，格式与 NPU 端 `npu_offload_results.json` 完全一致，可直接对比。

将此 JSON 拷回 NPU 环境，与 NPU 结果对比即可生成完整报告。
