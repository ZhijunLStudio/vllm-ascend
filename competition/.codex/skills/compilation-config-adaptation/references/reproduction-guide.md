# Reproduction Guide — --compilation-config Adaptation

## Environment Setup

### NPU Machine (vllm-ascend)

```bash
# Prerequisites: Ascend 910B, CANN 8.5.1, torch-npu 2.9.0
cd /root/work/vllm-ascend

# Workaround for vLLM dev version parsing
export VLLM_VERSION=0.9.0

# Python path (editable install)
PYTHON=/usr/local/python3.11.14/bin/python
```

### CUDA Machine (vLLM baseline)

```bash
# Prerequisites: CUDA GPU, vLLM installed
pip install vllm
# Model will auto-download from HuggingFace
```

## Step 1: Run Unit Tests (NPU)

```bash
cd /root/work/vllm-ascend
export VLLM_VERSION=0.9.0
pytest tests/ut/test_platform.py -v
```

Expected: 35 passed, 1 skipped, ~0.20s.

Key tests to verify:
- `test_check_and_update_config_all_compilation_modes` — 4 modes work
- `test_check_and_update_config_dynamic_shapes_for_stock_modes` — pass-through/fallback logic

## Step 2: Run E2E Batch Tests (NPU)

```bash
cd /root/work/vllm-ascend
export VLLM_VERSION=0.9.0
./run_compilation_tests.sh
```

Logs saved to `./test_logs/compilation_<timestamp>/`.

Expected: 15/15 PASS. Verify these key behaviors in logs:

| Test | Expected Log |
|------|-------------|
| test_1 (NONE) | `Cudagraph mode FULL_AND_PIECEWISE is not compatible with compilation mode 0. Overriding to NONE.` |
| test_2 (VLLM_COMPILE) | `PIECEWISE compilation enabled on NPU` + `Replaying aclgraph` |
| test_3 (STOCK_TORCH_COMPILE) | `STOCK_TORCH_COMPILE compilation mode enabled on Ascend NPU.` |
| test_4 (DYNAMO_TRACE_ONCE) | `DYNAMO_TRACE_ONCE compilation mode enabled on Ascend NPU.` |
| test_11 (VLLM_COMPILE+UNBACKED) | `WARNING ... UNBACKED dynamic shapes type ... Falling back to BACKED.` |
| test_12 (VLLM_COMPILE+eval_guards) | `WARNING ... evaluate_guards=True ... Setting to False.` |
| test_9 (DYNAMO+assume_32) | `INFO ... assume_32_bit_indexing is enabled.` |
| test_15 (INVALID) | `ValidationError: Invalid compilation mode: INVALID` |

## Step 3: Run E2E Batch Tests (CUDA)

Copy `run_compilation_tests_cuda.sh` to the CUDA machine, then:

```bash
./run_compilation_tests_cuda.sh
```

This provides the baseline for comparison. Same 15 test cases, same warmup + 3 prompts.

## Step 4: Compare Results

Fill in the comparison table in `END_TO_END_TEST_REPORT.md` Section 7 with CUDA vs NPU throughput and behavior data.

## Troubleshooting

### vLLM version parsing error
If you see `ValueError: invalid literal for int()` related to version parsing:
```bash
export VLLM_VERSION=0.9.0
```
This is only needed for dev/main branch. Released versions don't need it.

### Triton backends conflict
If triton import fails with AMD/NVIDIA backend errors:
```bash
rm -rf /usr/local/python3.11.14/lib/python3.11/site-packages/triton/backends/amd
rm -rf /usr/local/python3.11.14/lib/python3.11/site-packages/triton/backends/nvidia
```
