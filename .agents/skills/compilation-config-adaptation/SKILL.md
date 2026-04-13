---
name: compilation-config-adaptation
description: "Adapt vLLM's --compilation-config parameter (CompilationMode, DynamicShapesConfig) to Ascend NPU platform. Ensures all four compilation modes work correctly and dynamic shape configurations are properly handled with ACL Graph constraints."
---

# Compilation Config Adaptation for Ascend NPU

## Overview

Adapt vLLM's `--compilation-config` parameter to Ascend NPU, covering:
- `mode`: CompilationMode enum (NONE / VLLM_COMPILE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE)
- `dynamic_shapes_config`: DynamicShapesConfig (type / evaluate_guards / assume_32_bit_indexing)

Goal: eliminate silent-ignore behavior — every user-set parameter must either take effect correctly or produce a clear warning with guidance.

## Read Order

1. Start with this SKILL.md.
2. Read `references/reproduction-guide.md` for step-by-step reproduction instructions.

## Architecture Summary

### Four Compilation Modes on NPU

| Mode | Mechanism on NPU | cudagraph_mode default | use_inductor |
|------|-----------------|----------------------|-------------|
| `NONE` | Pure eager, no compilation | `NONE` | N/A |
| `VLLM_COMPILE` | VllmBackend → AscendCompiler → TorchAIR ACL Graph | `PIECEWISE` | `False` |
| `STOCK_TORCH_COMPILE` | torch.compile(backend="npu") via vLLM upper layer | `NONE` | platform default |
| `DYNAMO_TRACE_ONCE` | Single Dynamo trace (backend="npu") via vLLM upper layer | `NONE` | platform default |

### Key Design Decisions

1. **`dynamic_shapes_config` is NOT implemented at the platform layer.** The runtime behavior is fully handled by vLLM's `TorchCompileWithNoGuardsWrapper` (wrapper.py) and `decorators.py`. CUDA platform also does not handle it. Our platform layer only does compatibility checks for VLLM_COMPILE mode (because ACL Graph captures static graphs).

2. **UNBACKED → BACKED fallback for VLLM_COMPILE only.** ACL Graph is pure static graph capture (like CUDA Graph). UNBACKED dynamic shapes mark tensor dimensions as symbolic, which conflicts with static capture. The fallback produces a clear warning with alternative mode suggestions.

3. **`evaluate_guards=True` → False for VLLM_COMPILE only.** ACL Graph path does not support guard evaluation. Same pattern: warning + fallback.

4. **`get_compile_backend()` returns `"npu"`** — torch_npu provides an Inductor-based backend registered as `"npu"`. For STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE, this is used as `torch.compile(model, backend='npu')`. For VLLM_COMPILE, `compilation_config.backend` is overridden to `"eager"` so `make_compiler()` uses EagerAdaptor (actual compilation is handled by AscendCompiler → TorchAIR ACL Graph).

### Platform Layer Code Location

`vllm_ascend/platform.py` — `NPUPlatform.check_and_update_config()`

Key code sections:
- Lines ~327-337: STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE mode handling (log + default cudagraph_mode)
- Lines ~379-399: VLLM_COMPILE + PIECEWISE handling (ACL Graph setup)
- Lines ~456-493: `dynamic_shapes_config` handling (VLLM_COMPILE fallback logic)

## Hard Constraints

- Never modify vLLM upstream code — all changes go in `vllm-ascend`.
- `simple_compile_backend = "npu"` uses torch_npu's Inductor backend. Do not change to "eager" or "inductor".
- `get_compile_backend()` returns `"npu"`. For VLLM_COMPILE mode, `compilation_config.backend` is set to `"eager"` in `check_and_update_config` so `make_compiler()` uses EagerAdaptor.
- UNBACKED fallback is only for VLLM_COMPILE mode. Other modes must pass through.
- `VLLM_USE_BYTECODE_HOOK=0` is required for UNBACKED mode (vLLM requirement, not Ascend-specific).

## Testing

### Unit Tests

Located in `tests/ut/test_platform.py`:
- `test_check_and_update_config_all_compilation_modes` — all 4 modes set correctly
- `test_check_and_update_config_dynamic_shapes_for_stock_modes` — pass-through vs fallback logic
- `test_check_and_update_config_enforce_eager_mode` — enforce_eager forces NONE

Run: `pytest tests/ut/test_platform.py -v`

### End-to-End Tests

NPU batch test script: `run_compilation_tests.sh`
CUDA baseline script: `run_compilation_tests_cuda.sh`

Both scripts use identical test cases (15 total: 4 mode + 6 dynamic_shapes + 4 mixed + 1 error handling) with warmup + 3 variable-length prompts + baseline correctness comparison.

## When to Use This Skill

Use when:
- Modifying compilation-related code in `vllm_ascend/platform.py`
- Debugging compilation mode behavior on Ascend NPU
- Adding new compilation features that need platform-layer adaptation
- Investigating why a compilation config parameter is not taking effect on NPU
