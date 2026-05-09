### What this PR does / why we need it?

Previously, `--compilation-config mode` values `STOCK_TORCH_COMPILE` and `DYNAMO_TRACE_ONCE` were silently downgraded to eager mode on Ascend NPU. `dynamic_shapes_config` was also ignored.

This PR enables all four `CompilationMode` values and properly handles `dynamic_shapes_config`:
- `get_compile_backend()` returns `"npu"` (torch_npu Inductor backend) for stock/dynamo modes
- `VLLM_COMPILE` overrides backend to `"eager"` — actual compilation by AscendCompiler → TorchAIR ACL Graph
- `VLLM_COMPILE + UNBACKED` auto-downgrades to `BACKED` with a warning (ACL Graph static capture limitation). CUDA directly crashes on this combination — our graceful downgrade provides better UX.
- Other modes pass `dynamic_shapes_config` through unchanged, handled by vLLM's upper layer

### Does this PR introduce any user-facing change?

- `--compilation-config mode=STOCK_TORCH_COMPILE` and `mode=DYNAMO_TRACE_ONCE` now work on NPU instead of being silently ignored
- `--compilation-config '{"mode":"VLLM_COMPILE","dynamic_shapes_config":{"type":"unbacked"}}'` gracefully falls back to BACKED with a warning instead of crashing
- `get_compile_backend()` returns `"npu"` instead of the old `"vllm_ascend.compilation.compiler_interface.AscendCompiler"`

### How was this patch tested?

- Unit tests: `pytest tests/ut/test_platform.py -v` (35 passed)
- E2E on Ascend 910B with Qwen2.5-0.5B-Instruct: 15/15 test cases passed across all mode × dynamic_shapes_config combinations
- E2E on NVIDIA A800 with vLLM upstream as baseline comparison (12/15 passed, 3 UNBACKED cases expected failure on CUDA)

**Performance (Qwen2.5-0.5B-Instruct, max_model_len=256):**

| Mode | NPU 910B (toks/s) | CUDA A800 (toks/s) |
|------|--------------------|--------------------|
| NONE | 100.12 (1.00x) | 141.29 (1.00x) |
| STOCK_TORCH_COMPILE | 101.53 (1.01x) | 185.96 (1.32x) |
| DYNAMO_TRACE_ONCE | 126.52 (1.26x) | 191.28 (1.35x) |
| VLLM_COMPILE | 187.87 (1.88x) | 507.24 (3.59x) |
