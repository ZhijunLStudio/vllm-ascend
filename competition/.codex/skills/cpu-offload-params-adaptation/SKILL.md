---
name: cpu-offload-params-adaptation
description: "Adapt vLLM's --cpu-offload-params prefetch backend to Ascend NPU. Implements NPUPrefetchOffloader using torch.npu.Stream/Event to replace all torch.cuda.* APIs. Supports both eager and ACL Graph modes with event-based stream forking."
---

# CPU Offload Params Adaptation for Ascend NPU

## Overview

Adapt vLLM's `--cpu-offload-params` (prefetch backend) to Ascend NPU. This replaces all `torch.cuda.Stream`/`torch.cuda.Event` calls with `torch.npu.*` equivalents, and integrates with ACL Graph via event-based stream forking.

The prefetch backend offloads selected layer parameters to CPU pinned memory and asynchronously prefetches them to NPU HBM before each layer's forward pass. This trades throughput for GPU memory savings, allowing larger models or batch sizes on memory-constrained devices.

## When to Use

Use this skill when:
- Adapting vLLM's offload/prefetch mechanism to NPU
- Debugging `--cpu-offload-params` on Ascend hardware
- Adding new offloading backends (UVA not supported on NPU)
- Investigating ACL Graph compatibility with async copy operations

## Architecture

### Key Files

| File | Role |
|------|------|
| `vllm_ascend/offloader/npu_prefetch.py` | Core: NPUPrefetchOffloader + _NPUModuleOffloader (~280 lines) |
| `vllm_ascend/patch/worker/patch_offloader.py` | Factory patch: create_offloader → NPUPrefetchOffloader |
| `vllm_ascend/compilation/acl_graph.py` | ACL Graph integration (sync_prev_onload, join_after_forward) |
| `vllm_ascend/platform.py` | is_uva_available() → False |
| `vllm_ascend/worker/model_runner_v1.py` | post_init() call after model load |

### How It Works

```
CPU pinned memory: [layer_N+1 params] [layer_N+2 params] ...
                         │
                    copy_stream (async, non_blocking)
                         │
NPU HBM: [static buffer pool, prefetch_step slots]
                         │
              forward hook: wait_prefetch → compute → start_prefetch(next)
```

### ACL Graph Compatibility

The key innovation is **event-based stream forking**:

```python
# During ACL Graph capture:
fork_event = torch.npu.Event()
torch.npu.current_stream().record_event(fork_event)  # compute stream
self.copy_stream.wait_event(fork_event)              # copy stream waits
with torch.npu.stream(self.copy_stream):
    gpu_buffer.copy_(cpu_storage, non_blocking=True) # NOT recorded in graph
self._copy_done_event.record(self.copy_stream)
```

Three ACL Graph integration points in `acl_graph.py`:
1. **Before capture**: `get_offloader().sync_prev_onload()` — ensure pre-capture copies complete
2. **Inside capture after forward**: `get_offloader().join_after_forward()` — join unwaited copy events
3. **Before replay**: `get_offloader().sync_prev_onload()` — sync from capture-to-replay transition

### Factory Patch (Three Levels)

To ensure the NPU offloader is used regardless of import path:
1. `offloader_base.create_offloader = _npu_create_offloader` — module source
2. `offloader_pkg.create_offloader = _npu_create_offloader` — package re-export
3. `gpu_mr_module.create_offloader = _npu_create_offloader` — from-import capture

### UVA Degradation

NPU does not support CUDA's Unified Virtual Addressing. `is_uva_available()` returns `False`, and UVA backend requests are degraded to `NoopOffloader` with a clear warning directing users to prefetch backend instead.

## Hard Constraints

- Parameters must be in CPU **pinned** memory for async copy (NPU supports pin_memory)
- UVA backend is NOT available on NPU
- Reuses upstream `ParamInfo`, `StaticBufferPool`, `_CpuParamOffloader` (no CUDA-specific code)
- patch_offloader must be imported BEFORE `gpu_model_runner` (import order in `worker.py`)
- `post_init()` must be explicitly called in `NPUModelRunner.load_model()`

## Testing

### Unit Tests (13 tests)

```bash
pytest tests/ut/offloader/test_npu_prefetch_offloader.py -v
```

Covers: NPU stream/event creation, layer selection, factory patch, buffer pool, module offloader.

### E2E Tests (7 tests)

```bash
pytest tests/e2e/singlecard/test_cpu_offload_params.py -v
```

Covers: eager mode, ACL Graph mode, output correctness vs baseline, UVA graceful degradation.

### Unified Cross-Platform Tests

```bash
python param-selection/run_cuda_offload_tests.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct --mode both
```
