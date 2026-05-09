---
name: speculative-token-tree-adaptation
description: "Adapt vLLM's speculative_token_tree (EAGLE tree-structured speculative decoding) to Ascend NPU. Implements TREE_ATTN backend using npu_fused_infer_attention_score with TND layout and sparse_mode=3, replacing GPU FlashAttention."
---

# Speculative Token Tree Adaptation for Ascend NPU

## Overview

Adapt vLLM's `speculative_token_tree` (EAGLE tree-structured speculative decoding) to Ascend NPU. This replaces the GPU `TREE_ATTN` backend (FlashAttention) with an NPU-compatible backend using `npu_fused_infer_attention_score` with TND layout and `sparse_mode=3`.

The implementation enables tree-structured draft token generation where each level explores multiple candidate tokens, improving acceptance probability and throughput. On NPU 910B2C, this achieves +14.3% throughput improvement (tree n=2) vs speculative decoding baseline.

## When to Use

Use this skill when:
- Adding tree attention support for Ascend NPU
- Debugging `speculative_token_tree` configuration issues on NPU
- Extending the `TREE_ATTN` backend with new NPU kernel features
- Investigating tree mask format conversion (float32 ↔ int8) for NPU

## Architecture

### Backend Registration

The NPU `TREE_ATTN` backend overrides the GPU version:
```python
# vllm_ascend/attention/attention_v1.py
register_attention_backend("TREE_ATTN", AscendTreeAttentionBackend)
```

### Key Files

| File | Role |
|------|------|
| `vllm_ascend/attention/backends/tree_attn.py` | Core implementation (~686 lines): AscendTreeAttentionBackend, AscendTreeAttentionImpl, AscendTreeAttentionMetadataBuilder |
| `vllm_ascend/spec_decode/eagle_proposer.py` | propose_tree + _propose_tree methods for multi-level tree drafting |
| `vllm_ascend/attention/attention_mask.py` | GPU→NPU mask conversion (float32 to int8, pad to 2048×2048) |
| `vllm_ascend/ascend_forward_context.py` | slot_mapping parameter for tree drafting |

### NPU vs GPU Core Differences

| Aspect | GPU (FlashAttention) | NPU |
|--------|---------------------|-----|
| Attention kernel | FlashAttention | `npu_fused_infer_attention_score` |
| Mask dtype | float32 (0=attend, -inf=block) | int8 (0=attend, 1=block) |
| Mask size | tree_len × tree_len | 2048 × 2048 (padded) |
| Mask caching | None | Precomputed slice masks |

### Key Design Decisions

1. **Mask format conversion**: GPU uses float32 masks, NPU kernel requires int8 masks padded to 2048×2048. We precompute all possible `[start:end, start:end]` slice masks at init time to avoid runtime overhead.

2. **ACL Graph workspace precomputation**: Uses `npu_fused_infer_attention_score_get_max_workspace()` + `.out()` variant instead of ExternalEvent sync. Workspace is computed once per batch size and reused.

3. **No `.tolist()` calls**: `actual_seq_lengths` kept as tensors (`.to(torch.int64)`) to avoid CPU-GPU synchronization during tree level forward passes.

4. **Root token semantics**: All tokens attend to root (`[:, 0] = 0` in the attention mask), not root attending to all tokens. This matches upstream vLLM behavior.

## Hard Constraints

- NPU kernel: mask MUST be padded to 2048×2048 for `sparse_mode=3`
- Draft model must be EAGLE3-compatible (e.g., `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`)
- ACL Graph mode requires workspace precomputation for tree attention
- `speculative_token_tree` config format: `[(0,), (0, 0)]` for n=2, `[(0,), (0, 0), (0, 1)]` for n=3

## Testing

### Unit Tests (11 tests)

```bash
pytest tests/ut/attention/test_tree_attn_npu.py -v
```

Covers: mask conversion, ancestor relationships, depth counting, backend registration, KV cache validation, metadata creation, edge cases.

### E2E Tests

```bash
# NPU server with tree n=2
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --max-model-len 2048 \
  --speculative-config '{"method":"eagle3","model":"/data/models/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":2,"speculative_token_tree":"[(0,), (0, 0)]"}'
```
