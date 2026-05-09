## PR Title

```
[Attention][Feature] Implement Ascend NPU Tree Attention Backend
```

## PR Description

```markdown
### What this PR does / why we need it?

This PR implements the `TREE_ATTN` backend for Ascend NPUs to support tree-based speculative decoding (`speculative_token_tree`). It introduces `AscendTreeAttentionBackend`, `AscendTreeAttentionImpl`, and `AscendTreeAttentionMetadataBuilder`, along with utilities for converting GPU-format tree masks to NPU-compatible formats. The `EagleProposer` is extended with `_propose_tree` and `propose_tree` to support multi-level tree drafting.

**Key changes:**

- **`vllm_ascend/attention/backends/tree_attn.py`** — New `AscendTreeAttentionBackend` (686 lines). Decode uses `npu_fused_infer_attention_score` with TND layout and `sparse_mode=3`. Prefill uses BSH layout with `sparse_mode=0`. ACL Graph supported via workspace precomputation and `.out()` variant when in draft model context.
- **`vllm_ascend/spec_decode/eagle_proposer.py`** — Added `propose_tree` and `_propose_tree` methods to `AscendEagleProposer`, supporting multi-level tree drafting with per-level ACL Graph dispatch and slot mapping.
- **`vllm_ascend/attention/backends/__init__.py`** — Exports `AscendTreeAttentionBackend` and `AscendTreeAttentionMetadata`.
- **`vllm_ascend/attention/attention_v1.py`** — Registers `TREE_ATTN` backend overriding the GPU version.
- **`vllm_ascend/attention/attention_mask.py`** — Added `convert_tree_mask_for_npu` utility for GPU-to-NPU mask format conversion (float32 to int8, pad to 2048x2048).
- **`vllm_ascend/ascend_forward_context.py`** — Added `slot_mapping` parameter to `set_ascend_forward_context`.
- **`tests/ut/attention/test_tree_attn_npu.py`** — 11 unit tests covering mask conversion, ancestor relationships, depth counting, backend registration, metadata creation, and edge cases.

**Performance (NPU 910B2C vs CUDA A800, same vLLM dev version):**

| Config | CUDA A800 | Ascend NPU | Gap |
|--------|-----------|------------|-----|
| Tree n=2 | +15.6% | +14.3% | 1.3pp |
| Tree n=3 | +13.2% | +10.2% | 3.0pp |

Accept Length and per-position acceptance rates are consistent across both platforms (Mean Accept Length 1.46~1.50, Position 0 rate ~0.35).

### Does this PR introduce _any_ user-facing change?

Yes, it adds support for the `TREE_ATTN` backend on Ascend hardware. The `speculative_token_tree` configuration option works the same as on GPU. The TREE_ATTN backend is automatically registered when `vllm_ascend` is imported.

### How was this patch tested?

- **Unit tests**: 11 pytest cases covering mask conversion utilities, ancestor relationship logic, depth counting, backend registration, KV cache shape validation, metadata creation, and binary tree edge cases. All pass.
- **E2E benchmarks**: Tested with `speculative_token_tree` config (`tree n=2`, `tree n=3`) on Ascend NPU (910B2C), verified throughput improvement (+14.3%, +10.2%) vs standard speculative decoding baseline.
- **Cross-platform verification**: Same tree mask logic verified on CUDA A800 (5/5 tests pass), acceptance rates and lengths consistent between CUDA and NPU.
```

---

## Code Review 回复

### Review 1: output shape mismatch (critical) — 反驳

> This is not a bug. `npu_fused_infer_attention_score` with TND layout returns a 2D tensor of shape `[num_tokens, num_heads * head_size]`. The `.view()` call on line 699 reshapes it to 3D `[num_tokens, num_heads, head_size]`, then the assignment `output[:num_tokens] = attn_output[:num_tokens]` broadcasts the 3D view back into the 2D output tensor — this is identical to the pattern used in the standard decode path at `attention_v1.py:884-885` (`attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)` → `output[:num_tokens] = attn_output[:num_tokens]`).

### Review 2: block_table → block_tables (critical) — 接受

> Fixed. Changed `attn_metadata.block_table` to `attn_metadata.block_tables` in `eagle_proposer.py:1986` to match the `AscendMetadataForTree` attribute name.

### Review 3: .item() CPU sync (high) — 接受

> Fixed. Replaced `decode_meta.query_start_loc[-1].item()` with `num_tokens` which already holds the same value from line 615 (`num_tokens = decode_meta.num_actual_tokens`). This eliminates the host-device synchronization in the ACL Graph path.
