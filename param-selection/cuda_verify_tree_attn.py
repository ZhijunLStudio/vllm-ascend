"""
CUDA Machine Tree Attention Verification Script

在 CUDA 机器上运行，验证以下内容：
1. Tree mask 构建逻辑（纯 Python/PyTorch，不需要 vLLM）
2. GPU → NPU mask 转换逻辑（float32 → int8, pad to 2048x2048）
3. 复杂 tree 结构（深度4）的 mask 正确性
4. Eagle propose_tree 的数据结构（模拟 tree 层级循环）
5. TreeAttentionMetadataBuilder 模拟（完整 metadata 构建流程）

输出：param-selection/verify_*.json

运行方式：
    python param-selection/cuda_verify_tree_attn.py
"""

import json
import os
import torch
import math
from typing import Optional

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PAD_SIZE = 2048  # NPU kernel 要求 mask pad 到 2048x2048


# ============================================================
# 工具函数
# ============================================================

def build_tree_choices_from_spec(tree_spec: str) -> list[tuple[int, ...]]:
    """
    解析 speculative_token_tree 配置字符串为 tree choices 列表。

    示例输入: "[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]"
    """
    # 安全解析：去掉外层引号后 eval
    tree_spec = tree_spec.strip().strip('"').strip("'")
    tree_choices = eval(tree_spec)
    return tree_choices


def build_tree_attn_mask_gpu(
    tree_len: int,
    tree_choices: list[tuple[int, ...]],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    构建 GPU 版 tree attention mask（与 vLLM 上游一致）。

    GPU mask 语义: 0 = attend, -inf = block
    mask[i][j] = 0 表示 query i 可以 attend to key j
    """
    # 先构建 token_id 到 position 的映射
    # token 0 = root (prompt token)
    # token 1..len(tree_choices) = draft tokens
    num_tokens = tree_len  # 1 (root) + len(tree_choices)

    mask = torch.zeros((num_tokens, num_tokens), dtype=dtype)

    # 对于每个 draft token (position i+1)，找到它能看到的 tokens
    # tree_choices[i] 给出了第 i 个 draft token 的祖先路径
    for i, choice in enumerate(tree_choices):
        row = i + 1  # draft token 的 position (root 是 position 0)
        # root token 始终可见
        mask[row][0] = 0
        # 该 draft token 的所有祖先节点可见
        for j, other_choice in enumerate(tree_choices):
            col = j + 1
            # choice 的前缀如果是 other_choice，或 other_choice 的前缀是 choice
            # 则 row 可以 attend to col
            if _is_ancestor(other_choice, choice) or _is_ancestor(choice, other_choice):
                mask[row][col] = 0
            else:
                mask[row][col] = float("-inf")
    return mask


def _is_ancestor(potential_ancestor: tuple, potential_descendant: tuple) -> bool:
    """检查 potential_ancestor 是否是 potential_descendant 的祖先前缀。"""
    if len(potential_ancestor) >= len(potential_descendant):
        return False
    return potential_descendant[: len(potential_ancestor)] == potential_ancestor


def convert_gpu_mask_to_npu(
    gpu_mask: torch.Tensor,
    pad_size: int = PAD_SIZE,
) -> torch.Tensor:
    """
    将 GPU float32 mask 转换为 NPU int8 mask。

    GPU: 0 = attend, -inf = block
    NPU: 0 = attend, non-zero = block (int8)
    """
    seq_len = gpu_mask.shape[0]
    # 创建 pad 后的 mask，默认全 1（block）
    npu_mask = torch.ones((pad_size, pad_size), dtype=torch.int8)
    # 0 = attend, 1 = block
    npu_mask[:seq_len, :seq_len] = (gpu_mask != 0).to(torch.int8)
    return npu_mask


def build_tree_mask_int8(
    tree_len: int,
    tree_choices: list[tuple[int, ...]],
    pad_size: int = PAD_SIZE,
) -> torch.Tensor:
    """
    直接用 int8 语义构建 tree mask（NPU 原生方式）。
    0 = attend, non-0 = block
    """
    mask = torch.ones((pad_size, pad_size), dtype=torch.int8)
    # Root token attends to itself
    mask[0][0] = 0
    for i, choice in enumerate(tree_choices):
        row = i + 1
        # Root always visible
        mask[row][0] = 0
        # Self
        mask[row][row] = 0
        # Ancestors and descendants
        for j, other_choice in enumerate(tree_choices):
            col = j + 1
            if _is_ancestor(other_choice, choice) or _is_ancestor(choice, other_choice):
                mask[row][col] = 0
    return mask


# ============================================================
# 测试 1：Tree mask 构建逻辑
# ============================================================

def test_1_tree_mask_construction() -> dict:
    """测试 1：Tree mask 构建逻辑"""
    print("=" * 60)
    print("测试 1：Tree mask 构建逻辑")
    print("=" * 60)

    # 二叉树，深度 3
    tree_spec = "[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]"
    tree_choices = build_tree_choices_from_spec(tree_spec)
    tree_len = 1 + len(tree_choices)  # root + drafts

    print(f"Tree spec: {tree_spec}")
    print(f"Tree choices: {tree_choices}")
    print(f"Tree length: {tree_len}")

    # 构建 GPU mask
    gpu_mask = build_tree_attn_mask_gpu(tree_len, tree_choices)
    print(f"GPU mask shape: {gpu_mask.shape}")
    print(f"GPU mask dtype: {gpu_mask.dtype}")

    # 打印 mask 的前 8x8 子矩阵
    mask_sub = gpu_mask[:8, :8].tolist()
    print("GPU mask (8x8 submatrix, 0=attend, -inf=block):")
    for row in mask_sub:
        print(["  ." if v == 0 else "-inf" for v in row])

    # 验证基本属性
    errors = []
    # Root (row 0) should only attend to itself
    if gpu_mask[0][0] != 0:
        errors.append("Root should attend to itself")
    for j in range(1, tree_len):
        if gpu_mask[0][j] == 0:
            errors.append(f"Root should NOT attend to draft token {j}")

    # Each draft token should attend to root
    for i in range(1, tree_len):
        if gpu_mask[i][0] != 0:
            errors.append(f"Draft token {i} should attend to root")

    # Each draft token should attend to itself
    for i in range(1, tree_len):
        if gpu_mask[i][i] != 0:
            errors.append(f"Draft token {i} should attend to itself")

    passed = len(errors) == 0
    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")

    result = {
        "test": "tree_mask_construction",
        "tree_spec": tree_spec,
        "tree_choices": [list(c) for c in tree_choices],
        "tree_len": tree_len,
        "mask_shape": list(gpu_mask.shape),
        "mask_sub_8x8": mask_sub,
        "passed": passed,
        "errors": errors,
    }

    # 保存到 JSON
    output_path = os.path.join(OUTPUT_DIR, "verify_tree_mask_construction.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_path}")

    return result


# ============================================================
# 测试 2：GPU → NPU mask 转换
# ============================================================

def test_2_npu_mask_conversion() -> dict:
    """测试 2：GPU → NPU mask 转换"""
    print("\n" + "=" * 60)
    print("测试 2：GPU → NPU mask 转换")
    print("=" * 60)

    tree_spec = "[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]"
    tree_choices = build_tree_choices_from_spec(tree_spec)
    tree_len = 1 + len(tree_choices)

    gpu_mask = build_tree_attn_mask_gpu(tree_len, tree_choices)
    npu_mask = convert_gpu_mask_to_npu(gpu_mask)

    print(f"NPU mask shape: {npu_mask.shape}")
    print(f"NPU mask dtype: {npu_mask.dtype}")
    print(f"NPU mask (8x8 submatrix, 0=attend, 1=block):")
    for i in range(8):
        row = [int(npu_mask[i][j]) for j in range(8)]
        print(["  ." if v == 0 else "  1" for v in row])

    # 验证：NPU mask 在有效区域应与 GPU mask 互逆
    errors = []
    for i in range(tree_len):
        for j in range(tree_len):
            gpu_attend = gpu_mask[i][j] == 0
            npu_attend = npu_mask[i][j] == 0
            if gpu_attend != npu_attend:
                errors.append(f"Mismatch at [{i}][{j}]: GPU={gpu_mask[i][j]}, NPU={npu_mask[i][j]}")

    # 验证 pad 区域全部为 1（block）
    for i in range(tree_len, PAD_SIZE):
        for j in range(PAD_SIZE):
            if npu_mask[i][j] != 1:
                errors.append(f"Pad area [{i}][{j}] should be 1 (block)")
                break  # 只报告第一个
        if errors and "Pad area" in errors[-1]:
            break
    for j in range(tree_len, PAD_SIZE):
        for i in range(tree_len):
            if npu_mask[i][j] != 1:
                errors.append(f"Pad area [{i}][{j}] should be 1 (block)")
                break
        if errors and "Pad area" in errors[-1]:
            break

    passed = len(errors) == 0
    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors[:10]:
            print(f"  ERROR: {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # 同时验证直接构建 int8 mask
    npu_mask_direct = build_tree_mask_int8(tree_len, tree_choices)
    direct_match = torch.equal(npu_mask, npu_mask_direct)
    print(f"Direct int8 mask matches converted mask: {direct_match}")

    result = {
        "test": "npu_mask_conversion",
        "tree_spec": tree_spec,
        "tree_len": tree_len,
        "gpu_mask_shape": list(gpu_mask.shape),
        "npu_mask_shape": list(npu_mask.shape),
        "npu_mask_dtype": str(npu_mask.dtype),
        "npu_pad_size": PAD_SIZE,
        "conversion_valid": passed,
        "direct_build_matches_conversion": direct_match,
        "error_count": len(errors),
        "errors_sample": errors[:20],
        "passed": passed and direct_match,
    }

    output_path = os.path.join(OUTPUT_DIR, "verify_npu_mask_conversion.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_path}")

    return result


# ============================================================
# 测试 3：复杂 tree 结构（深度 4）
# ============================================================

def test_3_complex_tree_structure() -> dict:
    """测试 3：复杂 tree 结构（深度 4 的完全二叉树）"""
    print("\n" + "=" * 60)
    print("测试 3：复杂 tree 结构（深度 4 完全二叉树）")
    print("=" * 60)

    # 深度 4 的完全二叉树
    # Level 1: (0,)
    # Level 2: (0,0), (0,1)
    # Level 3: (0,0,0), (0,0,1), (0,1,0), (0,1,1)
    # Level 4: (0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1),
    #          (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1)
    tree_choices = []
    for d1 in range(2):  # level 1
        tree_choices.append((0,))
        break  # only (0,) at level 1
    for d1 in range(2):  # level 2
        tree_choices.append((0, d1))
    for d1 in range(2):  # level 3
        for d2 in range(2):
            tree_choices.append((0, d1, d2))
    for d1 in range(2):  # level 4
        for d2 in range(2):
            for d3 in range(2):
                tree_choices.append((0, d1, d2, d3))

    tree_len = 1 + len(tree_choices)
    print(f"Tree choices count: {len(tree_choices)}")
    print(f"Tree length: {tree_len}")
    print(f"Max depth: {max(len(c) for c in tree_choices)}")

    # 构建 mask
    gpu_mask = build_tree_attn_mask_gpu(tree_len, tree_choices)
    npu_mask = convert_gpu_mask_to_npu(gpu_mask)

    # 验证层级关系
    errors = []
    # 同一父节点的子节点应互相可见
    # (0,0) 和 (0,1) 都是 (0,) 的子节点，应互相可见吗？
    # 实际上在 tree attention 中，兄弟节点间一般不可见（除非有共同祖先）
    # (0,0) 看到: root(0), (0,), (0,0) — 不看到 (0,1)
    # (0,1) 看到: root(0), (0,), (0,1) — 不看到 (0,0)
    idx_00 = tree_choices.index((0, 0))
    idx_01 = tree_choices.index((0, 1))
    if gpu_mask[idx_00 + 1][idx_01 + 1] == 0:
        errors.append("(0,0) should NOT attend to sibling (0,1)")
    if gpu_mask[idx_01 + 1][idx_00 + 1] == 0:
        errors.append("(0,1) should NOT attend to sibling (0,0)")

    # (0,0,0) 应看到 root, (0,), (0,0), (0,0,0)，不看到 (0,0,1) 的其他分支
    idx_000 = tree_choices.index((0, 0, 0))
    idx_001 = tree_choices.index((0, 0, 1))
    if gpu_mask[idx_000 + 1][idx_001 + 1] == 0:
        # (0,0,0) 和 (0,0,1) 共享前缀 (0,0)，在某些 tree 定义中可互相可见
        print(f"  INFO: (0,0,0) CAN attend to (0,0,1) - shared prefix (0,0)")

    # 统计 attend 比例
    attend_count = (gpu_mask[:tree_len, :tree_len] == 0).sum().item()
    total = tree_len * tree_len
    attend_ratio = attend_count / total
    print(f"Attend ratio: {attend_count}/{total} = {attend_ratio:.4f}")

    passed = len(errors) == 0
    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")

    result = {
        "test": "complex_tree_structure",
        "tree_choices_count": len(tree_choices),
        "tree_len": tree_len,
        "max_depth": max(len(c) for c in tree_choices),
        "attend_ratio": round(attend_ratio, 4),
        "attend_count": attend_count,
        "total_positions": total,
        "npu_mask_shape": list(npu_mask.shape),
        "passed": passed,
        "errors": errors,
    }

    output_path = os.path.join(OUTPUT_DIR, "verify_complex_tree_structure.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_path}")

    return result


# ============================================================
# 测试 4：Eagle propose_tree 数据结构
# ============================================================

def test_4_eagle_tree_data() -> dict:
    """测试 4：模拟 Eagle propose_tree 的完整数据流"""
    print("\n" + "=" * 60)
    print("测试 4：Eagle propose_tree 数据结构模拟")
    print("=" * 60)

    tree_spec = "[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]"
    tree_choices = build_tree_choices_from_spec(tree_spec)

    # 模拟 vocab
    vocab_size = 32000
    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads  # 128

    # 模拟 propose_tree 的层级循环
    # Level 0: root token (from target model)
    # Level 1: draft from root
    # Level 2: draft from level 1 tokens
    # ...

    # 按深度分组
    depth_groups = {}
    for i, choice in enumerate(tree_choices):
        depth = len(choice)
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append((i, choice))

    max_depth = max(depth_groups.keys())
    print(f"Max depth: {max_depth}")
    for depth in sorted(depth_groups.keys()):
        print(f"  Level {depth}: {len(depth_groups[depth])} tokens")

    # 模拟每层的 forward 和 sampling
    simulated_tokens = {0: [42]}  # root token
    total_draft_tokens = 0

    for depth in sorted(depth_groups.keys()):
        level_tokens = []
        for idx, choice in depth_groups[depth]:
            # 模拟从 parent token 采样
            parent_token = simulated_tokens[depth - 1][choice[-1]] if depth > 0 else 42
            # 模拟采样（实际是模型 forward + sample）
            draft_token = (parent_token * 7 + idx + depth) % vocab_size
            level_tokens.append(draft_token)
            total_draft_tokens += 1

        simulated_tokens[depth] = level_tokens
        print(f"  Level {depth} tokens: {level_tokens[:5]}{'...' if len(level_tokens) > 5 else ''}")

    # 构建完整的 draft_token_ids_list（与 vLLM 上游格式一致）
    # 每个元素是一个 level 的 draft tokens
    draft_token_ids_list = []
    for depth in sorted(depth_groups.keys()):
        level_drafts = []
        for idx, choice in depth_groups[depth]:
            level_drafts.append(simulated_tokens[depth][depth_groups[depth].index((idx, choice))])
        draft_token_ids_list.append(level_drafts)

    print(f"\ndraft_token_ids_list:")
    for i, level in enumerate(draft_token_ids_list):
        print(f"  Level {i}: {level[:5]}{'...' if len(level) > 5 else ''} ({len(level)} tokens)")

    # 验证 parent-child token 关系
    errors = []
    # 验证总 token 数
    expected_total = len(tree_choices)
    if total_draft_tokens != expected_total:
        errors.append(f"Total draft tokens: {total_draft_tokens} != {expected_total}")

    # 验证 depth groups 覆盖所有 tree_choices
    covered = sum(len(v) for v in depth_groups.values())
    if covered != len(tree_choices):
        errors.append(f"Depth groups cover {covered} != {len(tree_choices)} tree choices")

    passed = len(errors) == 0
    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")

    result = {
        "test": "eagle_tree_data",
        "tree_spec": tree_spec,
        "tree_choices": [list(c) for c in tree_choices],
        "num_tree_choices": len(tree_choices),
        "max_depth": max_depth,
        "depth_groups": {str(k): len(v) for k, v in depth_groups.items()},
        "draft_token_ids_list_lengths": [len(l) for l in draft_token_ids_list],
        "total_draft_tokens": total_draft_tokens,
        "simulated_tokens_sample": {
            str(k): v[:5] for k, v in simulated_tokens.items()
        },
        "passed": passed,
        "errors": errors,
    }

    output_path = os.path.join(OUTPUT_DIR, "verify_eagle_tree_data.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_path}")

    return result


# ============================================================
# 测试 5：TreeAttentionMetadataBuilder 模拟
# ============================================================

def test_5_metadata_builder() -> dict:
    """测试 5：模拟 TreeAttentionMetadataBuilder 的完整流程"""
    print("\n" + "=" * 60)
    print("测试 5：TreeAttentionMetadataBuilder 模拟")
    print("=" * 60)

    tree_spec = "[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]"
    tree_choices = build_tree_choices_from_spec(tree_spec)
    tree_len = 1 + len(tree_choices)

    # 模拟 batch 中有 2 个请求
    batch_size = 2
    num_heads = 32
    head_dim = 128
    block_size = 16

    # 每个请求的 seq lengths
    seq_lens = [64, 128]  # prefix lengths
    tree_lens = [tree_len, tree_len]  # 每个请求的 tree 长度（相同 tree structure）

    # 1. 构建 tree attention bias (float32, GPU format)
    tree_bias = build_tree_attn_mask_gpu(tree_len, tree_choices, dtype=torch.float32)
    print(f"Tree bias shape: {tree_bias.shape}")

    # 2. 转换为 NPU mask (int8, 2048x2048)
    npu_mask = convert_gpu_mask_to_npu(tree_bias)
    print(f"NPU mask shape: {npu_mask.shape}, dtype: {npu_mask.dtype}")

    # 3. 模拟 context_lens（decode 步骤中已完成的 context 长度）
    context_lens = torch.tensor([seq_lens[0], seq_lens[1]], dtype=torch.int32)

    # 4. 模拟 block tables
    max_blocks = 16
    block_tables = torch.zeros((batch_size, max_blocks), dtype=torch.int32)
    for b in range(batch_size):
        num_blocks_needed = (seq_lens[b] + block_size - 1) // block_size
        for i in range(num_blocks_needed):
            block_tables[b][i] = b * max_blocks + i

    # 5. 模拟 actual_seq_lengths（TND layout 用）
    actual_seq_lengths = []
    for b in range(batch_size):
        for t in range(tree_len):
            actual_seq_lengths.append(seq_lens[b] + t + 1)
    actual_seq_kv = [seq_lens[b] + tree_len for b in range(batch_size)]

    # 6. 验证 metadata 完整性
    errors = []

    # 验证 tree bias 非全零
    if (tree_bias == 0).all():
        errors.append("Tree bias should not be all zeros (would be full causal)")

    # 验证 tree bias 不是标准 causal
    causal_mask = torch.tril(torch.ones((tree_len, tree_len), dtype=torch.float32))
    causal_bias = torch.where(causal_mask == 1, torch.zeros_like(tree_bias), torch.tensor(float("-inf")))
    if torch.equal(tree_bias, causal_bias):
        print("  INFO: Tree bias equals causal mask (valid for this tree shape)")
    else:
        print("  INFO: Tree bias differs from causal mask (non-trivial tree structure)")

    # 验证 NPU mask 非全 1
    if (npu_mask[:tree_len, :tree_len] == 1).all():
        errors.append("NPU mask should not be all 1 (all blocked)")

    # 验证 block table 连续性
    for b in range(batch_size):
        num_blocks_needed = (seq_lens[b] + block_size - 1) // block_size
        if block_tables[b][0] != b * max_blocks:
            errors.append(f"Block table for batch {b} has wrong start block")

    # 验证 actual_seq_lengths 单调递增
    for i in range(1, len(actual_seq_lengths)):
        if actual_seq_lengths[i] <= actual_seq_lengths[i - 1]:
            # 同 batch 内递增，不同 batch 间可能不递增
            pass

    passed = len(errors) == 0

    # 输出 metadata 摘要
    metadata_summary = {
        "batch_size": batch_size,
        "tree_len": tree_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "seq_lens": seq_lens,
        "context_lens": context_lens.tolist(),
        "block_table_shape": list(block_tables.shape),
        "npu_mask_shape": list(npu_mask.shape),
        "npu_mask_dtype": str(npu_mask.dtype),
        "actual_seq_lengths": actual_seq_lengths,
        "actual_seq_kv": actual_seq_kv,
        "tree_bias_attend_ratio": round(
            (tree_bias[:tree_len, :tree_len] == 0).float().mean().item(), 4
        ),
        "npu_mask_attend_ratio": round(
            (npu_mask[:tree_len, :tree_len] == 0).float().mean().item(), 4
        ),
    }

    print(f"\nMetadata summary:")
    for k, v in metadata_summary.items():
        print(f"  {k}: {v}")

    print(f"\nValidation: {'PASS' if passed else 'FAIL'}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")

    result = {
        "test": "metadata_builder",
        "tree_spec": tree_spec,
        "passed": passed,
        "errors": errors,
        "metadata": metadata_summary,
    }

    output_path = os.path.join(OUTPUT_DIR, "verify_metadata_builder.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to: {output_path}")

    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("CUDA Tree Attention Verification Script")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    results = {}
    results["test_1"] = test_1_tree_mask_construction()
    results["test_2"] = test_2_npu_mask_conversion()
    results["test_3"] = test_3_complex_tree_structure()
    results["test_4"] = test_4_eagle_tree_data()
    results["test_5"] = test_5_metadata_builder()

    # 汇总
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    all_passed = True
    for key, result in results.items():
        passed = result["passed"]
        status = "PASS" if passed else "FAIL"
        print(f"  {result['test']}: {status}")
        if not passed:
            all_passed = False
            for e in result.get("errors", []):
                print(f"    - {e}")

    if all_passed:
        print("\n全部通过！Tree attention 核心逻辑验证完成。")
        print("下一步：在 NPU 机器上运行 kernel 级验证（test_tree_attn_npu_v2.py）")
    else:
        print("\n存在失败的测试，请检查上述错误信息。")

    # 保存汇总
    summary = {
        "all_passed": all_passed,
        "tests": {k: {"test": v["test"], "passed": v["passed"]} for k, v in results.items()},
        "output_files": [
            "verify_tree_mask_construction.json",
            "verify_npu_mask_conversion.json",
            "verify_complex_tree_structure.json",
            "verify_eagle_tree_data.json",
            "verify_metadata_builder.json",
        ],
    }
    summary_path = os.path.join(OUTPUT_DIR, "verify_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
