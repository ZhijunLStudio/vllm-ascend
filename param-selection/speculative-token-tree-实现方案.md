# speculative_token_tree 实现方案

## 背景

vLLM 的 speculative decoding 支持 tree 结构，允许多个 draft token 并行验证，比线性 speculative decoding 有更高的接受率和更快的推理速度。

当前 Ascend 平台：
- `eagle_proposer.py`（1712行）只实现了线性的 draft/verify 流程
- 有 TODO 注释：`# TODO(wenlong): get more than one token for tree attention`
- `npu_fused_infer_attention_score` 支持 `atten_mask` 参数（和 GPU 的 `qq_bias` 概念兼容）
- **完全没有 tree attention 实现**

## GPU 端参考实现

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/attention/backends/tree_attn.py` | 445行 | TreeAttentionBackend + metadata builder |
| `vllm/v1/spec_decode/eagle.py` | 1792行 | EagleProposer 含 propose_tree（170行） |

### Tree Attention 原理

标准 causal attention：每个 token 只能看到自己和之前的 token（上三角 mask）

Tree attention：draft tokens 构成树状结构，每个节点只能看到：
- 自己的祖先节点
- 同分支的前序节点
- 不能看到其他分支的节点

```
标准 causal mask:          Tree mask (二叉树):
1 0 0 0 0                  1 0 0 0 0
1 1 0 0 0                  1 1 0 0 0
1 1 1 0 0                  1 0 1 0 0
1 1 1 1 0                  1 1 0 1 0
1 1 1 1 1                  1 0 1 0 1
```

GPU 端通过 `qq_bias` 参数传入 tree_len x tree_len 的 bias 矩阵（0 表示允许，-inf 表示禁止）。

NPU 端通过 `atten_mask` 参数传入（概念兼容）。

## 实现方案

### 第一步：NPU Tree Attention Backend

新建 `vllm_ascend/attention/backends/tree_attn.py`：
- `AscendTreeAttentionMetadata` — 继承 AscendMetadata，添加 `tree_attn_bias`
- `AscendTreeAttentionMetadataBuilder` — 构建 tree mask
- `AscendTreeAttentionImpl` — forward 中将 tree mask 作为 `atten_mask` 传入 `npu_fused_infer_attention_score`

### 第二步：Eagle Proposer Tree 模式

修改 `vllm_ascend/spec_decode/eagle_proposer.py`：
- 添加 `propose_tree` 方法（移植自 GPU 端 eagle.py）
- 添加 tree 层级循环逻辑
- 集成 `AscendTreeAttentionMetadataBuilder`
- 处理 PCP/DCP 兼容

### 第三步：Attention Backend 注册

修改 `vllm_ascend/attention/attention_v1.py`：
- 注册 `AscendTreeAttentionBackend` 到 backend 枚举

### 第四步：ACL Graph 兼容

确保 tree attention 的动态 shape 不破坏 ACL Graph capture：
- tree_len 可能每轮不同
- 需要用 max_tree_len 做 padding 或动态 shape 支持

### 第五步：测试

- 单元测试：tree mask 构建正确性
- E2E 测试：Eagle speculative decoding with tree 的接受率和吞吐量

## 验证前置条件

**必须在认领前验证**（见 `test_tree_attn_npu.py`）：
1. `npu_fused_infer_attention_score` 能否接受任意形状的 `atten_mask`
2. -inf/0 值是否被正确处理
3. 动态 tree mask（batch 推理）是否支持
4. 和 paged attention（block_table）是否兼容

## 风险与备选

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| kernel 不支持自定义 atten_mask | 无法实现 | 切换到 gptq 参数 |
| kernel 支持但性能差 | 功能可行但性能指标低分 | 用预计算 mask + sparse_mode 优化 |
| PCP/DCP 不兼容 | 多卡场景无法使用 | 先支持单卡，多卡作为后续 |
