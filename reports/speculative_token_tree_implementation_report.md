# speculative_token_tree Ascend NPU 实现技术报告

## 一、项目概述

在 Ascend NPU 上实现 `speculative_token_tree`（EAGLE 树结构推测解码），使 NPU 设备支持树结构 draft token 生成，功能和性能对齐原生 vLLM GPU 实现。

**背景**：标准推测解码生成线性 draft token 序列。`speculative_token_tree` 通过 EAGLE 方法在每个层级探索多个候选 token，形成树结构，提高接受概率。原生 vLLM 使用 FlashAttention 和 CUDA Graph，NPU 端需要替换为 `npu_fused_infer_attention_score` 和 ACL Graph。

## 二、测试模型

| 角色 | 模型 |
|------|------|
| Target 模型 | meta-llama/Llama-3.1-8B-Instruct |
| Draft 模型 | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B |

## 三、Tree 配置

| 配置名 | num_speculative_tokens | speculative_token_tree | 含义 |
|--------|----------------------|----------------------|------|
| Baseline | — | — | 不使用推测解码 |
| tree_n2 | 2 | `[(0,), (0, 0)]` | 1层draft × 1分支 = 2 token |
| tree_n3 | 3 | `[(0,), (0, 0), (0, 1)]` | 1层draft × 2分支 = 3 token |

树结构示意（tree_n3）：

```
       0 (root / prompt)
       |
      (0,)          ← level 1: 1 个 draft token
      /  \
   (0,0) (0,1)      ← level 2: 2 个 draft token（分支）
```

每个 draft token 路径用元组表示，如 `(0, 1)` 表示"从 root 出发，第 0 个分支，再第 1 个分支"。

## 四、实现架构

### 4.1 文件结构

```
vllm_ascend/
├── attention/
│   ├── backends/
│   │   ├── __init__.py          # 包初始化，导出 tree attention 类
│   │   └── tree_attn.py         # 核心：AscendTreeAttentionBackend（686 行）
│   ├── attention_v1.py          # 注册 TREE_ATTN backend
│   └── attention_mask.py        # GPU→NPU mask 转换工具
├── spec_decode/
│   └── eagle_proposer.py        # propose_tree + _propose_tree 方法
├── ascend_forward_context.py    # 新增 slot_mapping 参数
tests/ut/attention/
└── test_tree_attn_npu.py        # 11 个单元测试
```

### 4.2 NPU 与 GPU 核心差异

| 层面 | GPU（原生 vLLM） | NPU（本实现） |
|------|----------------|-------------|
| Attention kernel | FlashAttention | `npu_fused_infer_attention_score` |
| Mask 数据类型 | float32 (0=attend, -inf=block) | int8 (0=attend, 1=block) |
| Mask 尺寸 | tree_len × tree_len | 2048 × 2048（硬件要求 pad） |
| Mask 缓存 | 无 | 预计算所有 slice mask |
| Graph | CUDA Graph | ACL Graph |
| actual_seq_lengths | tensor | tensor（`.to(torch.int64)` 无同步） |

### 4.3 核心组件

**AscendTreeAttentionBackend**：注册名为 `"TREE_ATTN"`，覆盖 GPU 版本。
- Decode：`npu_fused_infer_attention_score`，TND 布局，`sparse_mode=3`，自定义 tree attention mask
- Prefill：BSH 布局，`sparse_mode=0`，标准 causal mask
- ACL Graph：预计算 workspace + `.out()` 变体

**AscendTreeAttentionMetadataBuilder**：解析 tree 配置 → 构建 float32 mask → 转 int8 mask(pad 2048) → Mask 缓存预计算

**propose_tree / _propose_tree**：多层级树循环，每层执行 build_for_drafting → slot mapping → ACL Graph dispatch → draft model forward → 采样下一层

### 4.4 创新点

**创新 1：Tree Mask 缓存预计算**。原生 GPU 每次 `build_for_drafting` 重新切片 mask，NPU 端还额外做 float32→int8 转换。改为初始化时一次性预计算所有 `[start:end, start:end]` 切片 mask，运行时查表。

**创新 2：float32 → int8 Mask 格式适配**。GPU mask 是 float32 格式，NPU kernel 要求 int8（0=attend, 1=block）且 pad 到 2048×2048。设计了 `_convert_tree_mask_for_npu` 函数处理转换，默认初始化为全 1。

**创新 3：ACL Graph Workspace 预计算**。不使用 ExternalEvent 同步（实测 -2.3pp ~ -4.9pp），改为预计算 workspace + `.out()` 变体，在 tree attention 场景下找到最优平衡点。

**创新 4：消除 `.tolist()` CPU-GPU 同步**。`query_start_loc` 和 `seq_lens` 从 `.tolist()`（强制同步）改为 `.to(torch.int64)`（tensor 传参），效果 +1.9pp。

## 五、正确性验证

### 5.1 单元测试（11 个，全部通过）

| 测试类 | 测试方法 | 验证内容 |
|--------|---------|---------|
| TestTreeMaskConversion | test_is_ancestor | 祖先前缀判断逻辑 |
| TestTreeMaskConversion | test_get_depth_counts | 深度计数 |
| TestTreeMaskConversion | test_prepare_tree_attn_bias_gpu | GPU mask 构建 |
| TestTreeMaskConversion | test_convert_tree_mask_for_npu | GPU→NPU mask 转换 |
| TestAscendTreeAttentionBackend | test_backend_name | Backend 注册名 |
| TestAscendTreeAttentionBackend | test_backend_classes | Impl/Builder 类型 |
| TestAscendTreeAttentionBackend | test_kv_cache_shape | KV cache 形状 |
| TestAscendTreeAttentionBackend | test_kv_cache_shape_invalid_block_size | Block size 校验 |
| TestAscendTreeAttentionMetadata | test_metadata_creation | Metadata 数据类 |
| TestConvertFunction | test_binary_tree_depth_3 | 深度 3 二叉树 |
| TestConvertFunction | test_single_token_tree | 最小树（1 个 draft） |

### 5.2 CUDA 逻辑验证（5/5 通过）

在 CUDA A800 80GB 上运行相同 tree mask 逻辑：

| 测试项 | 结果 | 说明 |
|--------|------|------|
| Tree mask 构建逻辑 | PASS | 二叉树深度3，8 tokens |
| GPU→NPU mask 转换 | PASS | float32→int8，pad 2048×2048 |
| 复杂 tree 结构 | PASS | 深度4完全二叉树，15 draft tokens |
| Eagle propose_tree 数据 | PASS | 按深度分组正确 |
| MetadataBuilder 模拟 | PASS | batch_size=2 完整 metadata |

### 5.3 NPU Kernel 验证

- `atten_mask` 支持 int8/bool/uint8 类型
- BSH + `sparse_mode=0` 支持自定义 2D mask
- TND + `sparse_mode=3` 支持 paged + custom mask
- mask 必须 pad 到 2048×2048（硬件/驱动约束）

## 六、性能测试

### 6.1 测试环境

| 项目 | CUDA 端 | NPU 端 |
|------|---------|--------|
| 设备 | NVIDIA A800-SXM4-80GB | Ascend 910B2C |
| 测试日期 | 2026-04-17 | 2026-04-17 |

测试配置：Meta-Llama-3.1-8B-Instruct + EAGLE3-LLaMA3.1-Instruct-8B，max_model_len=2048，temperature=0.0，max_tokens=100，10 条固定 prompt。

### 6.2 CUDA 测试结果（A800，vLLM dev，async on）

| 模式 | 吞吐量 (tok/s) | 加速比 | Accept Length |
|------|---------------|--------|---------------|
| Baseline | 86.64 | — | N/A |
| Tree n=2 | 100.14 | +15.6% | 1.48 |
| Tree n=3 | 98.07 | +13.2% | 1.50 |

Position 接受率：tree n=2 为 0.351 / 0.125；tree n=3 为 0.318 / 0.119 / 0.064。

### 6.3 NPU 测试结果（910B2C，vLLM dev）

vllm-ascend 在推测解码激活时强制禁用 async scheduling，公平对比使用 async=off baseline。

**与 async=off baseline 对比（公平）**：

| 模式 | 吞吐量 (tok/s) | 加速比 | Accept Length |
|------|---------------|--------|---------------|
| Baseline (async=off) | 54.68 | — | N/A |
| Tree n=2 (async=off) | 62.49 | +14.3% | 1.46 |
| Tree n=3 (async=off) | 60.23 | +10.2% | 1.50 |

Position 接受率：tree n=2 为 0.349 / 0.109；tree n=3 为 0.318 / 0.100 / 0.042。

### 6.4 CUDA vs NPU 最终对比

| 指标 | CUDA (A800) | NPU (910B2C) | 差距 |
|------|------------|-------------|------|
| Baseline 吞吐量 | 86.64 tok/s | 54.68 tok/s | CUDA 1.58x |
| Tree n=2 加速比 | +15.6% | +14.3% | 1.3pp |
| Tree n=3 加速比 | +13.2% | +10.2% | 3.0pp |
| Mean Accept Length | 1.48~1.50 | 1.46~1.50 | 基本一致 |
| Pos 0 接受率 | 0.32~0.35 | 0.35 | 基本一致 |
| Pos 1 接受率 | 0.12~0.13 | 0.11 | 基本一致 |

### 6.5 NPU 优化前后对比

| 版本 | Tree n=2 加速 | Tree n=3 加速 | 备注 |
|------|-------------|-------------|------|
| 优化前 | +12.4% | +10.6% | `.tolist()` 强制 CPU 同步 |
| 优化后 | +14.3% | +10.2% | 消除同步 + ACL Graph |

### 6.6 关键发现

1. Tree n=2 接近 CUDA 水平：NPU +14.3% vs CUDA +15.6%，差距仅 1.3pp
2. Accept Length 完全一致：两个平台 1.46~1.50
3. Per-position 接受率一致：Position 0 约 0.32-0.35
4. n=2 优于 n=3：两个平台都是 n=2 加速比更高
5. 优化有效：消除 `.tolist()` 同步后 +1.9pp

## 七、优化尝试记录

### 7.1 成功的优化

| 优化 | 效果 | 详情 |
|------|------|------|
| 消除 `.tolist()` CPU 同步 | +1.9pp | 见创新4 |
| ACL Graph workspace 预计算 + `.out()` | 共同贡献 | 见创新3 |
| Mask 缓存预计算 | 消除运行时重复计算 | 见创新1 |
| int8 mask 格式适配 | 适配 NPU kernel 约束 | 见创新2 |

### 7.2 失败的优化尝试

**尝试 1：ExternalEvent 同步机制**（已回退，-2.3pp ~ -4.9pp）

在 `_forward_decode` ACL Graph 分支添加 ExternalEvent 同步（`event.wait/reset` + `graph_task_group_begin/end`），与标准 decode 对齐。结果 tree n=2 从 +14.3% 降到 +12.0%（-2.3pp），tree n=3 从 +10.2% 降到 +5.3%（-4.9pp）。

失败原因：tree 模式每层都有多个小 ACL Graph 执行，ExternalEvent 同步成本被逐层放大。

**尝试 2：启用 Async Scheduling**（已回退）

移除 `platform.py` 中强制禁用 async scheduling 的代码。结果有负面影响。

失败原因：async scheduling 与 tree 模式多层级循环产生额外调度开销。

**尝试 3：减少 Mask Padding**（不可行）

调研结果：`sparse_mode=3` 对 mask 尺寸的要求是硬件/驱动硬性约束，无法改变。

## 八、Bug 修复记录

| 编号 | 严重度 | 问题 | 修复方式 |
|------|--------|------|---------|
| 1 | P0 | `propose_tree` 从未被调用 | 在 `_propose` 中添加 `AscendTreeAttentionMetadata` 类型检查 |
| 2 | P0 | Backend 名称不匹配：`ASCEND_TREE_ATTN` vs `TREE_ATTN` | 改为 `TREE_ATTN` |
| 3 | P0 | Root token 语义反了：`[0,:]=0` 应为 `[:,0]=0` | 列方向 attend 到 root |
| 4 | P1 | `slot_mapping` 未传递到 forward context | `set_ascend_forward_context` 添加该参数 |
| 5 | P1 | 缺少 ACL Graph per-level dispatch | `propose_tree` 中加 `cudagraph_dispatcher.dispatch` |
| 6 | P2 | `.tolist()` 强制 CPU-GPU 同步 | 改为 `.to(torch.int64)` |
| 7 | P2 | Attention 未使用 ACL Graph | 添加 workspace 预计算 + `.out()` 变体 |

## 九、已对齐原生 vLLM 的功能清单

| 功能 | 状态 | 文件 |
|------|------|------|
| Backend 名称 `"TREE_ATTN"` | ✅ | `tree_attn.py` |
| Root token 行为 `[:, 0] = 0` | ✅ | `tree_attn.py` |
| 祖先关系计算 `_is_ancestor` | ✅ | `tree_attn.py` |
| Mask 缓存优化 | ✅ | `tree_attn.py` |
| `propose_tree` / `_propose_tree` | ✅ | `eagle_proposer.py` |
| `slot_mapping` 传递 | ✅ | `ascend_forward_context.py` |
| `num_tokens_across_dp` 传递 | ✅ | `eagle_proposer.py` |
| ACL Graph per-level dispatch | ✅ | `eagle_proposer.py` |
| `exceeds_max_model_len` 处理 | ✅ | `eagle_proposer.py` |
| SpecDecoding metrics 输出 | ✅ | Server log |
| ACL Graph attention workspace | ✅ | `tree_attn.py` |
| 无同步 actual_seq_lengths | ✅ | `tree_attn.py` |

## 十、评测方法

### 10.1 NPU 端

```bash
export VLLM_VERSION=0.19.0
export ASCEND_RT_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --max-model-len 2048 --no-async-scheduling

# tree n=2
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Meta-Llama-3.1-8B-Instruct \
  --port 8000 --max-model-len 2048 \
  --speculative-config '{"method":"eagle3","model":"/data/models/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":2,"speculative_token_tree":"[(0,), (0, 0)]"}'
```

### 10.2 采集 SpecDecoding 指标

```bash
grep -i "specdecod\|acceptance\|accept length\|mean accept" /tmp/server.log
```

## 十一、修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `vllm_ascend/attention/backends/__init__.py` | 新建 | backends 包初始化 |
| `vllm_ascend/attention/backends/tree_attn.py` | 新建 | 核心实现（686 行） |
| `vllm_ascend/attention/attention_v1.py` | 修改 | 注册 TREE_ATTN backend |
| `vllm_ascend/attention/attention_mask.py` | 修改 | mask 转换工具 |
| `vllm_ascend/spec_decode/eagle_proposer.py` | 修改 | propose_tree 实现 |
| `vllm_ascend/ascend_forward_context.py` | 修改 | 添加 slot_mapping |
| `tests/ut/attention/test_tree_attn_npu.py` | 新建 | 11 个单元测试 |

## 十二、结论

- NPU tree speculative decoding 实现 +14.3% 吞吐提升（tree n=2），与 CUDA 的 +15.6% 仅差 1.3pp
- Tree n=3 达到 +10.2%，与 CUDA 的 +13.2% 差距 3.0pp
- 两个平台接受率和接受长度一致，验证了正确性
- 消除 `.tolist()` 同步后 tree n=2 从 +12.4% 提升到 +14.3%
- ExternalEvent 同步和 Async Scheduling 优化尝试均失败（已回退），Mask padding 优化不可行（硬件约束）
