# cpu-offload-params NPU 适配复现记录

## 一、参数说明

`--cpu-offload-params` 是 vLLM V2 架构引入的模型参数（权重）CPU 卸载功能，通过 `OffloadConfig` 配置，支持两种后端：

- **UVA** (Unified Virtual Addressing)：零拷贝访问 CPU pinned memory（CUDA 特有）
- **Prefetch**：分组异步预取，通过 copy stream 隐藏 H2D 传输延迟

### 子参数

| 子参数 | 类型 | 含义 |
|---|---|---|
| `--offload-backend` | str | "auto" / "uva" / "prefetch" |
| `--prefetch.offload-group-size` | int | 每组包含的模型层数 |
| `--prefetch.offload-num-in-group` | int | 每组中需要卸载的层数 |
| `--prefetch.offload-prefetch-step` | int | 预取步长（提前加载几层） |

---

## 二、NPU 适配方案

### 2.1 核心差异

| 层面 | GPU (上游 vLLM) | NPU (本实现) |
|---|---|---|
| Stream | `torch.cuda.Stream` | `torch.npu.Stream` |
| Event | `torch.cuda.Event` | `torch.npu.Event` |
| 流上下文 | `torch.cuda.stream(s)` | `torch.npu.stream(s)` |
| 流捕获检测 | `torch.cuda.is_current_stream_capturing()` | `torch.npu.is_current_stream_capturing()` |
| UVA | 支持 | **不支持**，自动降级为 NoopOffloader |
| 图捕获 | CUDA Graph | ACL Graph (`torch.npu.NPUGraph`) |

### 2.2 文件结构

```
vllm_ascend/
├── offloader/
│   ├── __init__.py              # 包初始化
│   └── npu_prefetch.py          # NPUPrefetchOffloader + _NPUModuleOffloader
├── patch/worker/
│   ├── __init__.py              # 添加 patch_offloader 导入
│   └── patch_offloader.py       # create_offloader 工厂 patch
├── compilation/
│   └── acl_graph.py             # 添加 offloader sync/join 集成
├── platform.py                  # 添加 is_uva_available() -> False
└── worker/
    └── model_runner_v1.py       # 移除 UVA 断言
```

### 2.3 复用的上游组件（无 CUDA 依赖）

| 组件 | 说明 |
|---|---|
| `ParamInfo` | 参数元数据数据类 |
| `StaticBufferPool` | 预分配 GPU buffer pool（纯 tensor 操作） |
| `CpuParamOffloader` | 将参数拷贝到 CPU pinned memory |
| `BaseOffloader` | 抽象基类，定义 wrap_modules/sync/join 接口 |
| `vllm::wait_prefetch` / `vllm::start_prefetch` | 自定义 op，自动调度到 NPUPrefetchOffloader |

### 2.4 ACL Graph 集成

在 `ACLGraphWrapper.__call__` 中添加 3 个集成点：

1. **捕获前**: `get_offloader().sync_prev_onload()` — 确保预取完成
2. **捕获内 forward 后**: `get_offloader().join_after_forward()` — 避免 unjoined stream
3. **回放前**: `get_offloader().sync_prev_onload()` — 确保异步操作完成

---

## 三、与现有 weight_prefetch.py 的关系

两者**互补共存**：

| 维度 | `weight_prefetch.py`（已有） | `NPUPrefetchOffloader`（新增） |
|---|---|---|
| 粒度 | 线性层级别（细粒度） | Decoder 层级别（粗粒度） |
| 机制 | CANN L2 cache prefetch hints | CPU→NPU HBM 异步拷贝 |
| 数据源 | GPU HBM → 片上 L2 | CPU pinned memory → GPU HBM |
| 内存影响 | 不额外占用 HBM | 需要 static buffer pool |

---

## 四、Bug 修复记录

（当前无 bug，随测试进行更新）

---

## 五、测试方案

### 5.1 单元测试

```
tests/ut/offloader/test_npu_prefetch_offloader.py
```

测试用例：
1. `test_creates_npu_stream` — 验证 Stream 类型为 torch.npu.Stream
2. `test_layer_selection_group_4_num_1` — 层选择逻辑
3. `test_layer_selection_group_2_num_1` — 层选择逻辑
4. `test_layer_selection_group_4_num_2` — 层选择逻辑
5. `test_offload_params_whitelist` — 参数白名单过滤
6. `test_returns_all_modules` — wrap_modules 返回所有模块
7. `test_wrap_modules_called_twice_raises` — 重复调用断言
8. `test_sync_prev_onload_no_modules` — 空模块同步
9. `test_prefetch_returns_npu_offloader` — 工厂 patch
10. `test_uva_returns_noop` — UVA 降级
11. `test_allocate_on_npu` — Buffer pool NPU 分配
12. `test_creates_npu_event` — NPU Event 类型验证
13. `test_cpu_param_offloader_creates_cpu_storage` — CPU 存储创建

### 5.2 E2E 测试

```
tests/e2e/singlecard/test_cpu_offload_params.py
```

测试配置：
```bash
vllm serve Qwen/Qwen3-0.6B \
  --offload-backend prefetch \
  --prefetch.offload-group-size 4 \
  --prefetch.offload-num-in-group 1 \
  --prefetch.offload-prefetch-step 1
```

验证项：
1. 模型正常加载
2. 生成结果正确性（与 baseline 对比）
3. 无 CUDA 相关错误
4. ACL Graph 正常捕获/回放

---

## 六、修改文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `vllm_ascend/offloader/__init__.py` | 新建 | 包初始化 |
| `vllm_ascend/offloader/npu_prefetch.py` | 新建 | 核心实现 (~270行) |
| `vllm_ascend/patch/worker/patch_offloader.py` | 新建 | 工厂 patch (~70行) |
| `vllm_ascend/patch/worker/__init__.py` | 修改 | 添加 patch_offloader 导入 |
| `vllm_ascend/platform.py` | 修改 | 添加 is_uva_available() |
| `vllm_ascend/compilation/acl_graph.py` | 修改 | 添加 offloader 集成 (~10行) |
| `vllm_ascend/worker/model_runner_v1.py` | 修改 | 移除 UVA 断言 |
| `vllm_ascend/_310p/model_runner_310p.py` | 修改 | 移除 UVA 断言 |
| `tests/ut/offloader/test_npu_prefetch_offloader.py` | 新建 | 单元测试 |
| `tests/e2e/singlecard/test_cpu_offload_params.py` | 新建 | E2E 测试 |
