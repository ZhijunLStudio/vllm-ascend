# cpu-offload-params CUDA 端测试指南

## 一、测试目标

在 CUDA GPU 上使用与 NPU 相同的模型和配置，验证 `--cpu-offload-params` prefetch 后端的功能正确性和性能基线，作为 NPU 实现的对比参照。

## 二、测试环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA A800-SXM4-80GB 或同级（需 >=40GB 显存跑 8B 模型 offload） |
| Python | 3.11+ |
| vLLM | dev 版本（与 NPU 端同版本 commit） |
| 模型 | Meta-Llama-3.1-8B-Instruct |

## 三、环境准备

```bash
# 1. 切到正确的 vllm 分支（与 NPU 端同版本）
cd /path/to/vllm
git log --oneline -1  # 确认 commit 与 NPU 端一致

# 2. 安装 vllm（如果未安装）
pip install -e .

# 3. 模型路径（按实际情况修改）
MODEL_PATH="/path/to/Meta-Llama-3.1-8B-Instruct"

# 4. 设置 GPU
export CUDA_VISIBLE_DEVICES=0
```

## 四、测试脚本

保存为 `run_cuda_offload_tests.py`，在 CUDA 机器上运行：

```bash
python run_cuda_offload_tests.py 2>&1 | tee cuda_offload_results.log
```

---

## 五、测试脚本内容

以下脚本包含完整的单元测试、功能测试和性能基准测试。

```python
#!/usr/bin/env python3
"""
cpu-offload-params CUDA 端测试脚本

使用方法:
    python run_cuda_offload_tests.py --model /path/to/Meta-Llama-3.1-8B-Instruct

测试项:
    1. 单元测试: factory patch、layer selection、static buffer pool
    2. 功能测试: 基线 vs offload 输出一致性
    3. 性能测试: 不同 offload 配置的吞吐量对比
"""

import argparse
import json
import os
import sys
import time

import torch

# ============================================================
# 第一部分: 单元测试
# ============================================================


def test_is_uva_available():
    """验证 CUDA 上 is_uva_available 返回 True"""
    from vllm.utils.platform_utils import is_uva_available
    assert is_uva_available() is True, "CUDA should support UVA"
    print("[PASS] test_is_uva_available")


def test_create_offloader_prefetch():
    """prefetch 后端应返回 PrefetchOffloader"""
    from vllm.config import OffloadConfig
    from vllm.model_executor.offloader import create_offloader
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader

    config = OffloadConfig(
        offload_backend="prefetch",
        prefetch={
            "offload_group_size": 8,
            "offload_num_in_group": 2,
            "offload_prefetch_step": 1,
        },
    )
    offloader = create_offloader(config)
    assert isinstance(offloader, PrefetchOffloader), (
        f"Expected PrefetchOffloader, got {type(offloader).__name__}"
    )
    print("[PASS] test_create_offloader_prefetch")


def test_create_offloader_uva():
    """UVA 后端应返回 UVAOffloader"""
    from vllm.config import OffloadConfig
    from vllm.model_executor.offloader import create_offloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    config = OffloadConfig(
        offload_backend="uva",
        uva={"cpu_offload_gb": 10},
    )
    offloader = create_offloader(config)
    assert isinstance(offloader, UVAOffloader), (
        f"Expected UVAOffloader, got {type(offloader).__name__}"
    )
    print("[PASS] test_create_offloader_uva")


def test_create_offloader_noop():
    """空配置应返回 NoopOffloader"""
    from vllm.config import OffloadConfig
    from vllm.model_executor.offloader import create_offloader
    from vllm.model_executor.offloader.base import NoopOffloader

    config = OffloadConfig()
    offloader = create_offloader(config)
    assert isinstance(offloader, NoopOffloader), (
        f"Expected NoopOffloader, got {type(offloader).__name__}"
    )
    print("[PASS] test_create_offloader_noop")


def test_static_buffer_pool_cuda():
    """StaticBufferPool 在 CUDA 上分配"""
    from vllm.model_executor.offloader.prefetch import ParamInfo, StaticBufferPool

    param_infos = [
        ParamInfo(name="weight", shape=(128, 64), stride=(64, 1), dtype=torch.float16),
        ParamInfo(name="bias", shape=(64,), stride=(1,), dtype=torch.float16),
    ]
    device = torch.device("cuda:0")
    pool = StaticBufferPool(
        param_infos=param_infos,
        slot_capacity=2,
        device=device,
    )
    buf = pool.get_buffer("weight", (128, 64), (64, 1), torch.float16, slot_idx=0)
    assert buf.device.type == "cuda", f"Expected cuda, got {buf.device}"
    assert buf.shape == (128, 64)
    assert buf.dtype == torch.float16
    print("[PASS] test_static_buffer_pool_cuda")


def test_layer_selection():
    """测试 layer selection 逻辑"""
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    import torch.nn as nn

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Linear(64, 128, bias=False).cuda()
            self.down_proj = nn.Linear(64, 64, bias=False).cuda()

        def forward(self, x):
            return self.down_proj(self.gate_up_proj(x))

    layers = nn.ModuleList([SimpleMLP() for _ in range(32)])

    # group_size=8, num_in_group=2: layers 6,7, 14,15, 22,23, 30,31 offloaded
    offloader = PrefetchOffloader(
        group_size=8, num_in_group=2, prefetch_step=1
    )
    offloader.wrap_modules(iter(list(layers)))
    assert len(offloader.module_offloaders) == 8, (
        f"Expected 8 offloaded modules, got {len(offloader.module_offloaders)}"
    )
    print("[PASS] test_layer_selection")


def run_unit_tests():
    """运行所有单元测试"""
    print("\n" + "=" * 60)
    print("单元测试")
    print("=" * 60)

    tests = [
        test_is_uva_available,
        test_create_offloader_prefetch,
        test_create_offloader_uva,
        test_create_offloader_noop,
        test_static_buffer_pool_cuda,
        test_layer_selection,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n单元测试结果: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# 第二部分: 功能测试 (E2E)
# ============================================================


def run_functional_tests(model_path):
    """运行功能测试: 验证 offload 输出与 baseline 一致"""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("功能测试 (E2E)")
    print("=" * 60)

    prompts = [
        "The theory of relativity was proposed by Albert Einstein. "
        "It fundamentally changed our understanding of",
        "Hello, how are you today? Please tell me a joke.",
        "Explain quantum computing in simple terms.",
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is",
        "Write a short story about a robot learning to paint.",
        "Machine learning is a subset of artificial intelligence that focuses on",
        "The solar system consists of eight planets orbiting around the Sun. "
        "The innermost planet is",
        "Python is a popular programming language because",
        "The history of the internet dates back to the 1960s when",
        "Climate change is one of the most pressing issues facing humanity today. "
        "The primary cause is",
    ]
    sampling_params = SamplingParams(max_tokens=30, temperature=0.0)

    results = {}

    # --- Test 1: Baseline (no offloading) ---
    print("\n--- Test 1: Baseline ---")
    llm = LLM(
        model=model_path,
        max_model_len=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    baseline_time = time.time() - t0
    baseline_texts = [o.outputs[0].text for o in outputs]
    baseline_tokens = sum(o.outputs[0].completion_token_ids.__len__() for o in outputs)
    del llm
    results["baseline"] = {
        "time": baseline_time,
        "total_tokens": baseline_tokens,
        "tokens_per_sec": baseline_tokens / baseline_time,
    }
    print(f"  Time: {baseline_time:.2f}s, Tokens: {baseline_tokens}, "
          f"Throughput: {baseline_tokens / baseline_time:.1f} tokens/s")

    # --- Test 2: Prefetch offload (group=8, num=2, step=1) ---
    print("\n--- Test 2: Prefetch (group=8, num=2, step=1) ---")
    llm = LLM(
        model=model_path,
        max_model_len=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        offload_group_size=8,
        offload_num_in_group=2,
        offload_prefetch_step=1,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    offload_time = time.time() - t0
    offload_texts = [o.outputs[0].text for o in outputs]
    offload_tokens = sum(o.outputs[0].completion_token_ids.__len__() for o in outputs)
    del llm
    results["prefetch_g8n2s1"] = {
        "time": offload_time,
        "total_tokens": offload_tokens,
        "tokens_per_sec": offload_tokens / offload_time,
    }
    print(f"  Time: {offload_time:.2f}s, Tokens: {offload_tokens}, "
          f"Throughput: {offload_tokens / offload_time:.1f} tokens/s")

    # 检查输出一致性
    all_match = True
    for i, (base, off) in enumerate(zip(baseline_texts, offload_texts)):
        if base != off:
            print(f"  [MISMATCH] Prompt {i}:")
            print(f"    Baseline: {base[:60]}...")
            print(f"    Offload:  {off[:60]}...")
            all_match = False
    if all_match:
        print("  Output consistency: ALL MATCH")
    else:
        print("  Output consistency: MISMATCH DETECTED")

    # --- Test 3: Prefetch offload (group=4, num=1, step=2) ---
    print("\n--- Test 3: Prefetch (group=4, num=1, step=2) ---")
    llm = LLM(
        model=model_path,
        max_model_len=512,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=2,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    offload_time3 = time.time() - t0
    offload_tokens3 = sum(o.outputs[0].completion_token_ids.__len__() for o in outputs)
    del llm
    results["prefetch_g4n1s2"] = {
        "time": offload_time3,
        "total_tokens": offload_tokens3,
        "tokens_per_sec": offload_tokens3 / offload_time3,
    }
    print(f"  Time: {offload_time3:.2f}s, Tokens: {offload_tokens3}, "
          f"Throughput: {offload_tokens3 / offload_time3:.1f} tokens/s")

    # --- Test 4: UVA offload ---
    print("\n--- Test 4: UVA (cpu_offload_gb=10) ---")
    try:
        llm = LLM(
            model=model_path,
            max_model_len=512,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            cpu_offload_gb=10,
        )
        t0 = time.time()
        outputs = llm.generate(prompts[:3], sampling_params, use_tqdm=False)
        uva_time = time.time() - t0
        uva_tokens = sum(o.outputs[0].completion_token_ids.__len__() for o in outputs)
        del llm
        results["uva_10gb"] = {
            "time": uva_time,
            "total_tokens": uva_tokens,
            "tokens_per_sec": uva_tokens / uva_time,
        }
        print(f"  Time: {uva_time:.2f}s, Tokens: {uva_tokens}, "
              f"Throughput: {uva_tokens / uva_time:.1f} tokens/s")
    except Exception as e:
        print(f"  UVA test failed: {e}")
        results["uva_10gb"] = {"error": str(e)}

    return results


# ============================================================
# 第三部分: 性能基准测试
# ============================================================


def run_benchmark(model_path):
    """运行性能基准测试: 单请求延迟 + 批量吞吐"""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    test_prompts = [
        "The theory of relativity was proposed by Albert Einstein. It fundamentally changed our understanding of space, time, and gravity. The key ideas include",
        "Hello, how are you today? Please tell me a joke.",
        "Explain quantum computing in simple terms.",
        "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
        "Write a short story about a robot learning to paint.",
        "Machine learning is a subset of artificial intelligence that focuses on",
        "The solar system consists of eight planets orbiting around the Sun. The innermost planet is",
        "Python is a popular programming language because",
        "The history of the internet dates back to the 1960s when",
        "Climate change is one of the most pressing issues facing humanity today. The primary cause is",
    ]

    configs = [
        {"name": "baseline", "group_size": 0, "num_in_group": 0, "prefetch_step": 0},
        {"name": "prefetch_g8n2s1", "group_size": 8, "num_in_group": 2, "prefetch_step": 1},
        {"name": "prefetch_g8n2s2", "group_size": 8, "num_in_group": 2, "prefetch_step": 2},
        {"name": "prefetch_g4n1s1", "group_size": 4, "num_in_group": 1, "prefetch_step": 1},
        {"name": "prefetch_g4n1s2", "group_size": 4, "num_in_group": 1, "prefetch_step": 2},
    ]

    all_results = {}

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        kwargs = {
            "model": model_path,
            "max_model_len": 512,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
        }
        if cfg["group_size"] > 0:
            kwargs["offload_group_size"] = cfg["group_size"]
            kwargs["offload_num_in_group"] = cfg["num_in_group"]
            kwargs["offload_prefetch_step"] = cfg["prefetch_step"]

        llm = LLM(**kwargs)

        # 测量加载时间（通过预热）
        t0 = time.time()
        _ = llm.generate(["warmup"], SamplingParams(max_tokens=1), use_tqdm=False)
        warmup_time = time.time() - t0

        # 单请求延迟测试
        latencies = []
        for prompt in test_prompts[:5]:
            t0 = time.time()
            output = llm.generate(
                [prompt],
                SamplingParams(max_tokens=50, temperature=0.0),
                use_tqdm=False,
            )
            lat = time.time() - t0
            tokens = output[0].outputs[0].completion_token_ids.__len__()
            latencies.append({
                "latency_s": round(lat, 3),
                "tokens": tokens,
                "tokens_per_sec": round(tokens / lat, 1),
            })

        # 批量吞吐测试
        t0 = time.time()
        outputs = llm.generate(
            test_prompts,
            SamplingParams(max_tokens=50, temperature=0.0),
            use_tqdm=False,
        )
        batch_time = time.time() - t0
        batch_tokens = sum(
            o.outputs[0].completion_token_ids.__len__() for o in outputs
        )

        del llm

        result = {
            "warmup_time_s": round(warmup_time, 2),
            "single_request_latencies": latencies,
            "avg_latency_s": round(
                sum(l["latency_s"] for l in latencies) / len(latencies), 3
            ),
            "batch_time_s": round(batch_time, 2),
            "batch_total_tokens": batch_tokens,
            "batch_tokens_per_sec": round(batch_tokens / batch_time, 1),
            "batch_prompts": len(test_prompts),
        }
        all_results[cfg["name"]] = result

        print(f"  Warmup: {warmup_time:.2f}s")
        print(f"  Avg single latency: {result['avg_latency_s']:.3f}s")
        print(f"  Batch throughput: {result['batch_tokens_per_sec']} tokens/s "
              f"({batch_tokens} tokens in {batch_time:.2f}s)")

    # 打印对比表
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print(f"{'Config':<25} {'Warmup(s)':<12} {'Avg Lat(s)':<12} {'Batch tok/s':<12} {'vs Baseline'}")
    print("-" * 73)
    baseline_throughput = all_results.get("baseline", {}).get("batch_tokens_per_sec", 0)
    for name, r in all_results.items():
        speedup = ""
        if baseline_throughput > 0 and name != "baseline":
            ratio = r["batch_tokens_per_sec"] / baseline_throughput
            speedup = f"{ratio:.2f}x"
        elif name == "baseline":
            speedup = "1.00x"
        print(f"{name:<25} {r['warmup_time_s']:<12} {r['avg_latency_s']:<12} "
              f"{r['batch_tokens_per_sec']:<12} {speedup}")

    return all_results


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="cpu-offload-params CUDA tests")
    parser.add_argument(
        "--model",
        type=str,
        default="/data/models/Meta-Llama-3.1-8B-Instruct",
        help="Path to the model",
    )
    parser.add_argument(
        "--skip-unit", action="store_true", help="Skip unit tests"
    )
    parser.add_argument(
        "--skip-functional", action="store_true", help="Skip functional tests"
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip benchmark tests"
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    results = {}

    # 1. 单元测试
    if not args.skip_unit:
        results["unit_tests_passed"] = run_unit_tests()

    # 2. 功能测试
    if not args.skip_functional:
        results["functional"] = run_functional_tests(args.model)

    # 3. 性能基准测试
    if not args.skip_benchmark:
        results["benchmark"] = run_benchmark(args.model)

    # 保存结果
    output_file = "cuda_offload_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
```

---

## 六、运行方法

```bash
# 完整测试（单元 + 功能 + 性能）
python run_cuda_offload_tests.py --model /data/models/Meta-Llama-3.1-8B-Instruct

# 只跑功能测试
python run_cuda_offload_tests.py --model /data/models/Meta-Llama-3.1-8B-Instruct --skip-unit --skip-benchmark

# 只跑性能基准
python run_cuda_offload_tests.py --model /data/models/Meta-Llama-3.1-8B-Instruct --skip-unit --skip-functional
```

## 七、预期输出

### 单元测试

```
[PASS] test_is_uva_available
[PASS] test_create_offloader_prefetch
[PASS] test_create_offloader_uva
[PASS] test_create_offloader_noop
[PASS] test_static_buffer_pool_cuda
[PASS] test_layer_selection

单元测试结果: 6 passed, 0 failed
```

### 功能测试

```
--- Test 1: Baseline ---
  Time: X.XXs, Tokens: XXX, Throughput: XX.X tokens/s

--- Test 2: Prefetch (group=8, num=2, step=1) ---
  Time: X.XXs, Tokens: XXX, Throughput: XX.X tokens/s
  Output consistency: ALL MATCH

--- Test 3: Prefetch (group=4, num=1, step=2) ---
  Time: X.XXs, Tokens: XXX, Throughput: XX.X tokens/s

--- Test 4: UVA (cpu_offload_gb=10) ---
  Time: X.XXs, Tokens: XXX, Throughput: XX.X tokens/s
```

### 性能对比

```
Config                    Warmup(s)    Avg Lat(s)   Batch tok/s  vs Baseline
-------------------------------------------------------------------------
baseline                  X.XX         X.XXX        XXX.X        1.00x
prefetch_g8n2s1           X.XX         X.XXX        XXX.X        0.XXx
prefetch_g8n2s2           X.XX         X.XXX        XXX.X        0.XXx
prefetch_g4n1s1           X.XX         X.XXX        XXX.X        0.XXx
prefetch_g4n1s2           X.XX         X.XXX        XXX.X        0.XXx
```

## 八、采集指标

| 指标 | 说明 | 对比方式 |
|------|------|---------|
| 模型加载时间 | 从 LLM() 到首次推理 | 各配置对比 |
| 单请求延迟 | 5 个 prompt 的平均延迟 | 各配置对比 |
| 批量吞吐量 | 10 个 prompt 串行的 tokens/s | vs baseline |
| 输出一致性 | offload 输出与 baseline 是否完全一致 | temperature=0.0 时应一致 |
| UVA 功能 | CUDA 上 UVA 后端是否正常工作 | CUDA 独有 |
| 内存节省 | offload 后可用 KV cache tokens | 各配置对比（日志中可见） |

## 九、结果文件

测试完成后会生成 `cuda_offload_results.json`，格式：

```json
{
  "unit_tests_passed": true,
  "functional": {
    "baseline": {"time": X, "total_tokens": X, "tokens_per_sec": X},
    "prefetch_g8n2s1": {"time": X, "total_tokens": X, "tokens_per_sec": X},
    "prefetch_g4n1s2": {"time": X, "total_tokens": X, "tokens_per_sec": X},
    "uva_10gb": {"time": X, "total_tokens": X, "tokens_per_sec": X}
  },
  "benchmark": {
    "baseline": {"warmup_time_s": X, "avg_latency_s": X, "batch_tokens_per_sec": X},
    "prefetch_g8n2s1": {"warmup_time_s": X, "avg_latency_s": X, "batch_tokens_per_sec": X},
    ...
  }
}
```

将此 JSON 文件复制回 NPU 环境，与 NPU 测试结果对比即可生成完整的 CUDA vs NPU 对比报告。
