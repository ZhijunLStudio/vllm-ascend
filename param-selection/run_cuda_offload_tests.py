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
    import torch.nn as nn
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Linear(64, 128, bias=False).cuda()
            self.down_proj = nn.Linear(64, 64, bias=False).cuda()

        def forward(self, x):
            return self.down_proj(self.gate_up_proj(x))

    layers = nn.ModuleList([SimpleMLP() for _ in range(32)])

    offloader = PrefetchOffloader(group_size=8, num_in_group=2, prefetch_step=1)
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


TEST_PROMPTS = [
    "The theory of relativity was proposed by Albert Einstein. It fundamentally changed our understanding of",
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


def run_functional_tests(model_path):
    """运行功能测试: 验证 offload 输出与 baseline 一致"""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("功能测试 (E2E)")
    print("=" * 60)

    sampling_params = SamplingParams(max_tokens=30, temperature=0.0)
    results = {}

    # --- Test 1: Baseline ---
    print("\n--- Test 1: Baseline ---")
    llm = LLM(
        model=model_path, max_model_len=512, gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    t0 = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params, use_tqdm=False)
    t = time.time() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    baseline_texts = [o.outputs[0].text for o in outputs]
    del llm
    results["baseline"] = {"time": round(t, 2), "total_tokens": tokens,
                           "tokens_per_sec": round(tokens / t, 1)}
    print(f"  Time: {t:.2f}s, Tokens: {tokens}, Throughput: {tokens/t:.1f} tok/s")

    # --- Test 2: Prefetch (group=8, num=2, step=1) ---
    print("\n--- Test 2: Prefetch (group=8, num=2, step=1) ---")
    llm = LLM(
        model=model_path, max_model_len=512, gpu_memory_utilization=0.9,
        enforce_eager=True, offload_group_size=8, offload_num_in_group=2,
        offload_prefetch_step=1,
    )
    t0 = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params, use_tqdm=False)
    t = time.time() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    offload_texts = [o.outputs[0].text for o in outputs]
    del llm
    results["prefetch_g8n2s1"] = {"time": round(t, 2), "total_tokens": tokens,
                                   "tokens_per_sec": round(tokens / t, 1)}
    print(f"  Time: {t:.2f}s, Tokens: {tokens}, Throughput: {tokens/t:.1f} tok/s")
    match = all(b == o for b, o in zip(baseline_texts, offload_texts))
    print(f"  Output consistency: {'ALL MATCH' if match else 'MISMATCH DETECTED'}")

    # --- Test 3: Prefetch (group=4, num=1, step=2) ---
    print("\n--- Test 3: Prefetch (group=4, num=1, step=2) ---")
    llm = LLM(
        model=model_path, max_model_len=512, gpu_memory_utilization=0.9,
        enforce_eager=True, offload_group_size=4, offload_num_in_group=1,
        offload_prefetch_step=2,
    )
    t0 = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params, use_tqdm=False)
    t = time.time() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    del llm
    results["prefetch_g4n1s2"] = {"time": round(t, 2), "total_tokens": tokens,
                                   "tokens_per_sec": round(tokens / t, 1)}
    print(f"  Time: {t:.2f}s, Tokens: {tokens}, Throughput: {tokens/t:.1f} tok/s")

    # --- Test 4: UVA ---
    print("\n--- Test 4: UVA (cpu_offload_gb=10) ---")
    try:
        llm = LLM(
            model=model_path, max_model_len=512, gpu_memory_utilization=0.9,
            enforce_eager=True, cpu_offload_gb=10,
        )
        t0 = time.time()
        outputs = llm.generate(TEST_PROMPTS[:3], sampling_params, use_tqdm=False)
        t = time.time() - t0
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        del llm
        results["uva_10gb"] = {"time": round(t, 2), "total_tokens": tokens,
                                "tokens_per_sec": round(tokens / t, 1)}
        print(f"  Time: {t:.2f}s, Tokens: {tokens}, Throughput: {tokens/t:.1f} tok/s")
    except Exception as e:
        print(f"  UVA test failed: {e}")
        results["uva_10gb"] = {"error": str(e)}

    return results


# ============================================================
# 第三部分: 性能基准测试
# ============================================================


def run_benchmark(model_path):
    """运行性能基准测试"""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

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
            "model": model_path, "max_model_len": 512,
            "gpu_memory_utilization": 0.9, "enforce_eager": True,
        }
        if cfg["group_size"] > 0:
            kwargs["offload_group_size"] = cfg["group_size"]
            kwargs["offload_num_in_group"] = cfg["num_in_group"]
            kwargs["offload_prefetch_step"] = cfg["prefetch_step"]

        llm = LLM(**kwargs)

        # Warmup
        t0 = time.time()
        _ = llm.generate(["warmup"], SamplingParams(max_tokens=1), use_tqdm=False)
        warmup_time = time.time() - t0

        # Single request latency
        latencies = []
        for prompt in TEST_PROMPTS[:5]:
            t0 = time.time()
            output = llm.generate(
                [prompt], SamplingParams(max_tokens=50, temperature=0.0),
                use_tqdm=False,
            )
            lat = time.time() - t0
            tokens = len(output[0].outputs[0].token_ids)
            latencies.append({"latency_s": round(lat, 3), "tokens": tokens,
                              "tokens_per_sec": round(tokens / lat, 1)})

        # Batch throughput
        t0 = time.time()
        outputs = llm.generate(
            TEST_PROMPTS, SamplingParams(max_tokens=50, temperature=0.0),
            use_tqdm=False,
        )
        batch_time = time.time() - t0
        batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        del llm

        result = {
            "warmup_time_s": round(warmup_time, 2),
            "single_request_latencies": latencies,
            "avg_latency_s": round(sum(l["latency_s"] for l in latencies) / len(latencies), 3),
            "batch_time_s": round(batch_time, 2),
            "batch_total_tokens": batch_tokens,
            "batch_tokens_per_sec": round(batch_tokens / batch_time, 1),
        }
        all_results[cfg["name"]] = result

        print(f"  Warmup: {warmup_time:.2f}s")
        print(f"  Avg single latency: {result['avg_latency_s']:.3f}s")
        print(f"  Batch throughput: {result['batch_tokens_per_sec']} tok/s")

    # Summary table
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print(f"{'Config':<25} {'Warmup(s)':<12} {'AvgLat(s)':<12} {'Batch tok/s':<12} {'vs Base'}")
    print("-" * 73)
    base_tp = all_results.get("baseline", {}).get("batch_tokens_per_sec", 0)
    for name, r in all_results.items():
        ratio = f"{r['batch_tokens_per_sec'] / base_tp:.2f}x" if base_tp > 0 else "N/A"
        print(f"{name:<25} {r['warmup_time_s']:<12} {r['avg_latency_s']:<12} "
              f"{r['batch_tokens_per_sec']:<12} {ratio}")

    return all_results


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="cpu-offload-params CUDA tests")
    parser.add_argument("--model", type=str,
                        default="/data/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-functional", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    if not args.skip_unit:
        results["unit_tests_passed"] = run_unit_tests()

    if not args.skip_functional:
        results["functional"] = run_functional_tests(args.model)

    if not args.skip_benchmark:
        results["benchmark"] = run_benchmark(args.model)

    output_file = "cuda_offload_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
