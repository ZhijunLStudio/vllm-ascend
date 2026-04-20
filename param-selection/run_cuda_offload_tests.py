#!/usr/bin/env python3
"""
cpu-offload-params 统一测试脚本 (CUDA + NPU)

使用方法:
    python run_offload_tests.py --model /path/to/Meta-Llama-3.1-8B-Instruct
    python run_offload_tests.py --model /path/to/Meta-Llama-3.1-8B-Instruct --mode eager
    python run_offload_tests.py --model /path/to/Meta-Llama-3.1-8B-Instruct --mode graph
    python run_offload_tests.py --model /path/to/Meta-Llama-3.1-8B-Instruct --mode both

测试项:
    1. 单元测试: factory patch、layer selection、static buffer pool
    2. 功能测试: 基线 vs offload 输出一致性 (eager + graph)
    3. 性能测试: 不同 offload 配置的吞吐量对比 (eager + graph)
"""

import argparse
import gc
import json
import re
import time
from contextlib import contextmanager
from io import StringIO
from logging import getLogger

import torch

logger = getLogger(__name__)

# ============================================================
# 平台检测
# ============================================================

def get_platform():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.npu.is_available():
            return "npu"
    except Exception:
        pass
    return "unknown"

PLATFORM = get_platform()
GRAPH_MODE_ENABLED = PLATFORM != "unknown"  # both cuda and npu support graphs


# ============================================================
# 单元测试
# ============================================================


def test_is_uva_available():
    from vllm.utils.platform_utils import is_uva_available
    if PLATFORM == "cuda":
        assert is_uva_available() is True
    elif PLATFORM == "npu":
        assert is_uva_available() is False
    print(f"[PASS] test_is_uva_available ({PLATFORM})")


def test_create_offloader_prefetch():
    from vllm.config import OffloadConfig
    from vllm.model_executor.offloader import create_offloader

    config = OffloadConfig(
        offload_backend="prefetch",
        prefetch={"offload_group_size": 8, "offload_num_in_group": 2,
                  "offload_prefetch_step": 1},
    )
    offloader = create_offloader(config)
    if PLATFORM == "npu":
        from vllm_ascend.offloader.npu_prefetch import NPUPrefetchOffloader
        assert isinstance(offloader, NPUPrefetchOffloader)
    else:
        from vllm.model_executor.offloader.prefetch import PrefetchOffloader
        assert isinstance(offloader, PrefetchOffloader)
    print(f"[PASS] test_create_offloader_prefetch ({PLATFORM})")


def test_create_offloader_uva():
    from vllm.config import OffloadConfig
    from vllm.model_executor.offloader import create_offloader

    config = OffloadConfig(
        offload_backend="uva",
        uva={"cpu_offload_gb": 10},
    )
    offloader = create_offloader(config)
    if PLATFORM == "npu":
        from vllm.model_executor.offloader.base import NoopOffloader
        assert isinstance(offloader, NoopOffloader), (
            f"NPU should degrade UVA to Noop, got {type(offloader).__name__}"
        )
    else:
        from vllm.model_executor.offloader.uva import UVAOffloader
        assert isinstance(offloader, UVAOffloader)
    print(f"[PASS] test_create_offloader_uva ({PLATFORM})")


def test_create_offloader_noop():
    from vllm.config import OffloadConfig
    from vllm.model_executor.offloader import create_offloader
    from vllm.model_executor.offloader.base import NoopOffloader

    config = OffloadConfig()
    offloader = create_offloader(config)
    assert isinstance(offloader, NoopOffloader)
    print("[PASS] test_create_offloader_noop")


def test_static_buffer_pool():
    from vllm.model_executor.offloader.prefetch import ParamInfo, StaticBufferPool

    device = torch.device(f"{PLATFORM}:0")
    param_infos = [
        ParamInfo(name="weight", shape=(128, 64), stride=(64, 1), dtype=torch.float16),
        ParamInfo(name="bias", shape=(64,), stride=(1,), dtype=torch.float16),
    ]
    pool = StaticBufferPool(param_infos=param_infos, slot_capacity=2, device=device)
    buf = pool.get_buffer("weight", (128, 64), (64, 1), torch.float16, slot_idx=0)
    assert buf.device.type == PLATFORM
    assert buf.shape == (128, 64)
    print(f"[PASS] test_static_buffer_pool ({PLATFORM})")


def test_layer_selection():
    import torch.nn as nn
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Linear(64, 128, bias=False).to(f"{PLATFORM}:0")
            self.down_proj = nn.Linear(64, 64, bias=False).to(f"{PLATFORM}:0")
        def forward(self, x):
            return self.down_proj(self.gate_up_proj(x))

    layers = nn.ModuleList([SimpleMLP() for _ in range(32)])
    offloader = PrefetchOffloader(group_size=8, num_in_group=2, prefetch_step=1)
    offloader.wrap_modules(iter(list(layers)))
    assert len(offloader.module_offloaders) == 8
    print("[PASS] test_layer_selection")


def run_unit_tests():
    print("\n" + "=" * 60)
    print("单元测试")
    print("=" * 60)

    tests = [
        test_is_uva_available,
        test_create_offloader_prefetch,
        test_create_offloader_uva,
        test_create_offloader_noop,
        test_static_buffer_pool,
        test_layer_selection,
    ]

    passed = failed = 0
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
# 测试配置
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

OFFLOAD_CONFIGS = [
    {"name": "baseline", "group_size": 0, "num_in_group": 0, "prefetch_step": 0},
    {"name": "prefetch_g8n2s1", "group_size": 8, "num_in_group": 2, "prefetch_step": 1},
    {"name": "prefetch_g8n2s2", "group_size": 8, "num_in_group": 2, "prefetch_step": 2},
    {"name": "prefetch_g4n1s1", "group_size": 4, "num_in_group": 1, "prefetch_step": 1},
    {"name": "prefetch_g4n1s2", "group_size": 4, "num_in_group": 1, "prefetch_step": 2},
]


def make_llm_kwargs(model_path, cfg, use_graph):
    kwargs = {
        "model": model_path,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.9,
    }
    if not use_graph:
        kwargs["enforce_eager"] = True
    if cfg["group_size"] > 0:
        kwargs["offload_group_size"] = cfg["group_size"]
        kwargs["offload_num_in_group"] = cfg["num_in_group"]
        kwargs["offload_prefetch_step"] = cfg["prefetch_step"]
    return kwargs


# ============================================================
# 功能测试
# ============================================================


def run_functional_tests(model_path, modes):
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("功能测试 (E2E)")
    print("=" * 60)

    sampling_params = SamplingParams(max_tokens=30, temperature=0.0)
    results = {}

    for mode in modes:
        use_graph = (mode == "graph")
        mode_label = f"_{mode}" if mode != "eager" else ""
        print(f"\n--- Mode: {mode} ---")

        # Baseline
        print(f"\n  Baseline ({mode})")
        gc.collect()
        llm = LLM(**make_llm_kwargs(model_path, OFFLOAD_CONFIGS[0], use_graph))
        t0 = time.time()
        outputs = llm.generate(TEST_PROMPTS, sampling_params, use_tqdm=False)
        t = time.time() - t0
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        baseline_texts = [o.outputs[0].text for o in outputs]
        del llm; gc.collect()
        key = f"baseline{mode_label}"
        results[key] = {"mode": mode, "time": round(t, 2), "total_tokens": tokens,
                        "tokens_per_sec": round(tokens / t, 1)}
        print(f"    Time: {t:.2f}s, Tokens: {tokens}, Throughput: {tokens/t:.1f} tok/s")

        # Offload configs
        for cfg in OFFLOAD_CONFIGS[1:]:
            print(f"\n  {cfg['name']} ({mode})")
            gc.collect()
            llm = LLM(**make_llm_kwargs(model_path, cfg, use_graph))
            t0 = time.time()
            outputs = llm.generate(TEST_PROMPTS, sampling_params, use_tqdm=False)
            t = time.time() - t0
            tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            offload_texts = [o.outputs[0].text for o in outputs]
            del llm; gc.collect()
            key = f"{cfg['name']}{mode_label}"
            results[key] = {"mode": mode, "time": round(t, 2), "total_tokens": tokens,
                            "tokens_per_sec": round(tokens / t, 1)}
            match = all(b == o for b, o in zip(baseline_texts, offload_texts))
            results[key]["output_match"] = match
            print(f"    Time: {t:.2f}s, Tokens: {tokens}, Throughput: {tokens/t:.1f} tok/s, Match: {match}")

    return results


# ============================================================
# 性能基准测试
# ============================================================


def run_benchmark(model_path, modes):
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    all_results = {}

    for mode in modes:
        use_graph = (mode == "graph")
        mode_suffix = f"_{mode}" if mode != "eager" else ""
        print(f"\n=== Mode: {mode} ===")

        for cfg in OFFLOAD_CONFIGS:
            name = f"{cfg['name']}{mode_suffix}"
            print(f"\n--- {name} ---")
            gc.collect()

            kwargs = make_llm_kwargs(model_path, cfg, use_graph)
            llm = LLM(**kwargs)

            # Warmup
            t0 = time.time()
            _ = llm.generate(["warmup"], SamplingParams(max_tokens=1), use_tqdm=False)
            warmup_time = time.time() - t0

            # Single request latency (5 prompts)
            latencies = []
            for prompt in TEST_PROMPTS[:5]:
                t0 = time.time()
                output = llm.generate(
                    [prompt], SamplingParams(max_tokens=50, temperature=0.0),
                    use_tqdm=False,
                )
                lat = time.time() - t0
                tokens = len(output[0].outputs[0].token_ids)
                latencies.append({
                    "latency_s": round(lat, 3),
                    "tokens": tokens,
                    "tokens_per_sec": round(tokens / lat, 1),
                })

            # Batch throughput
            t0 = time.time()
            outputs = llm.generate(
                TEST_PROMPTS, SamplingParams(max_tokens=50, temperature=0.0),
                use_tqdm=False,
            )
            batch_time = time.time() - t0
            batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

            del llm; gc.collect()

            result = {
                "mode": mode,
                "warmup_time_s": round(warmup_time, 2),
                "single_request_latencies": latencies,
                "avg_latency_s": round(
                    sum(l["latency_s"] for l in latencies) / len(latencies), 3),
                "batch_time_s": round(batch_time, 2),
                "batch_total_tokens": batch_tokens,
                "batch_tokens_per_sec": round(batch_tokens / batch_time, 1),
            }
            all_results[name] = result
            print(f"  Warmup: {warmup_time:.2f}s, Avg lat: {result['avg_latency_s']:.3f}s, "
                  f"Batch: {result['batch_tokens_per_sec']} tok/s")

    # Summary
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    print(f"{'Config':<30} {'Mode':<8} {'Warmup':<10} {'AvgLat':<10} {'Batch tok/s':<12} {'vs Base'}")
    print("-" * 80)

    for mode in modes:
        mode_suffix = f"_{mode}" if mode != "eager" else ""
        base_key = f"baseline{mode_suffix}"
        base_tp = all_results.get(base_key, {}).get("batch_tokens_per_sec", 0)
        for cfg in OFFLOAD_CONFIGS:
            name = f"{cfg['name']}{mode_suffix}"
            r = all_results.get(name)
            if not r:
                continue
            ratio = f"{r['batch_tokens_per_sec'] / base_tp:.2f}x" if base_tp > 0 and name != base_key else "1.00x"
            print(f"{cfg['name']:<30} {mode:<8} {r['warmup_time_s']:<10} {r['avg_latency_s']:<10} "
                  f"{r['batch_tokens_per_sec']:<12} {ratio}")

    return all_results


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="cpu-offload-params unified tests")
    parser.add_argument("--model", type=str,
                        default="/data/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-functional", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    print(f"Platform: {PLATFORM}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")

    if PLATFORM == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif PLATFORM == "npu":
        print(f"NPU: {torch.npu.get_device_name(0)}")

    modes = ["eager", "graph"] if args.mode == "both" else [args.mode]

    results = {"platform": PLATFORM, "modes": modes}

    if not args.skip_unit:
        results["unit_tests_passed"] = run_unit_tests()

    if not args.skip_functional:
        results["functional"] = run_functional_tests(args.model, modes)

    if not args.skip_benchmark:
        results["benchmark"] = run_benchmark(args.model, modes)

    output_file = f"{PLATFORM}_offload_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
