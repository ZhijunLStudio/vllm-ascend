# SPDX-License-Identifier: Apache-2.0
"""
CUDA E2E Reproduction Script for speculative_token_tree

This script runs on a CUDA machine to reproduce the tree speculative decoding
test with the same model pair and configuration as the NPU tests.

Target model: Meta-Llama-3.1-8B-Instruct
Draft model:  yuhuili/EAGLE3-LLaMA3.1-Instruct-8B

Usage:
    # 1. Download models (if not already present):
    python tree_spec_decode_cuda_e2e.py --download

    # 2. Run baseline (no speculative decoding):
    python tree_spec_decode_cuda_e2e.py --mode baseline

    # 3. Run tree speculative decoding:
    python tree_spec_decode_cuda_e2e.py --mode tree

    # 4. Run both and compare:
    python tree_spec_decode_cuda_e2e.py --mode both

Requirements:
    - CUDA GPU with >= 24GB VRAM
    - pip install vllm  (or from source)
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ============ Configuration ============
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
TARGET_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DRAFT_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
TARGET_DIR = os.path.join(MODELS_DIR, "Meta-Llama-3.1-8B-Instruct")
DRAFT_DIR = os.path.join(MODELS_DIR, "EAGLE3-LLaMA3.1-Instruct-8B")

BASELINE_PORT = 8000
TREE_PORT = 8001

TREE_CONFIGS = [
    {
        "name": "tree_n2",
        "num_speculative_tokens": 2,
        "speculative_token_tree": "[(0,), (0, 0)]",
    },
    {
        "name": "tree_n3",
        "num_speculative_tokens": 3,
        "speculative_token_tree": "[(0,), (0, 0), (0, 1)]",
    },
]

# Test prompts (same as NPU tests)
TEST_PROMPTS = [
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

MAX_TOKENS = 100
TEMPERATURE = 0.0
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 2048
NUM_REQUESTS = 10


def download_models():
    """Download target and draft models using huggingface-cli."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(TARGET_DIR, "config.json")):
        print(f"[DOWNLOAD] Target model: {TARGET_MODEL}")
        subprocess.run(
            ["huggingface-cli", "download", TARGET_MODEL, "--local-dir", TARGET_DIR],
            check=True,
        )
    else:
        print(f"[SKIP] Target model exists: {TARGET_DIR}")

    if not os.path.exists(os.path.join(DRAFT_DIR, "config.json")):
        print(f"[DOWNLOAD] Draft model: {DRAFT_MODEL}")
        subprocess.run(
            ["huggingface-cli", "download", DRAFT_MODEL, "--local-dir", DRAFT_DIR],
            check=True,
        )
    else:
        print(f"[SKIP] Draft model exists: {DRAFT_DIR}")

    print("[OK] All models downloaded.")


def kill_vllm():
    """Kill any running vLLM servers."""
    subprocess.run(["pkill", "-f", "vllm.entrypoints"], capture_output=True)
    time.sleep(3)


def start_server(port, speculative_config=None):
    """Start a vLLM OpenAI-compatible server."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET_DIR,
        "--port", str(port),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    ]
    if speculative_config:
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])

    log_file = f"/tmp/cuda_vllm_server_{port}.log"
    print(f"[START] Server on port {port}, log: {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc, log_file


def wait_for_server(port, timeout=120):
    """Wait for server to be ready."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(url, timeout=5)
            if req.status == 200:
                print(f"[READY] Server on port {port}")
                return True
        except Exception:
            pass
        time.sleep(5)
    print(f"[TIMEOUT] Server on port {port} not ready after {timeout}s")
    return False


def send_completion(port, prompt, max_tokens=MAX_TOKENS):
    """Send a completion request and return the response."""
    import urllib.request
    url = f"http://localhost:{port}/v1/completions"
    data = json.dumps({
        "model": TARGET_DIR,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


def measure_throughput(port, name, num_requests=NUM_REQUESTS):
    """Measure throughput with sequential requests."""
    print(f"\n[MEASURE] {name}: {num_requests} sequential requests")
    results = []
    start_time = time.time()

    for i in range(num_requests):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        req_start = time.time()
        resp = send_completion(port, prompt)
        req_time = time.time() - req_start
        completion_tokens = resp["usage"]["completion_tokens"]
        results.append({
            "prompt_idx": i % len(TEST_PROMPTS),
            "completion_tokens": completion_tokens,
            "time_s": round(req_time, 3),
            "tokens_per_s": round(completion_tokens / req_time, 2),
        })
        print(f"  Request {i+1}: {completion_tokens} tokens in {req_time:.2f}s ({completion_tokens/req_time:.1f} tokens/s)")

    total_time = time.time() - start_time
    total_tokens = sum(r["completion_tokens"] for r in results)
    avg_throughput = total_tokens / total_time

    summary = {
        "name": name,
        "total_requests": num_requests,
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "avg_throughput_tokens_per_s": round(avg_throughput, 2),
        "per_request": results,
    }
    print(f"  TOTAL: {total_tokens} tokens in {total_time:.2f}s = {avg_throughput:.2f} tokens/s")
    return summary


def get_spec_metrics(log_file):
    """Parse SpecDecoding metrics from server log."""
    metrics = []
    with open(log_file) as f:
        for line in f:
            if "SpecDecoding metrics" in line:
                # Extract: Mean acceptance length: X, Per-position acceptance rate: ...
                parts = line.strip().split("SpecDecoding metrics: ")[-1]
                metrics.append(parts)
    return metrics


def run_test(mode):
    """Run the specified test mode."""
    kill_vllm()
    results = {}

    if mode in ("baseline", "both"):
        proc, log_file = start_server(BASELINE_PORT)
        if not wait_for_server(BASELINE_PORT):
            proc.kill()
            return results

        baseline_result = measure_throughput(BASELINE_PORT, "Baseline (no speculative)")
        results["baseline"] = baseline_result

        if mode == "both":
            proc.kill()
            time.sleep(5)

    if mode in ("tree", "both"):
        for config in TREE_CONFIGS:
            kill_vllm()
            time.sleep(5)

            spec_config = {
                "method": "eagle3",
                "model": DRAFT_DIR,
                "num_speculative_tokens": config["num_speculative_tokens"],
                "speculative_token_tree": config["speculative_token_tree"],
            }
            proc, log_file = start_server(TREE_PORT, spec_config)
            if not wait_for_server(TREE_PORT):
                proc.kill()
                continue

            tree_result = measure_throughput(TREE_PORT, f"Tree {config['name']}")
            tree_result["spec_metrics"] = get_spec_metrics(log_file)
            results[config["name"]] = tree_result

            proc.kill()
            time.sleep(5)

    kill_vllm()
    return results


def print_comparison(results):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    baseline_tp = results.get("baseline", {}).get("avg_throughput_tokens_per_s", 0)

    print(f"\n{'Mode':<20} {'Throughput (tokens/s)':<25} {'vs Baseline':<15}")
    print("-" * 60)

    for name, data in results.items():
        tp = data.get("avg_throughput_tokens_per_s", 0)
        if name == "baseline":
            print(f"{'Baseline':<20} {tp:<25.2f} {'—':<15}")
        else:
            pct = ((tp - baseline_tp) / baseline_tp * 100) if baseline_tp > 0 else 0
            sign = "+" if pct >= 0 else ""
            print(f"{name:<20} {tp:<25.2f} {sign}{pct:.1f}%")

    # Print spec metrics
    for name, data in results.items():
        if "spec_metrics" in data and data["spec_metrics"]:
            print(f"\n{name} SpecDecoding metrics:")
            for m in data["spec_metrics"]:
                print(f"  {m}")


def save_results(results, output_path="tree_spec_decode_cuda_results.json"):
    """Save results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CUDA E2E test for tree speculative decoding")
    parser.add_argument("--mode", choices=["baseline", "tree", "both"], default="both",
                        help="Test mode: baseline only, tree only, or both")
    parser.add_argument("--download", action="store_true", help="Download models first")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="Directory for models")
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS, help="Number of test requests")
    args = parser.parse_args()

    global MODELS_DIR, TARGET_DIR, DRAFT_DIR, NUM_REQUESTS
    MODELS_DIR = args.models_dir
    TARGET_DIR = os.path.join(MODELS_DIR, "Meta-Llama-3.1-8B-Instruct")
    DRAFT_DIR = os.path.join(MODELS_DIR, "EAGLE3-LLaMA3.1-Instruct-8B")
    NUM_REQUESTS = args.num_requests

    print("=" * 60)
    print("CUDA Tree Speculative Decoding E2E Test")
    print("=" * 60)
    print(f"Target model: {TARGET_DIR}")
    print(f"Draft model:  {DRAFT_DIR}")
    print(f"Mode: {args.mode}")
    print(f"Requests: {NUM_REQUESTS}")
    print()

    if args.download:
        download_models()
        if args.mode == "download":
            return

    results = run_test(args.mode)
    print_comparison(results)
    save_results(results)


if __name__ == "__main__":
    main()
