#!/usr/bin/env python3
"""
Tree Speculative Decoding 集成测试脚本 v3

正确处理：
1. 设置 VLLM_VERSION 环境变量
2. 多次请求以获得稳定的统计
3. 捕获服务器日志中的 SpecDecoding metrics
4. 对比 baseline 和 tree speculative decoding
"""

import subprocess
import time
import requests
import json
import os
import re


def start_vllm_server(
    model_path: str,
    port: int = 8000,
    speculative_config: dict = None,
    log_file: str = "/tmp/vllm_server.log",
) -> subprocess.Popen:
    """启动 vLLM 服务器。"""
    # 设置环境变量
    env = os.environ.copy()
    env["VLLM_VERSION"] = "0.19.0"

    cmd = [
        "/usr/local/python3.11.14/bin/python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", "2048",
    ]

    if speculative_config:
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])

    print(f"启动 vLLM 服务器: {' '.join(cmd)}")

    # 启动服务器
    log_f = open(log_file, "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    return process, log_f


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """等待服务器启动。"""
    print(f"等待服务器启动: {url}")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print("服务器已启动!")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(2)
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0:
            print(f"  等待中... ({elapsed}s)")

    print(f"服务器启动超时 ({timeout}s)")
    return False


def parse_specdec_metrics(log_file: str) -> dict:
    """从日志文件中解析 SpecDecoding metrics。"""
    metrics = {}

    try:
        with open(log_file, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return metrics

    # 查找所有 SpecDecoding metrics 行
    lines = content.split("\n")
    spec_lines = [line for line in lines if "SpecDecoding metrics:" in line]

    if not spec_lines:
        return metrics

    # 使用最后一行（最新统计）
    last_line = spec_lines[-1]

    # Mean acceptance length
    match = re.search(r"Mean acceptance length:\s*([\d.]+)", last_line)
    if match:
        metrics["mean_acceptance_length"] = float(match.group(1))

    # Accepted throughput
    match = re.search(r"Accepted throughput:\s*([\d.]+)", last_line)
    if match:
        metrics["accepted_throughput"] = float(match.group(1))

    # Drafted throughput
    match = re.search(r"Drafted throughput:\s*([\d.]+)", last_line)
    if match:
        metrics["drafted_throughput"] = float(match.group(1))

    # Per-position acceptance rate
    match = re.search(r"Per-position acceptance rate:\s*([\d., ]+)", last_line)
    if match:
        rates = [float(x.strip()) for x in match.group(1).split(",")]
        metrics["per_position_acceptance_rate"] = rates

    # Avg Draft acceptance rate
    match = re.search(r"Avg Draft acceptance rate:\s*([\d.]+)", last_line)
    if match:
        metrics["avg_draft_acceptance_rate"] = float(match.group(1))

    # Accepted/Drafted tokens
    match = re.search(r"Accepted:\s*(\d+)\s*tokens", last_line)
    if match:
        metrics["accepted_tokens"] = int(match.group(1))

    match = re.search(r"Drafted:\s*(\d+)\s*tokens", last_line)
    if match:
        metrics["drafted_tokens"] = int(match.group(1))

    return metrics


def send_completion_request(url: str, model: str, prompt: str, max_tokens: int = 100) -> dict:
    """发送 completion 请求。"""
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )

    if response.status_code != 200:
        print(f"错误: {response.status_code}")
        return None

    return response.json()


def main():
    print("=" * 60)
    print("Tree Speculative Decoding 集成测试 v3")
    print("=" * 60)

    # 配置
    MODEL_PATH = "/root/work/vllm-ascend/models/Qwen2.5-7B-Instruct"
    PORT = 8000
    BASE_URL = f"http://localhost:{PORT}"

    # Tree speculative decoding 配置
    SPECULATIVE_CONFIG = {
        "method": "eagle3",
        "model": "/root/work/vllm-ascend/models/Qwen2.5-7B-Instruct_EAGLE3_UltraChat",
        "num_speculative_tokens": 5,
        "speculative_token_tree": "[(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)]",
    }

    test_modes = [
        {"name": "Baseline (无推测解码)", "config": None, "log_file": "/tmp/vllm_baseline.log"},
        {"name": "Tree 推测解码", "config": SPECULATIVE_CONFIG, "log_file": "/tmp/vllm_tree.log"},
    ]

    results = {}

    for mode in test_modes:
        print(f"\n{'=' * 60}")
        print(f"测试模式: {mode['name']}")
        print(f"{'=' * 60}")

        # 启动服务器
        server, log_f = start_vllm_server(MODEL_PATH, PORT, mode["config"], mode["log_file"])

        try:
            # 等待服务器启动
            if not wait_for_server(BASE_URL, timeout=300):
                print("服务器启动失败!")
                continue

            # 获取模型名称
            model_response = requests.get(f"{BASE_URL}/v1/models")
            model_name = model_response.json()["data"][0]["id"]
            print(f"模型名称: {model_name}")

            # 发送多个请求以获得稳定的统计
            print("\n发送推理请求...")
            prompts = [
                "The future of artificial intelligence is",
                "Explain the concept of machine learning in simple terms.",
                "Write a short story about a robot learning to paint.",
                "What are the main challenges in natural language processing?",
                "Describe the evolution of deep learning over the past decade.",
                "How does transfer learning work in practice?",
                "What is the difference between supervised and unsupervised learning?",
                "Explain the attention mechanism in transformers.",
                "What are the ethical considerations in AI development?",
                "How do neural networks learn from data?",
            ]

            total_tokens = 0
            start_time = time.time()

            for i, prompt in enumerate(prompts):
                result = send_completion_request(BASE_URL, model_name, prompt, max_tokens=100)
                if result:
                    tokens = result["usage"]["completion_tokens"]
                    total_tokens += tokens
                    print(f"  请求 {i+1}/{len(prompts)}: {tokens} tokens")
                time.sleep(1)  # 短暂间隔

            end_time = time.time()
            throughput = total_tokens / (end_time - start_time)

            # 等待日志写入
            time.sleep(5)

            # 解析 SpecDecoding metrics
            metrics = parse_specdec_metrics(mode["log_file"])
            metrics["throughput"] = throughput
            metrics["total_tokens"] = total_tokens

            results[mode["name"]] = {
                "metrics": metrics,
                "status": "success",
            }

            print(f"\n{mode['name']} 结果:")
            print(f"  吞吐量: {throughput:.2f} tokens/s")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[mode["name"]] = {
                "status": "failed",
                "error": str(e),
            }

        finally:
            # 停止服务器
            print("\n停止服务器...")
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
            log_f.close()
            time.sleep(5)  # 等待端口释放

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for mode_name, result in results.items():
        print(f"\n{mode_name}:")
        if result["status"] == "success":
            metrics = result.get("metrics", {})
            print(f"  吞吐量: {metrics.get('throughput', 0):.2f} tokens/s")
            if "mean_acceptance_length" in metrics:
                print(f"  Mean acceptance length: {metrics['mean_acceptance_length']:.2f}")
            if "avg_draft_acceptance_rate" in metrics:
                print(f"  Avg Draft acceptance rate: {metrics['avg_draft_acceptance_rate']:.1f}%")
            if "per_position_acceptance_rate" in metrics:
                print(f"  Per-position acceptance rate: {metrics['per_position_acceptance_rate']}")
        else:
            print(f"  状态: {result['status']}")
            if "error" in result:
                print(f"  错误: {result['error']}")

    # 对比分析
    if "Tree 推测解码" in results and "Baseline (无推测解码)" in results:
        baseline = results["Baseline (无推测解码)"]
        tree = results["Tree 推测解码"]

        if baseline["status"] == "success" and tree["status"] == "success":
            baseline_tp = baseline["metrics"].get("throughput", 0)
            tree_tp = tree["metrics"].get("throughput", 0)

            if baseline_tp > 0:
                improvement = (tree_tp - baseline_tp) / baseline_tp * 100
                print(f"\n性能对比:")
                print(f"  Baseline: {baseline_tp:.2f} tokens/s")
                print(f"  Tree SD: {tree_tp:.2f} tokens/s")
                print(f"  提升: {improvement:.1f}%")

            tree_metrics = tree["metrics"]
            if "mean_acceptance_length" in tree_metrics:
                mal = tree_metrics["mean_acceptance_length"]
                print(f"\nTree 推测解码 Mean Acceptance Length: {mal:.2f}x")
                print(f"（每次验证平均接受 {mal:.2f} 个 draft token）")


if __name__ == "__main__":
    main()
