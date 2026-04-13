#!/usr/bin/env python3
"""
批量测试 --compilation-config 的 mode 和 dynamic_shapes_config 参数
适用于 Ascend NPU (vllm-ascend) 和 CUDA GPU (vllm) 环境

使用方法:
  # 在 NPU 环境下运行全部测试
  python compilation_config_batch_test.py

  # 只运行 mode 测试
  python compilation_config_batch_test.py --test-mode

  # 只运行 dynamic_shapes_config 测试
  python compilation_config_batch_test.py --test-dynamic-shapes

  # 只运行组合测试
  python compilation_config_batch_test.py --test-combo

  # 使用自定义模型路径
  python compilation_config_batch_test.py --model /path/to/model

  # 使用 vllm serve 模式（需要先启动服务）
  python compilation_config_batch_test.py --serve-mode

  # 指定并发数和输出 token 数
  python compilation_config_batch_test.py --max-tokens 64 --num-prompts 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestResult:
    """单个测试的结果"""
    name: str
    config: dict
    success: bool = False
    throughput: Optional[float] = None  # tokens/s
    latency_ms: Optional[float] = None  # 端到端延迟 ms
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_time_s: float = 0.0
    error_msg: str = ""
    raw_output: str = ""


@dataclass
class TestSuite:
    """测试套件"""
    results: list = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    def summary(self):
        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        print(f"{'测试名称':<55} {'状态':<8} {'吞吐量(toks/s)':<15} {'输出tokens':<12} {'耗时(s)':<10}")
        print("-" * 80)
        for r in self.results:
            status = "✅ 成功" if r.success else "❌ 失败"
            tp = f"{r.throughput:.2f}" if r.throughput else "N/A"
            ot = str(r.output_tokens) if r.output_tokens else "N/A"
            tt = f"{r.total_time_s:.2f}" if r.total_time_s else "N/A"
            print(f"{r.name:<55} {status:<8} {tp:<15} {ot:<12} {tt:<10}")
        print("-" * 80)
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed
        print(f"总计: {len(self.results)} 个测试, {passed} 通过, {failed} 失败")
        print("=" * 80)

        # 输出失败详情
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            print("\n失败测试详情:")
            for r in failed_tests:
                print(f"\n  {r.name}:")
                print(f"    错误: {r.error_msg[:200]}")


# ==================== NPU (vllm-ascend) 测试 ====================

def run_npu_offline_test(test_name: str, compilation_config: dict,
                         model: str, prompt: str,
                         max_tokens: int = 32,
                         env_vars: dict = None) -> TestResult:
    """在 NPU 上使用 Offline Batched Inference 运行测试"""
    result = TestResult(name=test_name, config=compilation_config)

    config_json = json.dumps(compilation_config)
    # 转义引号以嵌入 Python 字符串
    config_json_escaped = config_json.replace('"', '\\"')

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    python_code = f"""
import time
import torch
from vllm import LLM, SamplingParams

config = {json.dumps(compilation_config)}

t0 = time.time()
llm = LLM(
    model='{model}',
    compilation_config=config,
    max_model_len=256,
    max_num_seqs=4,
    trust_remote_code=True,
)
t_init = time.time() - t0

sampling_params = SamplingParams(temperature=0.0, max_tokens={max_tokens})
prompts = ['{prompt}']

t1 = time.time()
outputs = llm.generate(prompts, sampling_params)
t_gen = time.time() - t1

output = outputs[0]
prompt_tokens = len(outputs[0].prompt_token_ids)
output_tokens = len(output.outputs[0].tokens) if hasattr(output.outputs[0], 'tokens') else len(output.outputs[0].token_ids)
throughput = output_tokens / t_gen if t_gen > 0 else 0

print(f"INIT_TIME={{t_init:.2f}}")
print(f"GEN_TIME={{t_gen:.2f}}")
print(f"PROMPT_TOKENS={{prompt_tokens}}")
print(f"OUTPUT_TOKENS={{output_tokens}}")
print(f"THROUGHPUT={{throughput:.2f}}")
print(f"TEXT={{output.outputs[0].text[:100]}}")
"""

    try:
        t_start = time.time()
        proc = subprocess.run(
            [sys.executable, "-c", python_code],
            capture_output=True, text=True, timeout=300, env=env
        )
        result.total_time_s = time.time() - t_start
        result.raw_output = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            result.success = False
            # 提取错误信息
            stderr_lines = proc.stderr.strip().split('\n')
            result.error_msg = '\n'.join(stderr_lines[-10:])  # 最后10行
            return result

        # 解析输出
        for line in proc.stdout.strip().split('\n'):
            if line.startswith("THROUGHPUT="):
                result.throughput = float(line.split("=")[1])
            elif line.startswith("PROMPT_TOKENS="):
                result.prompt_tokens = int(line.split("=")[1])
            elif line.startswith("OUTPUT_TOKENS="):
                result.output_tokens = int(line.split("=")[1])
            elif line.startswith("GEN_TIME="):
                result.latency_ms = float(line.split("=")[1]) * 1000

        result.success = True
    except subprocess.TimeoutExpired:
        result.success = False
        result.error_msg = "测试超时 (300s)"
    except Exception as e:
        result.success = False
        result.error_msg = str(e)

    return result


def run_npu_serve_test(test_name: str, compilation_config: dict,
                       model: str, prompt: str,
                       max_tokens: int = 32,
                       port: int = 8000,
                       env_vars: dict = None) -> TestResult:
    """在 NPU 上使用 vllm serve 模式运行测试（启动服务 + API 调用）"""
    result = TestResult(name=test_name, config=compilation_config)

    config_json = json.dumps(compilation_config)
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    # 启动 vllm serve 的命令
    serve_cmd = [
        "vllm", "serve", model,
        "--compilation-config", config_json,
        "--max-model-len", "256",
        "--max-num-seqs", "4",
        "--port", str(port),
        "--trust-remote-code",
        "--disable-log-requests",
    ]

    server_proc = None
    try:
        # 启动服务器
        print(f"  启动 vllm serve (port={port})...")
        server_proc = subprocess.Popen(
            serve_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env
        )

        # 等待服务就绪（最多120秒）
        import socket
        ready = False
        for i in range(120):
            time.sleep(1)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(("127.0.0.1", port))
                sock.close()
                ready = True
                break
            except (ConnectionRefusedError, socket.error):
                if server_proc.poll() is not None:
                    result.error_msg = f"服务启动失败: {server_proc.stderr.read().decode()[-500:]}"
                    return result
                continue

        if not ready:
            result.error_msg = "服务启动超时 (120s)"
            return result

        print(f"  服务已就绪，开始推理测试...")

        # 使用 curl 或 requests 发送请求
        try:
            import urllib.request
            req_data = json.dumps({
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }).encode()

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )

            t_start = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = json.loads(resp.read().decode())
            t_gen = time.time() - t_start

            result.throughput = resp_data.get("usage", {}).get("completion_tokens", 0) / t_gen if t_gen > 0 else 0
            result.prompt_tokens = resp_data.get("usage", {}).get("prompt_tokens", 0)
            result.output_tokens = resp_data.get("usage", {}).get("completion_tokens", 0)
            result.latency_ms = t_gen * 1000
            result.total_time_s = t_gen
            result.success = True
        except Exception as e:
            result.error_msg = f"API 调用失败: {e}"
    except Exception as e:
        result.error_msg = str(e)
    finally:
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    return result


# ==================== CUDA (vllm) 测试 ====================

def generate_cuda_commands(model: str, prompt: str, max_tokens: int = 32) -> list:
    """生成 CUDA 环境下的测试命令（供用户在 GPU 环境手动执行）"""
    commands = []

    # mode 测试
    for mode in ["NONE", "VLLM_COMPILE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE"]:
        commands.append({
            "name": f"CUDA mode={mode}",
            "config": {"mode": mode},
            "offline": (
                f'python -c "\n'
                f'from vllm import LLM, SamplingParams\n'
                f'llm = LLM(model=\'{model}\', compilation_config={{\'mode\': \'{mode}\'}}, max_model_len=256)\n'
                f'outputs = llm.generate([\'{prompt}\'], SamplingParams(temperature=0.0, max_tokens={max_tokens}))\n'
                f\'print(outputs[0].outputs[0].text)\n'
                f'"'
            ),
            "serve": (
                f'vllm serve {model} \\\\\n'
                f'  --compilation-config \'{{"mode": "{mode}"}}\''
            ),
        })

    # dynamic_shapes_config 测试
    ds_tests = [
        ("CUDA STOCK_TORCH_COMPILE + BACKED",
         {"mode": "STOCK_TORCH_COMPILE", "dynamic_shapes_config": {"type": "backed"}},
         {}),
        ("CUDA STOCK_TORCH_COMPILE + UNBACKED",
         {"mode": "STOCK_TORCH_COMPILE", "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": False}},
         {"VLLM_USE_BYTECODE_HOOK": "0"}),
        ("CUDA STOCK_TORCH_COMPILE + evaluate_guards=True",
         {"mode": "STOCK_TORCH_COMPILE", "dynamic_shapes_config": {"type": "backed", "evaluate_guards": True}},
         {"VLLM_USE_BYTECODE_HOOK": "0"}),
        ("CUDA DYNAMO_TRACE_ONCE + UNBACKED",
         {"mode": "DYNAMO_TRACE_ONCE", "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": False}},
         {"VLLM_USE_BYTECODE_HOOK": "0"}),
    ]

    for name, config, envs in ds_tests:
        env_prefix = " ".join(f'{k}={v}' for k, v in envs.items()) + " " if envs else ""
        commands.append({
            "name": name,
            "config": config,
            "env": envs,
            "offline": (
                f'{env_prefix}python -c "\n'
                f'from vllm import LLM, SamplingParams\n'
                f'llm = LLM(model=\'{model}\', compilation_config={json.dumps(config)}, max_model_len=256)\n'
                f'outputs = llm.generate([\'{prompt}\'], SamplingParams(temperature=0.0, max_tokens={max_tokens}))\n'
                f'print(outputs[0].outputs[0].text)\n'
                f'"'
            ),
            "serve": (
                f'{env_prefix}vllm serve {model} \\\\\n'
                f'  --compilation-config \'{json.dumps(config)}\''
            ),
        })

    return commands


# ==================== 测试定义 ====================

def get_npu_test_cases(model: str, prompt: str, max_tokens: int):
    """定义所有 NPU 测试用例"""
    tests = []

    # ---- 一、mode 参数测试 ----
    mode_tests = [
        {
            "name": "NPU mode=NONE (纯 eager)",
            "config": {"mode": "NONE"},
            "env": {},
            "desc": "纯 eager，不使用任何编译加速"
        },
        {
            "name": "NPU mode=VLLM_COMPILE (ACL Graph)",
            "config": {"mode": "VLLM_COMPILE"},
            "env": {},
            "desc": "使用 ACL Graph 分段编译，性能最佳"
        },
        {
            "name": "NPU mode=STOCK_TORCH_COMPILE (标准 torch.compile)",
            "config": {"mode": "STOCK_TORCH_COMPILE"},
            "env": {},
            "desc": "标准 torch.compile，eager 后端"
        },
        {
            "name": "NPU mode=DYNAMO_TRACE_ONCE (单次 trace)",
            "config": {"mode": "DYNAMO_TRACE_ONCE"},
            "env": {},
            "desc": "单次 Dynamo trace，移除 guards"
        },
    ]
    tests.append(("mode", mode_tests))

    # ---- 二、dynamic_shapes_config 参数测试 ----
    ds_tests = [
        {
            "name": "NPU DYNAMO_TRACE_ONCE + UNBACKED",
            "config": {
                "mode": "DYNAMO_TRACE_ONCE",
                "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": False}
            },
            "env": {"VLLM_USE_BYTECODE_HOOK": "0"},
            "desc": "UNBACKED 需要 VLLM_USE_BYTECODE_HOOK=0"
        },
        {
            "name": "NPU STOCK_TORCH_COMPILE + BACKED + evaluate_guards=True",
            "config": {
                "mode": "STOCK_TORCH_COMPILE",
                "dynamic_shapes_config": {"type": "backed", "evaluate_guards": True}
            },
            "env": {"VLLM_USE_BYTECODE_HOOK": "0"},
            "desc": "evaluate_guards=True 保留 SHAPE_ENV guards"
        },
        {
            "name": "NPU DYNAMO_TRACE_ONCE + BACKED",
            "config": {
                "mode": "DYNAMO_TRACE_ONCE",
                "dynamic_shapes_config": {"type": "backed", "evaluate_guards": False}
            },
            "env": {},
            "desc": "标准配置，BACKED + 不评估 guards"
        },
        {
            "name": "NPU STOCK_TORCH_COMPILE + UNBACKED",
            "config": {
                "mode": "STOCK_TORCH_COMPILE",
                "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": False}
            },
            "env": {"VLLM_USE_BYTECODE_HOOK": "0"},
            "desc": "UNBACKED 直接透传"
        },
        {
            "name": "NPU DYNAMO_TRACE_ONCE + assume_32_bit_indexing",
            "config": {
                "mode": "DYNAMO_TRACE_ONCE",
                "dynamic_shapes_config": {"assume_32_bit_indexing": True}
            },
            "env": {},
            "desc": "32 位索引假设，传给 Inductor"
        },
    ]
    tests.append(("dynamic_shapes", ds_tests))

    # ---- 三、组合测试 ----
    combo_tests = [
        {
            "name": "NPU VLLM_COMPILE + UNBACKED (自动回退)",
            "config": {
                "mode": "VLLM_COMPILE",
                "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": False}
            },
            "env": {},
            "desc": "VLLM_COMPILE 下 UNBACKED 自动回退到 BACKED + warning"
        },
        {
            "name": "NPU VLLM_COMPILE + evaluate_guards=True (自动设 False)",
            "config": {
                "mode": "VLLM_COMPILE",
                "dynamic_shapes_config": {"evaluate_guards": True}
            },
            "env": {},
            "desc": "VLLM_COMPILE 下 evaluate_guards=True 自动设为 False + warning"
        },
        {
            "name": "NPU DYNAMO_TRACE_ONCE + UNBACKED",
            "config": {
                "mode": "DYNAMO_TRACE_ONCE",
                "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": False}
            },
            "env": {"VLLM_USE_BYTECODE_HOOK": "0"},
            "desc": "DYNAMO_TRACE_ONCE 支持 UNBACKED 直接透传"
        },
    ]
    tests.append(("combo", combo_tests))

    return tests, model, prompt, max_tokens


# ==================== 命令行打印 ====================

def print_cuda_commands(model: str, prompt: str, max_tokens: int):
    """打印 CUDA 环境测试命令供用户参考"""
    print("\n" + "=" * 80)
    print("CUDA (vLLM) 测试命令（需要在 GPU 环境手动执行）")
    print("=" * 80)

    commands = generate_cuda_commands(model, prompt, max_tokens)
    for cmd in commands:
        print(f"\n{'#' * 70}")
        print(f"# {cmd['name']}")
        print(f"# 配置: {json.dumps(cmd['config'])}")
        if cmd.get('env'):
            print(f"# 环境变量: {cmd['env']}")
        print(f"{'#' * 70}")
        print(f"\n# --- Offline 模式 ---")
        print(cmd['offline'])
        print(f"\n# --- vllm serve 模式 ---")
        print(cmd['serve'])

    print("\n" + "=" * 80)
    print("说明:")
    print("  - Offline 模式: 直接用 python -c 运行，输出结果和吞吐量")
    print("  - vllm serve 模式: 启动 OpenAI 兼容 API 服务，用 curl 调用")
    print("  - CUDA 使用 Inductor 后端，NPU 使用 eager 后端")
    print("  - UNBACKED 需要 VLLM_USE_BYTECODE_HOOK=0 环境变量")
    print("=" * 80)


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="批量测试 --compilation-config 参数")
    parser.add_argument("--model", type=str,
                        default="/root/work/models/Qwen2.5-0.5B-Instruct/",
                        help="模型路径或 HuggingFace 模型名")
    parser.add_argument("--prompt", type=str,
                        default="Hello, my name is",
                        help="测试 prompt")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="最大输出 token 数")
    parser.add_argument("--num-prompts", type=int, default=1,
                        help="每个测试发送的 prompt 数量（当前仅支持 1）")
    parser.add_argument("--test-mode", action="store_true",
                        help="只运行 mode 参数测试")
    parser.add_argument("--test-dynamic-shapes", action="store_true",
                        help="只运行 dynamic_shapes_config 参数测试")
    parser.add_argument("--test-combo", action="store_true",
                        help="只运行组合测试")
    parser.add_argument("--serve-mode", action="store_true",
                        help="使用 vllm serve 模式（需要手动启动服务）")
    parser.add_argument("--print-cuda-commands", action="store_true",
                        help="只打印 CUDA 测试命令，不运行")
    parser.add_argument("--port", type=int, default=8000,
                        help="vllm serve 使用的端口")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要执行的命令，不实际运行")

    args = parser.parse_args()

    # 确定要运行的测试类别
    run_categories = set()
    if args.test_mode:
        run_categories.add("mode")
    if args.test_dynamic_shapes:
        run_categories.add("dynamic_shapes")
    if args.test_combo:
        run_categories.add("combo")
    if not run_categories:
        run_categories = {"mode", "dynamic_shapes", "combo"}

    tests, model, prompt, max_tokens = get_npu_test_cases(
        args.model, args.prompt, args.max_tokens
    )

    # 打印 CUDA 命令
    if args.print_cuda_commands:
        print_cuda_commands(model, prompt, max_tokens)
        return

    # 打印测试计划
    print("=" * 80)
    print("--compilation-config 批量测试")
    print("=" * 80)
    print(f"模型: {model}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"运行类别: {', '.join(sorted(run_categories))}")
    print(f"模式: {'vllm serve' if args.serve_mode else 'Offline Batched Inference'}")
    print("=" * 80)

    # Dry run
    if args.dry_run:
        print("\n[Dry Run] 将要执行的测试:\n")
        for category, test_cases in tests:
            if category not in run_categories:
                continue
            print(f"\n--- {category} ---")
            for tc in test_cases:
                config_str = json.dumps(tc["config"])
                env_str = " ".join(f"{k}={v}" for k, v in tc.get("env", {}).items())
                if env_str:
                    print(f"  {env_str} python -c \"from vllm import LLM; LLM(model='{model}', compilation_config={config_str})\"")
                else:
                    print(f"  python -c \"from vllm import LLM; LLM(model='{model}', compilation_config={config_str})\"")
        return

    # 运行测试
    suite = TestSuite()
    port = args.port

    for category, test_cases in tests:
        if category not in run_categories:
            continue

        print(f"\n{'=' * 80}")
        print(f"开始测试类别: {category}")
        print(f"{'=' * 80}")

        for i, tc in enumerate(test_cases):
            print(f"\n[{i+1}/{len(test_cases)}] {tc['name']}")
            print(f"  配置: {json.dumps(tc['config'])}")
            if tc.get("desc"):
                print(f"  说明: {tc['desc']}")
            if tc.get("env"):
                print(f"  环境变量: {tc['env']}")

            if args.serve_mode:
                result = run_npu_serve_test(
                    test_name=tc["name"],
                    compilation_config=tc["config"],
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    port=port,
                    env_vars=tc.get("env"),
                )
                port += 1  # 每个测试用不同端口
            else:
                result = run_npu_offline_test(
                    test_name=tc["name"],
                    compilation_config=tc["config"],
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    env_vars=tc.get("env"),
                )

            suite.add(result)

            if result.success:
                print(f"  结果: ✅ 吞吐量={result.throughput:.2f} toks/s, "
                      f"输出={result.output_tokens} tokens, "
                      f"耗时={result.total_time_s:.2f}s")
            else:
                print(f"  结果: ❌ 失败 - {result.error_msg[:100]}")

    # 打印汇总
    suite.summary()

    # 打印 CUDA 对照命令
    print("\n")
    print_cuda_commands(model, prompt, max_tokens)

    # 保存结果到文件
    result_file = "compilation_config_test_results.json"
    results_data = []
    for r in suite.results:
        results_data.append({
            "name": r.name,
            "config": r.config,
            "success": r.success,
            "throughput": r.throughput,
            "prompt_tokens": r.prompt_tokens,
            "output_tokens": r.output_tokens,
            "total_time_s": r.total_time_s,
            "error_msg": r.error_msg,
        })
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()
