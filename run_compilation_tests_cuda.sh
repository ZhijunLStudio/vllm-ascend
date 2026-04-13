#!/bin/bash
# ============================================================
# --compilation-config CUDA 对照测试脚本
# 在 CUDA GPU 环境下运行 vLLM 原版，作为 NPU 测试的对照基准
#
# 使用方式：
#   1. 将此脚本拷贝到 CUDA 机子上
#   2. 安装 vLLM：pip install vllm
#   3. 运行：./run_compilation_tests_cuda.sh
#
# 注意：此脚本使用 HuggingFace 模型 ID（非本地路径），
#       CUDA 机子会自动从 HuggingFace 下载模型。
# ============================================================

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS=32
MAX_MODEL_LEN=256
MAX_NUM_SEQS=4

LOG_DIR="./test_logs/cuda_compilation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
TOTAL=0
CORRECT=0
MISMATCH=0

# Baseline outputs (from test_1 NONE mode)
BASELINE_FILE="$LOG_DIR/baseline_outputs.txt"

run_test() {
    local name="$1"
    local config="$2"
    local extra_env="$3"
    local expect_fail="${4:-0}"  # 1 = expect failure (e.g. invalid mode)

    TOTAL=$((TOTAL + 1))
    local logfile="$LOG_DIR/test_${TOTAL}_$(echo "$name" | tr ' /:()+' '_').log"

    echo "------------------------------------------------------------"
    echo "[$TOTAL] $name"
    echo "  config: $config"
    [ -n "$extra_env" ] && echo "  env:    $extra_env"
    [ "$expect_fail" = "1" ] && echo "  expect: FAILURE"

    # 写临时 python 文件（配置通过环境变量传递，避免 JSON 注入）
    local pyfile="$LOG_DIR/run_test_${TOTAL}.py"
    cat > "$pyfile" << 'PYEOF'
import os, json, sys, time

# 从环境变量读取配置（避免 bash 引号注入问题）
config = json.loads(os.environ['TEST_CONFIG'])
model = os.environ.get('TEST_MODEL', '')
max_model_len = int(os.environ.get('TEST_MAX_MODEL_LEN', '256'))
max_num_seqs = int(os.environ.get('TEST_MAX_NUM_SEQS', '4'))
max_tokens = int(os.environ.get('TEST_MAX_TOKENS', '32'))

from vllm import LLM, SamplingParams

prompts = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms and provide examples.",
    "Write a detailed comparison between Python and Rust programming languages, "
    "covering performance, safety, ecosystem, and use cases.",
]

t0 = time.time()
llm = LLM(
    model=model,
    compilation_config=config,
    max_model_len=max_model_len,
    max_num_seqs=max_num_seqs,
    trust_remote_code=True,
)
t_init = time.time() - t0

sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

# --- Warmup ---
print("WARMUP_START")
_ = llm.generate([prompts[0]], sp)
print("WARMUP_DONE")

# --- 正式测试：3 次不同长度的问句 ---
t1 = time.time()
all_outputs = []
all_tokens = []
for prompt in prompts:
    outputs = llm.generate([prompt], sp)
    out = outputs[0]
    text = out.outputs[0].text
    tok_count = len(out.outputs[0].token_ids)
    all_outputs.append(text)
    all_tokens.append(tok_count)
t_gen = time.time() - t1

total_tokens = sum(all_tokens)
tp = total_tokens / t_gen if t_gen > 0 else 0

print(f'INIT_TIME={t_init:.2f}')
print(f'GEN_TIME={t_gen:.2f}')
print(f'TOTAL_TOKENS={total_tokens}')
print(f'THROUGHPUT={tp:.2f}')
for i, text in enumerate(all_outputs):
    print(f'OUTPUT_{i}={text}')
PYEOF

    export TEST_CONFIG="$config"
    export TEST_MODEL="$MODEL"
    export TEST_MAX_MODEL_LEN="$MAX_MODEL_LEN"
    export TEST_MAX_NUM_SEQS="$MAX_NUM_SEQS"
    export TEST_MAX_TOKENS="$MAX_TOKENS"

    local start_ms=$(date +%s%N)

    local exit_code=0
    if [ -n "$extra_env" ]; then
        eval "$extra_env" python "$pyfile" > >(tee "$logfile") 2>&1
        exit_code=$?
    else
        python "$pyfile" 2>&1 | tee "$logfile"
        exit_code=${PIPESTATUS[0]}
    fi

    local end_ms=$(date +%s%N)
    local elapsed=$(( (end_ms - start_ms) / 1000000 ))

    unset TEST_CONFIG TEST_MODEL TEST_MAX_MODEL_LEN TEST_MAX_NUM_SEQS TEST_MAX_TOKENS

    if [ "$expect_fail" = "1" ]; then
        if [ $exit_code -ne 0 ]; then
            echo "  >>> PASS  (预期失败, exit_code=$exit_code)  总耗时=${elapsed}ms"
            PASS=$((PASS + 1))
        else
            echo "  >>> FAIL  (预期失败但成功了)  总耗时=${elapsed}ms"
            FAIL=$((FAIL + 1))
        fi
        echo ""
        return
    fi

    if [ $exit_code -eq 0 ]; then
        local tp=$(grep "^THROUGHPUT=" "$logfile" | tail -1 | cut -d= -f2)
        local tt=$(grep "^TOTAL_TOKENS=" "$logfile" | tail -1 | cut -d= -f2)
        echo "  >>> PASS  吞吐量=${tp:-N/A} toks/s  总输出=${tt:-N/A} tokens  总耗时=${elapsed}ms"
        PASS=$((PASS + 1))

        # 正确性比对：与 baseline (NONE) 比较
        if [ -f "$BASELINE_FILE" ] && [ "$TOTAL" != "1" ]; then
            local match=1
            for i in 0 1 2; do
                local cur=$(grep "^OUTPUT_${i}=" "$logfile" | head -1 | sed "s/^OUTPUT_${i}=//")
                local base=$(grep "^OUTPUT_${i}=" "$BASELINE_FILE" | head -1 | sed "s/^OUTPUT_${i}=//")
                if [ -n "$cur" ] && [ -n "$base" ]; then
                    if [ "$cur" = "$base" ]; then
                        echo "    OUTPUT_${i}: 与 baseline 一致"
                    else
                        echo "    OUTPUT_${i}: 与 baseline 不一致"
                        echo "      baseline: ${base:0:60}"
                        echo "      current:  ${cur:0:60}"
                        match=0
                    fi
                fi
            done
            if [ $match -eq 1 ]; then
                CORRECT=$((CORRECT + 1))
            else
                MISMATCH=$((MISMATCH + 1))
            fi
        fi
    else
        echo "  >>> FAIL  exit_code=$exit_code  总耗时=${elapsed}ms"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

echo "============================================================"
echo "--compilation-config CUDA 对照测试"
echo "============================================================"
echo "模型:       $MODEL"
echo "Max tokens: $MAX_TOKENS"
echo "日志目录:   $LOG_DIR"
echo "============================================================"
echo ""

# ============================================================
# 一、mode 参数测试（第 1 个测试保存为 baseline）
# ============================================================
echo ""
echo "############################################################"
echo "# 一、mode 参数测试"
echo "############################################################"
echo ""

run_test "mode=NONE (纯 eager, baseline)" \
    '{"mode": "NONE"}' \
    ""

# 保存 baseline
baseline_log=$(ls "$LOG_DIR"/test_1_*.log 2>/dev/null | head -1)
if [ -n "$baseline_log" ]; then
    cp "$baseline_log" "$BASELINE_FILE"
    echo ">>> Baseline 已保存到 $BASELINE_FILE"
    echo ""
fi

run_test "mode=VLLM_COMPILE (CUDA Graph)" \
    '{"mode": "VLLM_COMPILE"}' \
    ""

run_test "mode=STOCK_TORCH_COMPILE (标准 torch.compile)" \
    '{"mode": "STOCK_TORCH_COMPILE"}' \
    ""

run_test "mode=DYNAMO_TRACE_ONCE (单次 trace)" \
    '{"mode": "DYNAMO_TRACE_ONCE"}' \
    ""

# ============================================================
# 二、dynamic_shapes_config 参数测试
# ============================================================
echo ""
echo "############################################################"
echo "# 二、dynamic_shapes_config 参数测试"
echo "############################################################"
echo ""

run_test "STOCK_TORCH_COMPILE + type=BACKED" \
    '{"mode": "STOCK_TORCH_COMPILE", "dynamic_shapes_config": {"type": "backed"}}' \
    ""

run_test "DYNAMO_TRACE_ONCE + type=UNBACKED" \
    '{"mode": "DYNAMO_TRACE_ONCE", "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": false}}' \
    "VLLM_USE_BYTECODE_HOOK=0"

run_test "STOCK_TORCH_COMPILE + evaluate_guards=True" \
    '{"mode": "STOCK_TORCH_COMPILE", "dynamic_shapes_config": {"type": "backed", "evaluate_guards": true}}' \
    "VLLM_USE_BYTECODE_HOOK=0"

run_test "STOCK_TORCH_COMPILE + type=UNBACKED" \
    '{"mode": "STOCK_TORCH_COMPILE", "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": false}}' \
    "VLLM_USE_BYTECODE_HOOK=0"

run_test "DYNAMO_TRACE_ONCE + assume_32_bit_indexing" \
    '{"mode": "DYNAMO_TRACE_ONCE", "dynamic_shapes_config": {"assume_32_bit_indexing": true}}' \
    ""

run_test "STOCK_TORCH_COMPILE + type=BACKED_SIZE_OBLIVIOUS" \
    '{"mode": "STOCK_TORCH_COMPILE", "dynamic_shapes_config": {"type": "backed_size_oblivious"}}' \
    ""

# ============================================================
# 三、组合测试
# ============================================================
echo ""
echo "############################################################"
echo "# 三、组合测试"
echo "############################################################"
echo ""

run_test "VLLM_COMPILE + UNBACKED" \
    '{"mode": "VLLM_COMPILE", "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": false}}' \
    ""

run_test "VLLM_COMPILE + evaluate_guards=True" \
    '{"mode": "VLLM_COMPILE", "dynamic_shapes_config": {"evaluate_guards": true}}' \
    "VLLM_USE_BYTECODE_HOOK=0"

run_test "DYNAMO_TRACE_ONCE + UNBACKED" \
    '{"mode": "DYNAMO_TRACE_ONCE", "dynamic_shapes_config": {"type": "unbacked", "evaluate_guards": false}}' \
    "VLLM_USE_BYTECODE_HOOK=0"

run_test "NONE + BACKED" \
    '{"mode": "NONE", "dynamic_shapes_config": {"type": "backed"}}' \
    ""

# ============================================================
# 四、错误处理测试
# ============================================================
echo ""
echo "############################################################"
echo "# 四、错误处理测试"
echo "############################################################"
echo ""

run_test "非法 mode=INVALID（应报错）" \
    '{"mode": "INVALID"}' \
    "" \
    1

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================================"
echo "测试完成"
echo "============================================================"
echo "总计:     $TOTAL 个测试"
echo "通过:     $PASS"
echo "失败:     $FAIL"
if [ $((CORRECT + MISMATCH)) -gt 0 ]; then
    echo "正确性:   $CORRECT 一致 / $MISMATCH 不一致（与 NONE baseline 比对）"
fi
echo "日志:     $LOG_DIR/"
echo "============================================================"
