#!/usr/bin/env bash
# Run benchmark_serving.py from SemiAnalysisAI/InferenceX (pinned commit).
#
# Usage:
#   bash scripts/benchmark.sh \
#       --input-len 1000 --output-len 1000 \
#       --concurrency 1 --num-prompts 10
#
# Options (all optional — defaults match common test config):
#   --endpoint       http://HOST:PORT  (default: http://$CONTROL_PLANE_IP:$VLLM_PORT)
#   --model          MODEL_NAME        (default: MiniMax-M2.7)
#   --tokenizer      PATH              (default: from cluster.env)
#   --input-len      N                 (default: 1000)
#   --output-len     N                 (default: 1000)
#   --concurrency    N                 (default: 1)
#   --num-prompts    N                 (default: 10)
#   --result-dir     DIR               (default: $REPO_ROOT/benchmark_results)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

# ── Defaults ─────────────────────────────────────────────────────────────────
ENDPOINT="http://$CONTROL_PLANE_IP:$VLLM_PORT"
# Checkpoint path (model.env's MODEL) doubles as the tokenizer path — must be
# captured before MODEL is reassigned to the served name below, or the
# tokenizer arg silently becomes the served name (not a real HF repo id /
# local path), which fails to load.
TOKENIZER="${MODEL_PATH:-${MODEL:-}}"
MODEL="${SERVED_MODEL_NAME:-}"                 # from model.env
INPUT_LEN=1000
OUTPUT_LEN=1000
CONCURRENCY=1
NUM_PROMPTS=10
RESULT_DIR="$REPO_ROOT/benchmark_results"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint)   ENDPOINT="$2";    shift 2 ;;
        --model)      MODEL="$2";       shift 2 ;;
        --tokenizer)  TOKENIZER="$2";   shift 2 ;;
        --input-len)  INPUT_LEN="$2";   shift 2 ;;
        --output-len) OUTPUT_LEN="$2";  shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Clone InferenceX if not present ──────────────────────────────────────────
INFERENCEX_DIR="$REPO_ROOT/InferenceX"
if [ ! -d "$INFERENCEX_DIR" ]; then
    echo "Cloning SemiAnalysisAI/InferenceX@$INFERENCEX_COMMIT..."
    git clone https://github.com/SemiAnalysisAI/InferenceX.git "$INFERENCEX_DIR"
    git -C "$INFERENCEX_DIR" checkout "$INFERENCEX_COMMIT"
fi

BENCH_SCRIPT="$INFERENCEX_DIR/utils/bench_serving/benchmark_serving.py"
BENCH_DIR="$INFERENCEX_DIR/utils/bench_serving"

# ── Python env for benchmarking (use control plane venv — has transformers/tokenizers) ─
VENV="$REPO_ROOT/.venv_cp"
[ ! -d "$VENV" ] && { echo "ERROR: .venv_cp not found — run bash launch/control_plane.sh once to create it"; exit 1; }
PYTHON="$VENV/bin/python"

mkdir -p "$RESULT_DIR"
# Resolve to absolute — the script cd's into InferenceX's BENCH_DIR below, and
# benchmark_serving.py opens --result-dir relative to its own cwd, not the
# caller's. A relative path here silently ran the whole benchmark, printed
# correct results to stdout, then crashed writing the JSON summary.
RESULT_DIR=$(cd "$RESULT_DIR" && pwd)

echo "=== Benchmark ==="
echo "    endpoint:    $ENDPOINT"
echo "    model:       $MODEL"
echo "    ISL/OSL:     $INPUT_LEN / $OUTPUT_LEN"
echo "    concurrency: $CONCURRENCY"
echo "    num-prompts: $NUM_PROMPTS"
echo ""

cd "$BENCH_DIR"
"$PYTHON" benchmark_serving.py \
    --backend openai \
    --base-url "$ENDPOINT" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --max-concurrency "$CONCURRENCY" \
    --num-prompts "$NUM_PROMPTS" \
    --trust-remote-code \
    --ignore-eos \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "90,99,99.9"
