#!/usr/bin/env bash
# Standard 9-config sweep (ISL 1k/10k/100k x concurrency 1/2/4). Parameterized
# so the same script can target any endpoint/model/result-dir -- point
# --endpoint/--model/--label at whichever stack you want to measure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENDPOINT=""
MODEL=""
LABEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --endpoint) ENDPOINT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[ -n "$LABEL" ] || { echo "ERROR: --label is required (used to name the result dir)"; exit 1; }

RESULT_DIR="benchmark_results/${LABEL}"
mkdir -p "$RESULT_DIR"

EXTRA_ARGS=()
[ -n "$ENDPOINT" ] && EXTRA_ARGS+=(--endpoint "$ENDPOINT")
[ -n "$MODEL" ] && EXTRA_ARGS+=(--model "$MODEL")

run() {
    echo "=== $(date) : ISL=$1 OSL=$2 conc=$3 prompts=$4 ==="
    bash scripts/benchmark.sh "${EXTRA_ARGS[@]}" \
        --input-len "$1" --output-len "$2" --concurrency "$3" --num-prompts "$4" --result-dir "$RESULT_DIR"
    echo "=== done: ISL=$1 OSL=$2 conc=$3 $(date) ==="
    echo ""
}

run 1000   1000 1 10
run 1000   1000 2 20
run 1000   1000 4 40
run 10000  1000 1 10
run 10000  1000 2 20
run 10000  1000 4 40
run 100000 1000 1 10
run 100000 1000 2 10
run 100000 1000 4 10

echo "=== ALL 9 CONFIGS COMPLETE $(date) ==="
