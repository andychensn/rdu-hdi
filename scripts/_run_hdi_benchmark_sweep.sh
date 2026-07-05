#!/usr/bin/env bash
set -euo pipefail
cd /import/snvm-sc-scratch1/andyc/rdu-hdi

RESULT_DIR="benchmark_results/hdi_real_clean_20260705"
mkdir -p "$RESULT_DIR"
ENDPOINT="http://localhost:8192"

run() {
    echo "=== $(date) : ISL=$1 OSL=$2 conc=$3 prompts=$4 ==="
    bash scripts/benchmark.sh --endpoint "$ENDPOINT" --model MiniMax-M2.7 \
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
