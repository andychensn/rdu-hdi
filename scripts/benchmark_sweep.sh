#!/usr/bin/env bash
# Run a full benchmark sweep: ISL × concurrency grid.
# Waits for RDU decode to register before starting.
#
# Usage:
#   bash scripts/benchmark_sweep.sh [--result-dir DIR] [--label LABEL]
#   LABEL defaults to "dynamo_docker"
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

LABEL="dynamo_docker"
RESULT_DIR="$REPO_ROOT/benchmark_results"
RDU_LOG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)      LABEL="$2";      shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --rdu-log)    RDU_LOG="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SWEEP_DIR="$RESULT_DIR/$LABEL"
mkdir -p "$SWEEP_DIR"

FRONTEND="http://$CONTROL_PLANE_IP:$VLLM_PORT"

# ── Wait for RDU decode to register ──────────────────────────────────────────
echo "Waiting for RDU decode to register..."
for i in $(seq 1 120); do
    if [ -n "$RDU_LOG" ] && grep -q "RDU worker registered" "$RDU_LOG" 2>/dev/null; then
        echo "  RDU worker registered (${i}×10s)"
        break
    fi
    # Fallback: check if frontend is serving (both workers up means RDU connected)
    # For now just poll the RDU log
    if [[ $i -eq 120 ]]; then
        echo "ERROR: RDU did not register within 20 min"
        [ -n "$RDU_LOG" ] && tail -20 "$RDU_LOG"
        exit 1
    fi
    [[ $((i % 6)) -eq 0 ]] && echo "  ${i}×10s..." && [ -n "$RDU_LOG" ] && tail -2 "$RDU_LOG" 2>/dev/null || true
    sleep 10
done

echo ""
echo "=== Starting sweep: ISL=1k/10k/100k × conc=1/2/4 ==="
echo "    results → $SWEEP_DIR"
echo ""

ISLS=(1000 10000 100000)
CONCS=(1 2 4)

for ISL in "${ISLS[@]}"; do
    for CONC in "${CONCS[@]}"; do
        NUM_PROMPTS=$(( CONC * 5 < 10 ? 10 : CONC * 5 ))
        TAG="isl${ISL}_conc${CONC}"
        echo "--- $TAG (num_prompts=$NUM_PROMPTS) ---"
        bash "$REPO_ROOT/scripts/benchmark.sh" \
            --input-len "$ISL" \
            --output-len 1000 \
            --concurrency "$CONC" \
            --num-prompts "$NUM_PROMPTS" \
            --result-dir "$SWEEP_DIR" \
            2>&1 | tee "$SWEEP_DIR/${TAG}.log"
        echo ""
    done
done

echo "=== Sweep complete ==="
echo "Results in $SWEEP_DIR:"
ls -la "$SWEEP_DIR/"*.json 2>/dev/null || echo "(no JSON files yet — check logs)"
