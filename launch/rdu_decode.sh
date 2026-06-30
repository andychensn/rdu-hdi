#!/usr/bin/env bash
# RDU decode worker — submits via snrdu, waits for Dynamo registration.
# Usage: bash launch/rdu_decode.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

RDU_VENV="$REPO_ROOT/.venv_rdu"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RDU_LOG="$LOG_DIR/${TS}_rdu_decode.log"

[ -z "${PEF:-}" ] && { echo "ERROR: PEF must be set (path to .pef file)"; exit 1; }

# ── Hardware paths (set by snrdu/SambaNova environment on s339) ─────────────
SOFTWARE_BUILD=${SOFTWARE_BUILD:-/import/snvm-sc-scratch1/guoyaof/software/runtime/build}
BAR2_INSTALL=${BAR2_INSTALL:-/import/snvm-sc-scratch2/jayr/sambaflow_gTkgyGCEBH/bazel-install}
BAR2_RUNTIME_LIBS=${BAR2_RUNTIME_LIBS:-/import/snvm-sc-scratch1/guoyaof/sw_ddr_rdma/runtime/build/graph/lib}
BAR2_PRELOAD=${BAR2_PRELOAD:-/import/snvm-sc-scratch1/guoyaof/sw_ddr_rdma_install/bar2_preload_libs}

# Cache dirs
RDU_CACHE=${RDU_CACHE:-$REPO_ROOT/.rdu_cache}
mkdir -p "$RDU_CACHE"

# ── Wait for GPU prefill to register first ───────────────────────────────────
# If RDU starts before GPU, Dynamo builds an aggregated (not disaggregated)
# pipeline and requests fail silently. Always start GPU prefill first.
FRONTEND="http://$CONTROL_PLANE_IP:$VLLM_PORT"
echo "Waiting for GPU prefill worker to register at $FRONTEND..."
for i in $(seq 1 120); do
    if curl -sf --max-time 3 "$FRONTEND/v1/models" 2>/dev/null | grep -q '"id"'; then
        echo "  GPU prefill registered — proceeding to launch RDU (${i}×5s)"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: GPU prefill did not register within 10 min."
        echo "  Start 'bash launch/gpu_prefill.sh' before launching RDU."
        exit 1
    fi
    printf "."
    sleep 5
done
echo ""

# ── snrdu submission ──────────────────────────────────────────────────────────
echo "Submitting RDU decode on $RDU_NODE..."
snrdu run \
    -sp "$RDU_PARTITION" \
    --qos "$RDU_QOS" \
    --nodelist "$RDU_NODE" \
    --allow-local-lib-python \
    --reservation "$RDU_RESERVATION" \
    --pef "$PEF" \
    --timeout "$RDU_TIMEOUT" \
    -o "$RDU_LOG" \
    -- bash "$REPO_ROOT/launch/rdu_inner.sh" &
SNRDU_PID=$!
echo "  snrdu PID=$SNRDU_PID  log=$RDU_LOG"

echo "  Waiting for RDU worker to register (~12 min for BAR2 init)..."
for i in $(seq 1 90); do
    echo -n ".(${i}×10s)"
    if grep -q "dynamo.backend.generate" "$RDU_LOG" 2>/dev/null; then
        echo ""
        echo "  RDU worker registered (${i}×10s)"
        echo "  $RDU_LOG"
        exit 0
    fi
    sleep 10
done
echo ""
echo "ERROR: RDU worker did not register within 900s"
tail -20 "$RDU_LOG"
exit 1
