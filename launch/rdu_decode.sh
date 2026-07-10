#!/usr/bin/env bash
# RDU decode worker — submits an snrdu job, waits for Dynamo registration.
# Runs via Docker (self-built coe_api/rdu_engine + BAR2 runtime baked in).
# Build the image first: bash docker/rdu/build.sh
#
# Usage: bash launch/rdu_decode.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RDU_LOG="$LOG_DIR/${TS}_rdu_decode.log"

# ── Inner: runs ON the RDU node (inside the snrdu allocation) ─────────────────
if [[ "${1:-}" == "--inner" ]]; then
    RDMA_DEVICES=""
    for dev in /dev/infiniband /dev/uverbs*; do
        [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
    done

    echo "=== starting persistent RDU decode container (self-built, fully baked-in) on $(hostname) $(date) ==="
    exec sudo -g docker /usr/bin/docker-run-wrapper --pull=always --net=host --rm \
        --name "rdu-decode-${SLURM_JOB_ID}" \
        --device /dev/rdu --device /dev/rdu_mem_map \
        $RDMA_DEVICES \
        --ulimit memlock=-1 \
        --cap-add IPC_LOCK \
        -e CONTROL_PLANE_IP="$CONTROL_PLANE_IP" \
        -e ETCD_PORT="$ETCD_PORT" \
        -e NATS_PORT="$NATS_PORT" \
        -e GPU_ROCE_IP="$GPU_ROCE_IP" \
        -e MODEL="$MODEL" \
        -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e PEF="$PEF" \
        -e MODEL_CONFIG="$MODEL_CONFIG" \
        -e GPU_PREFILL_BLOCK_SIZE="$BLOCK_SIZE" \
        -e VLLM_RDU_PLUGIN_TIME_PROFILE=1 \
        -e RDU_HDI_PROFILE_DIR="$REPO_ROOT/benchmark_results/rdu_traces" \
        "$RDU_IMAGE"
fi

# ── Outer: submit snrdu job and wait for registration ──────────────────────────
# NOTE: unlike launch/gpu_prefill.sh's frontend /v1/models poll, RDU readiness
# can't be checked that way -- GPU prefill's earlier registration already makes
# /v1/models succeed, so that check would report "ready" as soon as GPU comes
# up, before RDU decode does anything at all. Instead, grep this job's own log
# for the same registration line dynamo_runtime's _core module prints for any
# worker -- scoped to RDU decode's own log file, this is unambiguous.
#
# NOTE: `snrdu run` behaves like `sbatch`, not `srun`, despite the name --
# it prints "Submitted batch job N" and exits within a couple seconds,
# regardless of the job's own runtime. Backgrounding it and tracking its PID
# (the way gpu_prefill.sh tracks a real, attached `srun`) is therefore
# useless as a liveness check -- the PID is already gone almost immediately
# on a healthy submit. Track the real SLURM job ID via squeue instead.
echo "Submitting RDU decode on $RDU_NODE (image: $RDU_IMAGE)..."
SNRDU_OUT="$LOG_DIR/${TS}_rdu_decode_snrdu.out"
snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
    --allow-local-lib-python --reservation "$RDU_RESERVATION" \
    --pef "$PEF" --timeout "$RDU_TIMEOUT" \
    -o "$RDU_LOG" \
    -- bash "$(readlink -f "${BASH_SOURCE[0]}")" --inner > "$SNRDU_OUT" 2>&1 &

JOB_ID=""
for i in $(seq 1 30); do
    JOB_ID=$(grep -o 'Submitted batch job [0-9]*' "$SNRDU_OUT" 2>/dev/null | grep -o '[0-9]*$' || true)
    [ -n "$JOB_ID" ] && break
    sleep 1
done
if [ -z "$JOB_ID" ]; then
    echo "ERROR: snrdu did not report a job ID within 30s. Output:"
    cat "$SNRDU_OUT"
    exit 1
fi
echo "  snrdu job=$JOB_ID  log=$RDU_LOG"

echo "  Waiting for RDU worker to register (up to ${RDU_DECODE_REGISTER_TIMEOUT}s, ~12-14 min typical: BAR2/PEF init)..."
for i in $(seq 1 "$RDU_DECODE_REGISTER_TIMEOUT"); do
    if grep -q "Registered base model '$SERVED_MODEL_NAME' MDC" "$RDU_LOG" 2>/dev/null; then
        echo "  RDU worker registered (${i}s)"
        echo "  $RDU_LOG"
        exit 0
    fi
    if [ -z "$(squeue -j "$JOB_ID" -h 2>/dev/null)" ]; then
        echo "ERROR: SLURM job $JOB_ID is no longer running. Log:"
        tail -30 "$RDU_LOG"
        exit 1
    fi
    [[ $((i % 30)) -eq 0 ]] && echo "  ${i}s..." && tail -2 "$RDU_LOG" 2>/dev/null || true
    sleep 1
done
echo "ERROR: RDU worker did not register within ${RDU_DECODE_REGISTER_TIMEOUT}s"
tail -20 "$RDU_LOG"
exit 1
