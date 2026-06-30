#!/usr/bin/env bash
# RDU decode worker — submits SLURM job via snrdu, waits for Dynamo registration.
# Usage: bash launch/rdu_decode.sh
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

RDU_VENV="${RDU_VENV:-$REPO_ROOT/.venv_rdu}"
RDU_CACHE="${RDU_CACHE:-$REPO_ROOT/.rdu_cache}"
LOG_DIR="$REPO_ROOT/logs"

# ── Inner: runs ON the RDU node inside the snrdu job ─────────────────────────
if [[ "${1:-}" == "--inner" ]]; then
    mkdir -p "$RDU_CACHE"

    # Hardware paths — BAR2/SambaFlow are SambaNova system deps (like CUDA, set by snrdu)
    SOFTWARE_BUILD=${SOFTWARE_BUILD:-/import/snvm-sc-scratch1/guoyaof/software/runtime/build}
    BAR2_INSTALL=${BAR2_INSTALL:-/import/snvm-sc-scratch2/jayr/sambaflow_gTkgyGCEBH/bazel-install}
    BAR2_RUNTIME_LIBS=${BAR2_RUNTIME_LIBS:-/import/snvm-sc-scratch1/guoyaof/sw_ddr_rdma/runtime/build/graph/lib}
    BAR2_PRELOAD=${BAR2_PRELOAD:-/import/snvm-sc-scratch1/guoyaof/sw_ddr_rdma_install/bar2_preload_libs}

    # UCX/NIXL: prefer repo-built; fall back to SOFTWARE_BUILD
    RDU_UCX="${RDU_UCX:-$REPO_ROOT/rdu-ucx-install}"
    if [ -d "$RDU_UCX/lib" ]; then
        _UCX_LIB="$RDU_UCX/lib"
        _UCX_MOD="$RDU_UCX/lib/ucx"
        _NIXL_LIB="$RDU_UCX/../wheelhouse"
    else
        _UCX_LIB="$SOFTWARE_BUILD/ucx-install/lib"
        _UCX_MOD="$SOFTWARE_BUILD/ucx-install/lib/ucx"
        _NIXL_LIB="$SOFTWARE_BUILD/nixl-install/lib"
    fi

    RDU_ROCE_IP_LOCAL=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep '^10\.17\.' | head -1 || true)
    RDU_ROCE_IP_LOCAL=${RDU_ROCE_IP_LOCAL:-$(hostname -I 2>/dev/null | awk '{print $1}')}

    echo "=== RDU decode on $(hostname) $(date) ==="
    echo "    etcd:        http://$CONTROL_PLANE_IP:$ETCD_PORT"
    echo "    producer:    $GPU_ROCE_IP:5600"
    echo "    side-channel: $RDU_ROCE_IP_LOCAL:5600"

    source "$RDU_VENV/bin/activate"

    RDU_CONFIG="${MODEL_CONFIG:-}"
    [ -n "$RDU_CONFIG" ] && [ ! -f "$RDU_CONFIG" ] && {
        echo "WARNING: MODEL_CONFIG=$RDU_CONFIG not found — running without model config"
        RDU_CONFIG=""
    }

    PYTHONNOUSERSITE=1 \
    ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
    DYN_REQUEST_PLANE=tcp \
    HF_HUB_OFFLINE=1 \
    VLLM_NIXL_SIDE_CHANNEL_HOST="$RDU_ROCE_IP_LOCAL" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    VLLM_PD_CHUNK_OVERLAP=1 \
    VLLM_PD_PRODUCER_CHUNK_OVERLAP=1 \
    VLLM_PD_PRODUCER_HOST="$GPU_ROCE_IP" \
    VLLM_PD_PRODUCER_PORT=5600 \
    SN_REMOTE_KV_MEMTYPE=VRAM \
    RAW_FORCE_PER_XFER=1 \
    SN_MULTI_AGENT_NICS=bnxt_re0,bnxt_re2,bnxt_re4,bnxt_re6 \
    VLLM_CPU_KVCACHE_SPACE=200 \
    VLLM_CPU_OMP_THREADS_BIND=nobind \
    UCX_TLS=rc,tcp \
    UCX_NET_DEVICES=bnxt_re0:1,bnxt_re2:1,bnxt_re4:1,bnxt_re6:1 \
    UCX_MAX_RNDV_RAILS=4 \
    UCX_MAX_RMA_RAILS=1 \
    UCX_MULTI_LANE_MAX_RATIO=100 \
    UCX_IB_PREFER_NEAREST_DEVICE=n \
    UCX_IB_ROCE_REACHABILITY_MODE=all \
    UCX_RCACHE_MAX_UNRELEASED=1024 \
    UCX_LOG_LEVEL=WARN \
    UCX_MODULE_DIR="$_UCX_MOD" \
    NIXL_PLUGIN_DIR="$_NIXL_LIB" \
    INFERENCE_MODE=local_queue \
    PROG_LOAD=HBM \
    ARG_LOAD=HBM \
    ENABLE_STRICT_CONVERSION=1 \
    SF_RNT_FSM_POLL_BUSY_WAIT=1 \
    SF_RNT_DMA_POLL_BUSY_WAIT=1 \
    SF_RNT_NUMA_BIND=2 \
    SF_RNT_LOG_LEVEL=ERR \
    AL_EXEC_LOOPING=1 \
    GRAPH_TIMEOUT_USEC=120000000 \
    TRITON_CACHE_DIR="$RDU_CACHE/triton" \
    VLLM_CACHE_ROOT="$RDU_CACHE/vllm" \
    HF_HOME="$RDU_CACHE/huggingface" \
    VLLM_CONFIG_ROOT="$RDU_CACHE/vllm_config" \
    TRANSFORMERS_CACHE="$RDU_CACHE/huggingface" \
    PYTHONPATH="$BAR2_INSTALL/python:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$_UCX_LIB:$_NIXL_LIB:$SOFTWARE_BUILD/etcd-cpp-api-install/lib:$SOFTWARE_BUILD/gflags-install/lib:$BAR2_RUNTIME_LIBS:$BAR2_INSTALL/lib:${LD_LIBRARY_PATH:-}" \
    LD_PRELOAD="$BAR2_PRELOAD/libc_samba_runtime.so:$BAR2_PRELOAD/libcpp_samba_runtime.so${LD_PRELOAD:+:$LD_PRELOAD}" \
        exec python -m dynamo.vllm \
            --model "$MODEL" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --disaggregation-mode decode \
            --load-format dummy \
            --max-num-seqs 2 \
            --tensor-parallel-size 1 \
            --no-enable-prefix-caching \
            --max-model-len "$MAX_MODEL_LEN" \
            --compilation-config '{"mode": 0}' \
            ${RDU_CONFIG:+--additional-config "{\"rdu_config\": \"$RDU_CONFIG\"}"} \
            --trust-remote-code \
            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"rdu","kv_connector_extra_config":{"rdu_mode":"real","rdu_ddr_cache_budget_gb":30,"backends":["UCX"],"enforce_handshake_compat":false}}'
fi

# ── Outer: wait for GPU, submit via snrdu, poll for registration ──────────────
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RDU_LOG="$LOG_DIR/${TS}_rdu_decode.log"

[ -z "${PEF:-}" ] && { echo "ERROR: PEF not set — check config/model.env"; exit 1; }

FRONTEND="http://$CONTROL_PLANE_IP:$VLLM_PORT"
echo "Waiting for GPU prefill to register at $FRONTEND..."
for i in $(seq 1 120); do
    if curl -sf --max-time 3 "$FRONTEND/v1/models" 2>/dev/null | grep -q '"id"'; then
        echo "  GPU prefill registered (${i}×5s)"
        break
    fi
    [ "$i" -eq 120 ] && { echo "ERROR: GPU prefill did not register within 10 min."; exit 1; }
    printf "."
    sleep 5
done
echo ""

echo "Submitting RDU decode on $RDU_NODE..."
snrdu run \
    -sp "$RDU_PARTITION" \
    --qos "$RDU_QOS" \
    --nodelist "$RDU_NODE" \
    --allow-local-lib-python \
    ${RDU_RESERVATION:+--reservation "$RDU_RESERVATION"} \
    --pef "$PEF" \
    --timeout "$RDU_TIMEOUT" \
    -o "$RDU_LOG" \
    -- bash "$(readlink -f "${BASH_SOURCE[0]}")" --inner &
SNRDU_PID=$!
echo "  snrdu PID=$SNRDU_PID  log=$RDU_LOG"

echo "  Waiting for RDU worker to register (~12 min for BAR2 init)..."
for i in $(seq 1 90); do
    printf ".(${i}×10s)"
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
