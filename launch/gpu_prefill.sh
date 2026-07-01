#!/usr/bin/env bash
# GPU prefill worker — submits SLURM job, waits for Dynamo registration.
# Runs via Docker (vllm/vllm-openai base + UCX/NIXL/patch baked in).
# Build the image first: bash scripts/build_docker_gpu.sh
#
# Usage: bash launch/gpu_prefill.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

GPU_CACHE_ROOT="$REPO_ROOT/.gpu_cache"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR" "$GPU_CACHE_ROOT"
TS=$(date +%Y%m%d_%H%M%S)
GPU_LOG="$LOG_DIR/${TS}_gpu_prefill.log"

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"backends":["UCX"]}}'

# ── Inner: runs ON the GPU node ───────────────────────────────────────────────
if [[ "${1:-}" == "--inner" ]]; then
    LOCAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep '^10\.17\.' | head -1 || true)
    LOCAL_IP=${LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}

    echo "=== GPU prefill (Docker) on $(hostname) ==="
    echo "    image:    $GPU_IMAGE"
    echo "    RoCE IP:  $LOCAL_IP"

    # Mount RDMA/IB devices so UCX can register GPU memory for RoCE NIXL transfer
    RDMA_DEVICES=""
    for dev in /dev/infiniband /dev/uverbs* /dev/nvidia-uvm /dev/nvidiactl; do
        [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
    done
    exec sudo -g docker /usr/bin/cuda-docker-run-wrapper \
        --pull=always \
        --net=host --rm \
        --entrypoint python3 \
        --ulimit memlock=-1 \
        $RDMA_DEVICES \
        -e "ETCD_ENDPOINTS=http://$CONTROL_PLANE_IP:$ETCD_PORT" \
        -e "NATS_SERVER=nats://$CONTROL_PLANE_IP:$NATS_PORT" \
        -e "DYN_REQUEST_PLANE=tcp" \
        -e "VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP" \
        -e "VLLM_NIXL_SIDE_CHANNEL_PORT=5600" \
        -e "VLLM_PD_CHUNK_OVERLAP=1" \
        -e "VLLM_PD_STAGE_TIMING=1" \
        -e "FLASHINFER_DISABLE_VERSION_CHECK=1" \
        -e "VLLM_USE_DEEP_GEMM=0" \
        -e "UCX_MODULE_DIR=/opt/ucx/lib/ucx" \
        -e "HF_HOME=$GPU_CACHE_ROOT/huggingface" \
        -e "VLLM_CACHE_ROOT=$GPU_CACHE_ROOT/vllm" \
        -e "TRITON_CACHE_DIR=$GPU_CACHE_ROOT/triton" \
        -e "TORCHINDUCTOR_CACHE_DIR=$GPU_CACHE_ROOT/inductor" \
        -e "FLASHINFER_WORKSPACE_BASE=$GPU_CACHE_ROOT/flashinfer" \
        -e "VLLM_CONFIG_ROOT=$GPU_CACHE_ROOT/vllm_config" \
        "$GPU_IMAGE" \
        -m dynamo.vllm \
            --model "$MODEL" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --disaggregation-mode prefill \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --block-size "$BLOCK_SIZE" \
            --no-enable-prefix-caching \
            --trust-remote-code \
            --kv-transfer-config "$KV_CONFIG"
fi

# ── Outer: submit SLURM job and wait for registration ─────────────────────────
echo "Submitting GPU prefill on $GPU_NODE (image: $GPU_IMAGE)..."
srun \
    -p "$GPU_PARTITION" -w "$GPU_NODE" \
    --gres="$GPU_GRES" \
    ${GPU_RESERVATION:+--reservation "$GPU_RESERVATION"} \
    -t "$GPU_TIME" \
    bash "$(readlink -f "${BASH_SOURCE[0]}")" --inner \
    > "$GPU_LOG" 2>&1 &
SRUN_PID=$!
echo "  srun PID=$SRUN_PID  log=$GPU_LOG"

FRONTEND="http://$CONTROL_PLANE_IP:$VLLM_PORT"
echo "  Waiting for GPU worker to register (up to ${GPU_PREFILL_REGISTER_TIMEOUT}s)..."
for i in $(seq 1 "$GPU_PREFILL_REGISTER_TIMEOUT"); do
    if curl -sf --max-time 3 "$FRONTEND/v1/models" 2>/dev/null | grep -q '"id"'; then
        echo "  GPU worker registered (${i}s)"
        echo "  $GPU_LOG"
        exit 0
    fi
    if ! kill -0 $SRUN_PID 2>/dev/null; then
        echo "ERROR: srun exited early. Log:"
        tail -30 "$GPU_LOG"
        exit 1
    fi
    [[ $((i % 30)) -eq 0 ]] && echo "  ${i}s..." && tail -2 "$GPU_LOG" 2>/dev/null || true
    sleep 1
done
echo "ERROR: GPU worker did not register within ${GPU_PREFILL_REGISTER_TIMEOUT}s"
tail -20 "$GPU_LOG"
exit 1
