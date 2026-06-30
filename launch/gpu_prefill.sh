#!/usr/bin/env bash
# GPU prefill worker — submits SLURM job, waits for Dynamo registration.
# Usage: bash launch/gpu_prefill.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"

GPU_VENV="$REPO_ROOT/.venv_gpu"
UCX_INSTALL="$REPO_ROOT/ucx-install"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
GPU_LOG="$LOG_DIR/${TS}_gpu_prefill.log"

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"backends":["UCX"]}}'

if [[ "${1:-}" == "--inner" ]]; then
    # ── Inner: runs ON the GPU node ─────────────────────────────────────────
    LOCAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep '^10\.17\.' | head -1 || true)
    LOCAL_IP=${LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}

    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    # vllm._C compiled for CUDA 12; include cu12 runtime + cu13 from torch + system
    PYVER=$(ls "$GPU_VENV/lib/" | grep "python3\." | head -1)
    CUDA12_LIBS="$GPU_VENV/lib/$PYVER/site-packages/nvidia/cuda_runtime/lib"
    export LD_LIBRARY_PATH="$CUDA12_LIBS:$UCX_INSTALL/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    # Add venv bin to PATH so triton/torch.compile can invoke ninja as subprocess
    export PATH="$GPU_VENV/bin:$PATH"

    echo "=== GPU prefill on $(hostname) ==="
    echo "    venv: $GPU_VENV"
    echo "    RoCE IP: $LOCAL_IP"
    "$GPU_VENV/bin/python" -c "import torch; print('    torch:', torch.__version__, 'cuda:', torch.version.cuda)"
    "$GPU_VENV/bin/python" -c "import vllm; print('    vllm:', vllm.__version__)"

    ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
    DYN_REQUEST_PLANE=tcp \
    VLLM_NIXL_SIDE_CHANNEL_HOST="$LOCAL_IP" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    VLLM_PD_CHUNK_OVERLAP=1 \
    VLLM_PD_STAGE_TIMING=1 \
    exec "$GPU_VENV/bin/python" -m dynamo.vllm \
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

# ── Outer: submit SLURM and wait ─────────────────────────────────────────────
echo "Submitting GPU prefill on $GPU_NODE..."
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
    [[ $((i % 30)) -eq 0 ]] && echo "  ${i}s elapsed..." && tail -2 "$GPU_LOG" 2>/dev/null || true
    sleep 1
done
echo "ERROR: GPU worker did not register within ${GPU_PREFILL_REGISTER_TIMEOUT}s"
tail -20 "$GPU_LOG"
exit 1
