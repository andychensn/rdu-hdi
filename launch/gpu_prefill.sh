#!/usr/bin/env bash
# GPU prefill worker(s) — submits one SLURM job per configured worker, waits for
# each to register with Dynamo. Runs via Docker (vllm/vllm-openai base +
# UCX/NIXL/patch baked in).
# Build the image first: bash docker/gpu/build.sh
#
# Usage: bash launch/gpu_prefill.sh
#   Number/placement of workers is controlled by GPU_NODES/GPU_RESERVATIONS in
#   config/cluster.env — one entry per worker, index-paired (same node repeated
#   colocates workers on that node; empty-string reservation omits --reservation
#   for that worker).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

GPU_CACHE_ROOT="$REPO_ROOT/.gpu_cache"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR" "$GPU_CACHE_ROOT"
TS=$(date +%Y%m%d_%H%M%S)

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"backends":["UCX"]}}'

# ── Inner: runs ON the GPU node ───────────────────────────────────────────────
if [[ "${1:-}" == "--inner" ]]; then
    IDX="${2:?worker index required}"
    LOCAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep '^10\.17\.' | head -1 || true)
    LOCAL_IP=${LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
    # Distinct NIXL side-channel port per worker: vLLM defaults every worker to
    # the same fixed port (GPU_NIXL_BASE_PORT) when data_parallel_index=0, which
    # collides if two workers share a node under --net=host.
    NIXL_PORT=$((GPU_NIXL_BASE_PORT + IDX))

    echo "=== GPU prefill worker $IDX (Docker) on $(hostname) ==="
    echo "    image:      $GPU_IMAGE"
    echo "    RoCE IP:    $LOCAL_IP"
    echo "    NIXL port:  $NIXL_PORT"

    # Mount RDMA/IB devices so UCX can register GPU memory for RoCE NIXL transfer
    RDMA_DEVICES=""
    for dev in /dev/infiniband /dev/uverbs* /dev/nvidia-uvm /dev/nvidiactl; do
        [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
    done
    exec sudo -g docker /usr/bin/cuda-docker-run-wrapper \
        --pull=always \
        --net=host --rm \
        --name "gpu-prefill-${IDX}" \
        --entrypoint python3 \
        --ulimit memlock=-1 \
        $RDMA_DEVICES \
        -e "ETCD_ENDPOINTS=http://$CONTROL_PLANE_IP:$ETCD_PORT" \
        -e "NATS_SERVER=nats://$CONTROL_PLANE_IP:$NATS_PORT" \
        -e "VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP" \
        -e "VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT" \
        -e "NCCL_IB_DISABLE=1" \
        -e "NCCL_P2P_LEVEL=NVL" \
        --shm-size=1g \
        -e "UCX_MODULE_DIR=/opt/ucx/lib/ucx" \
        -e "UCX_TLS=rc,cuda_copy,cuda_ipc" \
        -e "UCX_NET_DEVICES=bnxt_re0:1" \
        -e "UCX_IB_ROCE_REACHABILITY_MODE=all" \
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
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --block-size "$BLOCK_SIZE" \
            --enable-prefix-caching \
            --reasoning-parser minimax_m2_append_think \
            --trust-remote-code \
            --kv-transfer-config "$KV_CONFIG"
fi

# ── Outer: submit one SLURM job per configured worker, wait for all to register ──
NUM_WORKERS=${#GPU_NODES[@]}
echo "Submitting $NUM_WORKERS GPU prefill worker(s): ${GPU_NODES[*]}"

SRUN_PIDS=()
GPU_LOGS=()
JOB_NAMES=()

for i in "${!GPU_NODES[@]}"; do
    node="${GPU_NODES[$i]}"
    reservation="${GPU_RESERVATIONS[$i]:-}"
    GPU_LOG="$LOG_DIR/${TS}_gpu_prefill_${i}.log"
    GPU_LOGS+=("$GPU_LOG")
    JOB_NAME="gpu-prefill-${i}-${TS}"
    JOB_NAMES+=("$JOB_NAME")
    echo "  worker $i: node=$node reservation=${reservation:-<none>} log=$GPU_LOG job-name=$JOB_NAME"
    srun \
        -p "$GPU_PARTITION" -w "$node" \
        --gres="$GPU_GRES" \
        --mem="$GPU_MEM" \
        --job-name="$JOB_NAME" \
        ${reservation:+--reservation "$reservation"} \
        -t "$GPU_TIME" \
        bash "$(readlink -f "${BASH_SOURCE[0]}")" --inner "$i" \
        > "$GPU_LOG" 2>&1 &
    SRUN_PIDS+=("$!")
done
echo "  srun PIDs: ${SRUN_PIDS[*]}"

# Fail-fast teardown. scancel-by-job-name is the primary mechanism (targets
# exactly this launch's jobs, not scancel-everything-for-the-user); killing
# the local srun client PID is a best-effort fallback on top. Per
# docs/local/XPYD_SCALING_DESIGN.md §5.1, scancel has NOT reliably stopped a
# worker's foreground Docker container in this repo before -- this needs
# empirical confirmation (check `docker ps`/`nvidia-smi` on the node after a
# real teardown), and may need a container-level kill mechanism added if it
# doesn't hold up.
teardown_all() {
    echo "  Tearing down all worker(s) from this launch..."
    for job_name in "${JOB_NAMES[@]}"; do
        echo "    scancel --name=$job_name"
        scancel --name="$job_name" 2>/dev/null || true
    done
    for pid in "${SRUN_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
}

REGISTERED=()
for i in "${!GPU_NODES[@]}"; do REGISTERED[$i]=0; done

echo "  Waiting for all $NUM_WORKERS worker(s) to register (up to ${GPU_PREFILL_REGISTER_TIMEOUT}s each)..."
FAILED=0
for t in $(seq 1 "$GPU_PREFILL_REGISTER_TIMEOUT"); do
    all_done=1
    for i in "${!GPU_NODES[@]}"; do
        if [[ "${REGISTERED[$i]}" -eq 1 ]]; then
            continue
        fi
        if grep -q "Registered base model '$SERVED_MODEL_NAME' MDC" "${GPU_LOGS[$i]}" 2>/dev/null; then
            echo "  worker $i registered (${t}s)"
            REGISTERED[$i]=1
            continue
        fi
        if ! kill -0 "${SRUN_PIDS[$i]}" 2>/dev/null; then
            echo "ERROR: worker $i's srun exited early before registering. Log:"
            tail -30 "${GPU_LOGS[$i]}"
            FAILED=1
        fi
        all_done=0
    done
    if [[ "$FAILED" -eq 1 ]]; then
        break
    fi
    if [[ "$all_done" -eq 1 ]]; then
        break
    fi
    if [[ $((t % 30)) -eq 0 ]]; then
        reg_count=0
        for r in "${REGISTERED[@]}"; do
            if [[ "$r" -eq 1 ]]; then
                reg_count=$((reg_count + 1))
            fi
        done
        echo "  ${t}s... ($reg_count of $NUM_WORKERS registered)"
    fi
    sleep 1
done

if [[ "$FAILED" -eq 1 ]]; then
    teardown_all
    exit 1
fi

for i in "${!GPU_NODES[@]}"; do
    if [[ "${REGISTERED[$i]}" -ne 1 ]]; then
        echo "ERROR: worker $i did not register within ${GPU_PREFILL_REGISTER_TIMEOUT}s. Log:"
        tail -20 "${GPU_LOGS[$i]}"
        teardown_all
        exit 1
    fi
done

echo "  All $NUM_WORKERS GPU prefill worker(s) registered:"
for i in "${!GPU_LOGS[@]}"; do
    echo "    worker $i: ${GPU_LOGS[$i]}"
done
