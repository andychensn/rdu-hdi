#!/usr/bin/env bash
# GPU prefill worker on VCC's GPU_HOST. Single worker (1P1D) -- see
# vcc/cluster.env if you need to scale to 2P.
#
# Differs from launch/gpu_prefill.sh in three ways, all VCC infra facts, not
# design choices (see vcc/README.md for the investigation behind each):
#   1. No shared filesystem with the login node -- this script ships ITSELF
#      plus a resolved env file to GPU_HOST via scp, then runs via SSH,
#      instead of relying on config/*.env and this same script path already
#      being visible on the target node the way srun's shared-NFS model
#      allows.
#   2. No nvidia-container-toolkit/CDI on GPU_HOST -- GPU passthrough uses a
#      manual recipe instead (raw --device + --security-opt label=disable +
#      host driver libs copied into a mounted dir + LD_LIBRARY_PATH set at
#      launch). Confirmed live: real GPU compute (a 4096x4096 matmul) inside
#      rootless Podman with this recipe, on the B200 hardware, with vllm's
#      own torch build.
#   3. GPU_HOST's RoCE NICs are Mellanox mlx5, not Broadcom bnxt_re --
#      UCX_NET_DEVICES names the real local interface, not vnc+idc's list.
#
# LMCache (CPU-tier KV offload, see launch/gpu_prefill.sh) is intentionally
# NOT carried over here -- adds complexity with no evidence yet that VCC's
# demo workload needs it. Add it back the same way if that changes.
#
# Usage: bash vcc/launch/gpu_prefill.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)

# ── Inner: runs ON GPU_HOST (shipped over via scp, invoked over ssh) ────────
if [[ "${1:-}" == "--inner" ]]; then
    # shellcheck disable=SC1091
    source /tmp/vcc_gpu_prefill_env.sh

    # One-time GPU driver userspace lib injection (idempotent -- skipped on
    # every launch after the first). Copies (NOT symlinks -- an absolute-path
    # symlink resolves inside the CONTAINER's own rootfs and silently
    # fails) libcuda/libnvidia-ml/libnvidia-ptxjitcompiler out of the host's
    # driver install into a plain directory bind-mounted into the container.
    [ -e "$GPU_MODEL_PATH" ] || { echo "ERROR: $GPU_MODEL_PATH not found on $(hostname)"; exit 1; }

    if [ ! -e "$GPU_DRIVER_LIBS_DIR/libcuda.so.1" ]; then
        echo "First run on $(hostname): copying NVIDIA driver userspace libs into $GPU_DRIVER_LIBS_DIR..."
        mkdir -p "$GPU_DRIVER_LIBS_DIR"
        for lib in libcuda.so libnvidia-ml.so libnvidia-ptxjitcompiler.so; do
            src=$(ldconfig -p 2>/dev/null | grep "^\s*${lib}\." | head -1 | awk '{print $NF}')
            [ -n "$src" ] || { echo "ERROR: $lib not found via ldconfig -p on this host"; exit 1; }
            cp -L "$src" "$GPU_DRIVER_LIBS_DIR/$(basename "$src")"
        done
        # Create the unversioned + .1 names the dynamic linker/torch look for,
        # pointing at whatever exact versioned file we just copied.
        for lib in libcuda.so libnvidia-ml.so libnvidia-ptxjitcompiler.so; do
            real=$(find "$GPU_DRIVER_LIBS_DIR" -maxdepth 1 -name "${lib}.*" | sort -V | tail -1)
            [ -n "$real" ] || continue
            ln -sf "$(basename "$real")" "$GPU_DRIVER_LIBS_DIR/${lib}.1"
            ln -sf "$(basename "$real")" "$GPU_DRIVER_LIBS_DIR/${lib}"
        done
    fi

    GPU_DEVICE_FLAGS=""
    for dev in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
        [ -e "$dev" ] && GPU_DEVICE_FLAGS="$GPU_DEVICE_FLAGS --device $dev"
    done
    RDMA_DEVICES=""
    for dev in /dev/infiniband/*; do
        [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
    done

    echo "=== GPU prefill (VCC/Podman) on $(hostname) $(date) ==="
    echo "    image:      $GPU_IMAGE"
    echo "    RoCE IP:    $GPU_ROCE_IP"
    echo "    devices:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    # docker-run-wrapper/cuda-docker-run-wrapper (vnc+idc) auto-mount
    # /import,/scratch into the container -- plain `podman run` here does
    # not, and --net=host only shares the network namespace. Mount
    # GPU_MODEL_PATH explicitly.
    exec podman run --rm --net=host \
        --name "vcc-gpu-prefill" \
        --entrypoint python3 \
        --security-opt label=disable \
        --ulimit memlock=-1:-1 \
        --shm-size=1g \
        $GPU_DEVICE_FLAGS \
        $RDMA_DEVICES \
        -v "$GPU_DRIVER_LIBS_DIR:$GPU_DRIVER_LIBS_DIR:ro" \
        -v "$GPU_MODEL_PATH:$GPU_MODEL_PATH:ro" \
        -e "LD_LIBRARY_PATH=$GPU_DRIVER_LIBS_DIR" \
        -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
        -e "ETCD_ENDPOINTS=http://$CONTROL_PLANE_IP:$ETCD_PORT" \
        -e "NATS_SERVER=nats://$CONTROL_PLANE_IP:$NATS_PORT" \
        -e "VLLM_NIXL_SIDE_CHANNEL_HOST=$GPU_ROCE_IP" \
        -e "VLLM_NIXL_SIDE_CHANNEL_PORT=$GPU_NIXL_PORT" \
        -e "NCCL_IB_DISABLE=1" \
        -e "NCCL_P2P_LEVEL=NVL" \
        -e "UCX_TLS=rc,cuda_copy,cuda_ipc" \
        -e "UCX_NET_DEVICES=$UCX_NET_DEVICES" \
        -e "UCX_MAX_RNDV_RAILS=1" \
        -e "UCX_IB_ROCE_REACHABILITY_MODE=all" \
        -e "HF_HOME=$GPU_CACHE_ROOT/huggingface" \
        -e "VLLM_CACHE_ROOT=$GPU_CACHE_ROOT/vllm" \
        -e "TRITON_CACHE_DIR=$GPU_CACHE_ROOT/triton" \
        -e "TORCHINDUCTOR_CACHE_DIR=$GPU_CACHE_ROOT/inductor" \
        -e "FLASHINFER_WORKSPACE_BASE=$GPU_CACHE_ROOT/flashinfer" \
        -e "VLLM_CONFIG_ROOT=$GPU_CACHE_ROOT/vllm_config" \
        "$GPU_IMAGE" \
        -m dynamo.vllm \
            --model "$GPU_MODEL_PATH" \
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
            --kv-transfer-config "$KV_CONFIG" \
            --kv-events-config "$KV_EVENTS_CONFIG"
fi

# ── Outer: resolve config on the login node, ship to GPU_HOST, launch, wait ──
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/vcc/cluster.env"
source "$REPO_ROOT/vcc/model.env"

LOG_DIR="$REPO_ROOT/vcc/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
GPU_LOG="$LOG_DIR/${TS}_gpu_prefill.log"

KV_EVENTS_CONFIG="{\"enable_kv_cache_events\": true, \"endpoint\": \"tcp://*:${GPU_KV_EVENTS_PORT}\"}"
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"backends":["UCX"]}}'

END=$((GPU_DEVICE_START + TENSOR_PARALLEL_SIZE - 1))
CUDA_VISIBLE_DEVICES=$(seq -s, "$GPU_DEVICE_START" "$END")

# GPU_HOST's RoCE NICs are Mellanox mlx5 (confirmed 2026-07-15 via
# ibv_devinfo), not Broadcom bnxt_re -- mlx5_1 is the HCA whose active RoCE
# port resolved to $GPU_ROCE_IP in the survey. Override via env if your pair
# of nodes differs.
UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_1:1}"

ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT
cat > "$ENV_FILE" <<EOF
GPU_IMAGE=$GPU_IMAGE
CONTROL_PLANE_IP=$CONTROL_PLANE_IP
ETCD_PORT=$ETCD_PORT
NATS_PORT=$NATS_PORT
GPU_ROCE_IP=$GPU_ROCE_IP
GPU_NIXL_PORT=$GPU_NIXL_PORT
UCX_NET_DEVICES=$UCX_NET_DEVICES
GPU_MODEL_PATH=$GPU_MODEL_PATH
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
MAX_MODEL_LEN=$MAX_MODEL_LEN
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
BLOCK_SIZE=$BLOCK_SIZE
KV_CONFIG=$(printf '%q' "$KV_CONFIG")
KV_EVENTS_CONFIG=$(printf '%q' "$KV_EVENTS_CONFIG")
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPU_DRIVER_LIBS_DIR=$GPU_DRIVER_LIBS_DIR
GPU_CACHE_ROOT=/home/andyc/.vcc-gpu-cache
EOF

echo "Shipping launch script + config to $GPU_HOST..."
scp -q -o BatchMode=yes "$(readlink -f "${BASH_SOURCE[0]}")" "$GPU_HOST:/tmp/vcc_gpu_prefill.sh"
scp -q -o BatchMode=yes "$ENV_FILE" "$GPU_HOST:/tmp/vcc_gpu_prefill_env.sh"

echo "Launching GPU prefill on $GPU_HOST (image: $GPU_IMAGE)..."
ssh -o BatchMode=yes "$GPU_HOST" "bash /tmp/vcc_gpu_prefill.sh --inner" > "$GPU_LOG" 2>&1 &
SSH_PID=$!
echo "  log=$GPU_LOG  ssh_pid=$SSH_PID"

teardown() {
    echo "  Tearing down GPU prefill..."
    ssh -o BatchMode=yes "$GPU_HOST" "podman stop -t 5 vcc-gpu-prefill" 2>/dev/null || true
    kill "$SSH_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}

echo "  Waiting for GPU prefill to register (up to ${GPU_PREFILL_REGISTER_TIMEOUT}s)..."
for t in $(seq 1 "$GPU_PREFILL_REGISTER_TIMEOUT"); do
    if grep -q "Registered base model '$SERVED_MODEL_NAME' MDC" "$GPU_LOG" 2>/dev/null; then
        echo "  GPU prefill registered (${t}s)"
        echo "  $GPU_LOG"
        exit 0
    fi
    if ! kill -0 "$SSH_PID" 2>/dev/null; then
        echo "ERROR: GPU prefill's ssh session exited early before registering. Log:"
        tail -30 "$GPU_LOG"
        teardown
        exit 1
    fi
    [[ $((t % 30)) -eq 0 ]] && echo "  ${t}s..." && tail -2 "$GPU_LOG" 2>/dev/null
    sleep 1
done
echo "ERROR: GPU prefill did not register within ${GPU_PREFILL_REGISTER_TIMEOUT}s"
tail -20 "$GPU_LOG"
teardown
exit 1
