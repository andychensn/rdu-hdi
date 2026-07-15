#!/usr/bin/env bash
# RDU decode worker on VCC's RDU_HOST. Differs from launch/rdu_decode.sh in
# the same three infra-fact ways as vcc/launch/gpu_prefill.sh (no shared
# filesystem -> ship script+env via scp; no SLURM -> direct ssh instead of
# snrdu; no dockerd -> rootless Podman instead of docker-run-wrapper), plus
# two RDU-specific ones:
#   1. SELinux is Enforcing on RDU_HOST -- confirmed live that
#      --device /dev/rdu passthrough is silently denied without
#      --security-opt label=disable (host perms are already 0666; this is
#      purely an SELinux relabeling issue, same class as GPU_HOST's, not a
#      root/toolkit problem).
#   2. docker/rdu/entrypoint.sh's RoCE-IP autodetection is hardcoded to the
#      vnc+idc cluster's 10.17.0.0/16 convention -- RDU_HOST's fabric is
#      172.16.0.0/24, so RDU_ROCE_IP_OVERRIDE is passed explicitly (see that
#      file's own comment; this needs the image rebuilt once to pick up,
#      it's a new entrypoint.sh env-var check, not yet in any already-built
#      image tag).
#
# Usage: bash vcc/launch/rdu_decode.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)

# ── Inner: runs ON RDU_HOST (shipped over via scp, invoked over ssh) ────────
if [[ "${1:-}" == "--inner" ]]; then
    # shellcheck disable=SC1091
    source /tmp/vcc_rdu_decode_env.sh

    for p in "$RDU_MODEL_PATH" "$RDU_PEF_PATH" "$MODEL_CONFIG_PATH"; do
        [ -e "$p" ] || { echo "ERROR: $p not found on $(hostname) -- see vcc/README.md's PEF staging section"; exit 1; }
    done

    RDMA_DEVICES=""
    for dev in /dev/infiniband/*; do
        [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
    done

    # docker-run-wrapper (vnc+idc) auto-mounts /import,/scratch into the
    # container -- plain `podman run` here does no such thing, and --net=host
    # only shares the network namespace, not the filesystem. Every host path
    # the container needs to read must be bind-mounted explicitly.
    echo "=== starting RDU decode (VCC/Podman) on $(hostname) $(date) ==="
    exec podman run --rm --net=host \
        --name "vcc-rdu-decode" \
        --security-opt label=disable \
        --device /dev/rdu --device /dev/rdu_mem_map \
        $RDMA_DEVICES \
        --ulimit memlock=-1:-1 \
        --cap-add IPC_LOCK \
        -v "$RDU_MODEL_PATH:$RDU_MODEL_PATH:ro" \
        -v "$(dirname "$RDU_PEF_PATH"):$(dirname "$RDU_PEF_PATH"):ro" \
        -v "$MODEL_CONFIG_PATH:$MODEL_CONFIG_PATH:ro" \
        -e CONTROL_PLANE_IP="$CONTROL_PLANE_IP" \
        -e ETCD_PORT="$ETCD_PORT" \
        -e NATS_PORT="$NATS_PORT" \
        -e GPU_ROCE_IP="$GPU_ROCE_IP" \
        -e RDU_ROCE_IP_OVERRIDE="$RDU_ROCE_IP" \
        -e MODEL="$RDU_MODEL_PATH" \
        -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e PEF="$RDU_PEF_PATH" \
        -e MODEL_CONFIG="$MODEL_CONFIG_PATH" \
        -e GPU_PREFILL_BLOCK_SIZE="$BLOCK_SIZE" \
        -e VLLM_RDU_PLUGIN_TIME_PROFILE=1 \
        "$RDU_IMAGE"
fi

# ── Outer: resolve config, ship the adjusted model YAML + env, launch, wait ──
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/vcc/cluster.env"
source "$REPO_ROOT/vcc/model.env"

LOG_DIR="$REPO_ROOT/vcc/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RDU_LOG="$LOG_DIR/${TS}_rdu_decode.log"

# config/minimax_m2.yaml's checkpoint_path already matches RDU_MODEL_PATH
# exactly (both /scratch/MiniMax-M2.7-FP8-RDU-packed) -- only pef_path needs
# rewriting to wherever the PEF actually lives on RDU_HOST.
ADJUSTED_YAML="$(mktemp)"
trap 'rm -f "$ADJUSTED_YAML" "$ENV_FILE"' EXIT
sed "s|^pef_path:.*|pef_path: \"$RDU_PEF_PATH\"|" "$REPO_ROOT/config/minimax_m2.yaml" > "$ADJUSTED_YAML"
REMOTE_MODEL_CONFIG="/tmp/vcc_${MODEL_CONFIG_NAME}"

ENV_FILE="$(mktemp)"
cat > "$ENV_FILE" <<EOF
RDU_IMAGE=$RDU_IMAGE
CONTROL_PLANE_IP=$CONTROL_PLANE_IP
ETCD_PORT=$ETCD_PORT
NATS_PORT=$NATS_PORT
GPU_ROCE_IP=$GPU_ROCE_IP
RDU_ROCE_IP=$RDU_ROCE_IP
RDU_MODEL_PATH=$RDU_MODEL_PATH
RDU_PEF_PATH=$RDU_PEF_PATH
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
MAX_MODEL_LEN=$MAX_MODEL_LEN
BLOCK_SIZE=$BLOCK_SIZE
MODEL_CONFIG_PATH=$REMOTE_MODEL_CONFIG
EOF

echo "Shipping launch script + config to $RDU_HOST..."
scp -q -o BatchMode=yes "$(readlink -f "${BASH_SOURCE[0]}")" "$RDU_HOST:/tmp/vcc_rdu_decode.sh"
scp -q -o BatchMode=yes "$ENV_FILE" "$RDU_HOST:/tmp/vcc_rdu_decode_env.sh"
scp -q -o BatchMode=yes "$ADJUSTED_YAML" "$RDU_HOST:$REMOTE_MODEL_CONFIG"

echo "Launching RDU decode on $RDU_HOST (image: $RDU_IMAGE)..."
ssh -o BatchMode=yes "$RDU_HOST" "bash /tmp/vcc_rdu_decode.sh --inner" > "$RDU_LOG" 2>&1 &
SSH_PID=$!
echo "  log=$RDU_LOG  ssh_pid=$SSH_PID"

teardown() {
    echo "  Tearing down RDU decode..."
    ssh -o BatchMode=yes "$RDU_HOST" "podman stop -t 5 vcc-rdu-decode" 2>/dev/null || true
    kill "$SSH_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}

echo "  Waiting for RDU decode to register (up to ${RDU_DECODE_REGISTER_TIMEOUT}s, ~12-14 min typical: BAR2/PEF init)..."
for t in $(seq 1 "$RDU_DECODE_REGISTER_TIMEOUT"); do
    if grep -q "Registered base model '$SERVED_MODEL_NAME' MDC" "$RDU_LOG" 2>/dev/null; then
        echo "  RDU decode registered (${t}s)"
        echo "  $RDU_LOG"
        exit 0
    fi
    if ! kill -0 "$SSH_PID" 2>/dev/null; then
        echo "ERROR: RDU decode's ssh session exited early before registering. Log:"
        tail -30 "$RDU_LOG"
        teardown
        exit 1
    fi
    [[ $((t % 30)) -eq 0 ]] && echo "  ${t}s..." && tail -2 "$RDU_LOG" 2>/dev/null
    sleep 1
done
echo "ERROR: RDU decode did not register within ${RDU_DECODE_REGISTER_TIMEOUT}s"
tail -20 "$RDU_LOG"
teardown
exit 1
