#!/usr/bin/env bash
# Launch etcd + NATS + Dynamo HTTP frontend on the login node.
# Usage:
#   bash launch/control_plane.sh          # start
#   bash launch/control_plane.sh --stop   # stop all three
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

ETCD="$REPO_ROOT/vendor/bin/etcd"
NATS="$REPO_ROOT/vendor/bin/nats-server"
# Lean control plane venv — python3.12 + ai-dynamo only, no GPU/CUDA needed.
# Auto-created on first run. Lives on NFS, accessible from login node.
CP_VENV="$REPO_ROOT/.venv_cp"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# Bootstrap control plane venv if missing
if [ ! -x "$CP_VENV/bin/python" ]; then
    echo "Creating control plane venv ($CP_VENV)..."
    python3.12 -m venv "$CP_VENV"
    "$CP_VENV/bin/pip" install -q "ai-dynamo[vllm]==$DYNAMO_VERSION"
    echo "  Control plane venv ready"
fi

if [[ "${1:-}" == "--stop" ]]; then
    echo "Stopping control plane..."
    pkill -f "$ETCD.*$ETCD_PORT" 2>/dev/null && echo "  etcd stopped" || true
    pkill -f "$NATS.*$NATS_PORT" 2>/dev/null && echo "  nats stopped" || true
    pkill -f "dynamo.frontend.*$VLLM_PORT" 2>/dev/null && echo "  frontend stopped" || true
    rm -rf "$REPO_ROOT/.etcd_data"
    exit 0
fi

rm -rf "$REPO_ROOT/.etcd_data"

echo "Starting etcd on $CONTROL_PLANE_IP:$ETCD_PORT..."
"$ETCD" \
    --listen-client-urls "http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    --advertise-client-urls "http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    --data-dir "$REPO_ROOT/.etcd_data" \
    --log-level warn \
    >> "$LOG_DIR/${TS}_etcd.log" 2>&1 &
ETCD_PID=$!
sleep 2

echo "Starting NATS on $CONTROL_PLANE_IP:$NATS_PORT..."
"$NATS" --port "$NATS_PORT" >> "$LOG_DIR/${TS}_nats.log" 2>&1 &
NATS_PID=$!
sleep 1

echo "Starting Dynamo frontend on $CONTROL_PLANE_IP:$VLLM_PORT..."
ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
    "$CP_VENV/bin/python" -m dynamo.frontend \
    --http-port "$VLLM_PORT" \
    >> "$LOG_DIR/${TS}_frontend.log" 2>&1 &
FRONTEND_PID=$!
sleep 3

# Verify
if curl -sf --max-time 5 "http://$CONTROL_PLANE_IP:$VLLM_PORT/health" &>/dev/null || \
   curl -sf --max-time 5 "http://$CONTROL_PLANE_IP:$VLLM_PORT/v1/models" &>/dev/null; then
    echo "Control plane ready:"
else
    echo "Control plane started (health endpoint may not be ready yet):"
fi

echo "  etcd    PID=$ETCD_PID  port=$ETCD_PORT"
echo "  nats    PID=$NATS_PID  port=$NATS_PORT"
echo "  frontend PID=$FRONTEND_PID  http://$CONTROL_PLANE_IP:$VLLM_PORT"
echo "  logs    $LOG_DIR/${TS}_*.log"
