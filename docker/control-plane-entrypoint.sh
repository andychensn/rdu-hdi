#!/usr/bin/env bash
# Control-plane container entrypoint: etcd + NATS + Dynamo HTTP frontend, one
# process set per container: etcd/NATS run in the background, the frontend
# runs in the foreground as the container's main process (so `docker stop`/
# k8s termination signals reach it directly), and a trap tears down the
# background processes on exit so the container doesn't leave orphans
# behind on a shared node.
set -euo pipefail

: "${CONTROL_PLANE_IP:?CONTROL_PLANE_IP must be set}"
: "${ETCD_PORT:=12379}"
: "${NATS_PORT:=14222}"
: "${VLLM_PORT:=18000}"

ETCD_DATA_DIR="${ETCD_DATA_DIR:-/tmp/etcd-data}"
mkdir -p "$ETCD_DATA_DIR"

cleanup() {
    echo "Shutting down control plane..."
    [ -n "${ETCD_PID:-}" ] && kill "$ETCD_PID" 2>/dev/null || true
    [ -n "${NATS_PID:-}" ] && kill "$NATS_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

echo "Starting etcd on 0.0.0.0:$ETCD_PORT (advertised as $CONTROL_PLANE_IP:$ETCD_PORT)..."
/opt/vendor/bin/etcd \
    --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --advertise-client-urls "http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    --data-dir "$ETCD_DATA_DIR" \
    --log-level warn &
ETCD_PID=$!
sleep 2

echo "Starting NATS on 0.0.0.0:$NATS_PORT..."
/opt/vendor/bin/nats-server --port "$NATS_PORT" &
NATS_PID=$!
sleep 1

echo "Starting Dynamo frontend on 0.0.0.0:$VLLM_PORT..."
exec env \
    ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
    DYN_REQUEST_PLANE=tcp \
    python3 -m dynamo.frontend --http-port "$VLLM_PORT"
