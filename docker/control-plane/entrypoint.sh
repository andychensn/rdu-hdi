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
: "${BLOCK_SIZE:=64}"

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

echo "Starting Dynamo frontend on 0.0.0.0:$VLLM_PORT (router-mode kv, block-size $BLOCK_SIZE)..."
# --router-mode kv: prefill-side KV-cache-aware + load-aware routing.
# Confirmed via live test that the default round-robin mode ignores cache
# state entirely -- kv mode's cost function blends cache-overlap credit with
# each worker's in-flight prefill-token load (verified against the actual
# ai-dynamo/dynamo v1.2.1 source, not just docs).
# --router-kv-overlap-score-credit 1.0 (default): favor cache/TTFT, matches
# our long-shared-prefix (system-prompt/repo-context) workload.
# --router-temperature 0.4 (NOT the 0.0 default): softmax-samples over cost
# logits instead of deterministic argmin -- with only 2 prefill workers,
# deterministic selection risks pinning all cache-hit traffic on one worker;
# revisit as worker count grows and full determinism becomes safer.
# --active-decode-blocks-threshold None: disables the router's binary
# busy/free gate for decode workers (default 1.0 = reject outright once a
# worker's KV-block utilization hits 100%, which a --max-num-seqs=2 decode
# instance reaches at just 2 concurrent requests). Confirmed live that this
# gate was rejecting prefill-already-completed requests with "Service
# Unavailable" instead of letting them queue -- vLLM's own scheduler already
# queues correctly once a request reaches it, so with a single decode
# worker (nothing to route around it toward) this check can only reject
# incorrectly, never help. Revisit once a second decode worker exists.
exec env \
    ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
    python3 -m dynamo.frontend --http-port "$VLLM_PORT" \
        --router-mode kv \
        --kv-cache-block-size "$BLOCK_SIZE" \
        --router-kv-overlap-score-credit 1.0 \
        --router-temperature 0.4 \
        --active-decode-blocks-threshold None
