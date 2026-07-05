#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/import/snvm-sc-scratch1/andyc/rdu-hdi
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

RDMA_DEVICES=""
for dev in /dev/infiniband /dev/uverbs*; do
    [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
done

echo "=== starting persistent RDU decode container (self-built, fully baked-in) on $(hostname) $(date) ==="
exec sudo -g docker /usr/bin/docker-run-wrapper --pull=always --net=host --rm \
    --name rdu-decode \
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
    "$RDU_IMAGE"
