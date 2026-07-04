#!/usr/bin/env bash
# Test the rdu-hdi-rdu-decode Docker image end-to-end. Run via snrdu on the
# RDU node. Assumes control plane + GPU prefill are already running.
REPO_ROOT=/import/snvm-sc-scratch1/andyc/rdu-hdi
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

# --net=host alone gives network-NAMESPACE visibility (needed for
# hostname -I to see the real RoCE IP) but NOT RDMA hardware access --
# that's a separate device-node passthrough requirement, same as
# launch/gpu_prefill.sh's RDMA_DEVICES construction on the GPU side.
RDMA_DEVICES=""
for dev in /dev/infiniband /dev/uverbs*; do
    [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
done

echo "=== starting container (foreground, backgrounded via shell) ==="
sudo -g docker /usr/bin/docker-run-wrapper --pull=always --net=host --rm \
    --name test-rdu-decode \
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
    -e BAR2_INSTALL="$BAR2_INSTALL" \
    -e BAR2_RUNTIME_LIBS="$BAR2_RUNTIME_LIBS" \
    -e BAR2_PRELOAD="$BAR2_PRELOAD" \
    "$RDU_IMAGE" &
CONTAINER_PID=$!
echo "  shell PID: $CONTAINER_PID"

echo "=== waiting ~14 min for BAR2 init ==="
for i in $(seq 1 84); do
    sleep 10
    printf "."
done
echo ""

echo "=== curl test ==="
curl -s -m 90 http://"$CONTROL_PLANE_IP":"$VLLM_PORT"/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiniMax-M2.7","prompt":"Hello, how are you?","max_tokens":16,"temperature":0}'
echo ""

echo "=== killing container process ==="
kill "$CONTAINER_PID" 2>&1 || true
sleep 3
echo "=== done ==="
