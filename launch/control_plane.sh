#!/usr/bin/env bash
# Launch (or stop) the control-plane container (etcd + NATS + a bare vLLM
# frontend), matching the shape of launch/gpu_prefill.sh and launch/rdu_decode.sh.
#
# Usage:
#   bash launch/control_plane.sh          # launch (foreground; backgrounds itself: `&`)
#   bash launch/control_plane.sh --stop   # graceful SIGTERM to the running container
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

if [[ "${1:-}" == "--stop" ]]; then
    sudo -g docker /usr/bin/docker-wrapper exec rdu-hdi-control-plane sh -c 'kill -TERM 1'
    exit 0
fi

# NOTE: use -e VAR="$VAR" (explicit value), not bare -e VAR — sudo strips the
# calling shell's environment by default, so bare -e VAR forwards an EMPTY
# value and the entrypoint fails with "CONTROL_PLANE_IP must be set".
exec sudo -g docker /usr/bin/docker-run-wrapper --pull=always --net=host --rm \
    --name rdu-hdi-control-plane \
    -e CONTROL_PLANE_IP="$CONTROL_PLANE_IP" -e ETCD_PORT="$ETCD_PORT" \
    -e NATS_PORT="$NATS_PORT" -e VLLM_PORT="$VLLM_PORT" \
    -e BLOCK_SIZE="$BLOCK_SIZE" \
    "$CONTROL_PLANE_IMAGE"
