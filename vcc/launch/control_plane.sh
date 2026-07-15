#!/usr/bin/env bash
# Launch (or stop) the control-plane container on VCC's CONTROL_PLANE_HOST.
# Same image/entrypoint as launch/control_plane.sh (vnc+idc) -- only the
# orchestration differs: direct SSH + rootless Podman instead of
# sudo -g docker /usr/bin/docker-run-wrapper (that wrapper doesn't exist on
# VCC, and rootless Podman needs no sudo at all for a container this simple:
# no GPU/RDU device passthrough, no SELinux-label concern).
#
# Usage:
#   bash vcc/launch/control_plane.sh          # launch (foreground; backgrounds itself: `&`)
#   bash vcc/launch/control_plane.sh --stop   # graceful SIGTERM to the running container
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/vcc/cluster.env"
source "$REPO_ROOT/vcc/model.env"

if [[ "${1:-}" == "--stop" ]]; then
    ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" "podman exec rdu-hdi-control-plane sh -c 'kill -TERM 1'"
    exit 0
fi

echo "Starting control plane on $CONTROL_PLANE_HOST (image: $CONTROL_PLANE_IMAGE)..."
exec ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" \
    "podman run --net=host --rm \
        --name rdu-hdi-control-plane \
        -e CONTROL_PLANE_IP=$CONTROL_PLANE_IP -e ETCD_PORT=$ETCD_PORT \
        -e NATS_PORT=$NATS_PORT -e VLLM_PORT=$VLLM_PORT \
        -e BLOCK_SIZE=$BLOCK_SIZE \
        $CONTROL_PLANE_IMAGE"
