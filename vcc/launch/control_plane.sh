#!/usr/bin/env bash
# Launch (or stop) the control-plane container on VCC's CONTROL_PLANE_HOST.
# Same image/entrypoint as launch/control_plane.sh (vnc+idc) -- only the
# orchestration differs: direct SSH + rootless Podman instead of
# sudo -g docker /usr/bin/docker-run-wrapper (that wrapper doesn't exist on
# VCC, and rootless Podman needs no sudo at all for a container this simple:
# no GPU/RDU device passthrough, no SELinux-label concern).
#
# Usage:
#   bash vcc/launch/control_plane.sh          # launch detached, returns once started
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
# -d (detached): a plain foreground `podman run` over a non-pty ssh session
# dies the moment that ssh connection drops (idle timeout, network blip --
# the same class of drop hit during VCC image transfers) -- confirmed live:
# this exact thing silently killed a control-plane container hours into an
# otherwise-idle run, with --rm erasing all trace of it. Detached, the
# container is a podman-managed background process independent of the ssh
# session that started it, matching "leave the endpoint live" intent.
exec ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" \
    "podman run -d --replace --net=host --rm \
        --name rdu-hdi-control-plane \
        -e CONTROL_PLANE_IP=$CONTROL_PLANE_IP -e ETCD_PORT=$ETCD_PORT \
        -e NATS_PORT=$NATS_PORT -e VLLM_PORT=$VLLM_PORT \
        -e BLOCK_SIZE=$BLOCK_SIZE \
        $CONTROL_PLANE_IMAGE"
