#!/usr/bin/env bash
# Transfer already-built rdu-hdi images from this login node (vnc+idc) onto
# the VCC nodes, without ever touching a registry.
#
# Why not just push/pull through sc-artifacts2.sambanovasystems.com like the
# vnc+idc launch scripts do: confirmed live (2026-07-15) that VCC nodes can
# never resolve sc-artifacts2.sambanovasystems.com or
# github.sambanovasystems.com by DNS -- both require being on the corporate
# network, which these standalone demo boxes structurally aren't and won't
# be (per infra team). This isn't a temporary gap; it's the permanent shape
# of this environment. It doesn't block anything here though: the registry
# is only ever used for push/pull, and build/rdu_env.sh + build/bar2.sh's
# clones of github.sambanovasystems.com's private repos are a BUILD-time
# step only, already baked into the image layers by the time an image is
# built here -- VCC only ever needs to RUN a finished image, never fetch
# anything from either host at runtime (confirmed by grepping every
# runtime-reached script -- docker/rdu/entrypoint.sh,
# docker/control-plane/entrypoint.sh -- for pull/clone/curl/wget references:
# none found).
#
# Prerequisites: images already built + resident in this login node's local
# image store (bash docker/gpu/build.sh, docker/control-plane/build.sh,
# docker/rdu/build.sh -- these produce a local image immediately, no
# separate push needed for this script to work).
#
# Usage:
#   bash vcc/deploy_images.sh              # deploy all 3 images
#   bash vcc/deploy_images.sh gpu           # just the GPU prefill image (to GPU_HOST)
#   bash vcc/deploy_images.sh control-plane # just the control-plane image (to CONTROL_PLANE_HOST)
#   bash vcc/deploy_images.sh rdu           # just the RDU decode image (to RDU_HOST)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/vcc/cluster.env"

TMP_DIR="${TMPDIR:-/tmp}/vcc-image-transfer"
mkdir -p "$TMP_DIR"

# save (login node, real dockerd via docker-wrapper) -> rsync (resumable,
# unlike scp, which matters for multi-GB single-file transfers over a link
# whose sustained throughput we've only spot-checked at ~30MB/s) -> load
# (target node, rootless Podman). podman load preserves the full image
# reference baked into the tarball's manifest (e.g.
# sc-artifacts2.sambanovasystems.com/sw-docker-scratch/rdu-hdi-gpu-prefill:v0.16.0-rdu-hdi.5)
# -- no retagging needed, vcc/launch/*.sh reference the SAME $GPU_IMAGE/
# $RDU_IMAGE/$CONTROL_PLANE_IMAGE variables config/cluster.env already
# defines. The registry-qualified name never needs to actually resolve on
# the target node; podman only treats it as a local storage key once loaded.
deploy_one() {
    local image="$1" host="$2" label="$3"
    local tar_name
    tar_name="$(echo "$image" | tr '/:' '__').tar"
    local tar_path="$TMP_DIR/$tar_name"

    echo "=== [$label] Saving $image (login node) ==="
    sudo -g docker /usr/bin/docker-wrapper save "$image" -o "$tar_path"
    ls -lh "$tar_path"

    echo "=== [$label] Transferring to $host:~/vcc-images/ ==="
    ssh -o BatchMode=yes "$host" "mkdir -p ~/vcc-images"
    rsync -avzP -e "ssh -o BatchMode=yes" "$tar_path" "$host:~/vcc-images/$tar_name"

    echo "=== [$label] Loading on $host (rootless Podman) ==="
    ssh -o BatchMode=yes "$host" "podman load -i ~/vcc-images/$tar_name"

    rm -f "$tar_path"
    echo "=== [$label] Done: $image is now loaded locally on $host ==="
}

WHAT="${1:-all}"
case "$WHAT" in
    all|gpu|control-plane|rdu) ;;
    *) echo "Usage: $0 [all|gpu|control-plane|rdu]" >&2; exit 1 ;;
esac

[[ "$WHAT" == "all" || "$WHAT" == "gpu" ]] && deploy_one "$GPU_IMAGE" "$GPU_HOST" "gpu-prefill"
[[ "$WHAT" == "all" || "$WHAT" == "control-plane" ]] && deploy_one "$CONTROL_PLANE_IMAGE" "$CONTROL_PLANE_HOST" "control-plane"
[[ "$WHAT" == "all" || "$WHAT" == "rdu" ]] && deploy_one "$RDU_IMAGE" "$RDU_HOST" "rdu-decode"

rmdir "$TMP_DIR" 2>/dev/null || true
echo "=== All requested images deployed ==="
