#!/usr/bin/env bash
# Build and push the GPU prefill Docker image.
# Run from login node (sc-vnc9) — no GPU required.
#
# Usage:
#   bash docker/gpu/build.sh             # builds + pushes with default tag
#   bash docker/gpu/build.sh --no-push   # build only, skip push
#   bash docker/gpu/build.sh --no-cache  # force a cache-free rebuild
#                                                 # (use for reproducibility testing —
#                                                 # this daemon is shared with other
#                                                 # users, so we can't just `docker
#                                                 # system prune`; --no-cache gets the
#                                                 # same guarantee for just this build)
#
# Image is tagged: $REGISTRY/$IMAGE_NAME:v${VLLM_VERSION}-rdu-hdi.${PATCH}
# where PATCH is an incrementing integer per rdu-hdi change above vllm.
# Update PATCH when UCX/NIXL commits or the vllm patch changes.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"

REGISTRY=sc-artifacts2.sambanovasystems.com/sw-docker-scratch
IMAGE_NAME=rdu-hdi-gpu-prefill
PUSH=true
NO_CACHE=false

# Parse args
for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=false ;;
        --no-cache) NO_CACHE=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$GPU_IMAGE_TAG"

# ── Broadcom OOT libbnxt_re ───────────────────────────────────────────────────
# The host GPU nodes run Broadcom's OOT bnxt_re kernel driver (v237.1.137.0),
# which requires matching OOT userspace (libbnxt_re). Ubuntu's inbox version
# sends incompatible UVERBS attributes causing ibv_open_device EINVAL.
#
# This tarball is IT-managed infrastructure software (like CUDA for NVIDIA),
# delivered to SambaNova by Broadcom as part of the NIC hardware support.
# IT maintains the canonical copy at the path below.
#
# If this path is missing, contact IT (ask Kurt McDougall or check
# /import/it-tools/idc/fw/brcm/ for the latest version).
BRCM_ROCELIB=/import/it-tools/idc/fw/brcm/237/bcm_237.1.148.0a/drivers_linux/bnxt_rocelib
BRCM_TARBALL="$BRCM_ROCELIB/libbnxt_re-237.1.137.0.tar.gz"
RDMA_LIBS_DIR="$REPO_ROOT/.rdma-libs"

mkdir -p "$RDMA_LIBS_DIR"
if [ ! -f "$RDMA_LIBS_DIR/libbnxt_re-237.1.137.0.tar.gz" ]; then
    echo "=== Copying Broadcom OOT libbnxt_re from IT-managed path ==="
    [ -f "$BRCM_TARBALL" ] || {
        echo "ERROR: Broadcom tarball not found at $BRCM_TARBALL"
        echo "  Contact IT to restore it, or check /import/it-tools/idc/fw/brcm/"
        exit 1
    }
    cp "$BRCM_TARBALL" "$RDMA_LIBS_DIR/"
    echo "  Copied: $(basename $BRCM_TARBALL)"
else
    echo "=== Broadcom OOT libbnxt_re already in .rdma-libs/ ==="
fi

# ── UCX + NIXL source (SambaNova org fork, internal GitHub Enterprise) ───────
# Cloned here, host-side (needs the invoking user's own SSH access to
# github.sambanovasystems.com), rather than inside the Dockerfile's own
# `docker build` — the isolated build container has no credentials for the
# internal GitHub Enterprise host. docker/gpu/Dockerfile just COPYs the
# already-pinned source staged here (same shape as build/bar2.sh's
# fetch_sources() for SambaNova/software).
GPU_BUILD_SRC_DIR="$REPO_ROOT/gpu-build-src"
mkdir -p "$GPU_BUILD_SRC_DIR"

fetch_pinned_source() {
    local name="$1" url="$2" branch="$3" commit="$4"
    local dest="$GPU_BUILD_SRC_DIR/$name"

    if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD)" = "$commit" ]; then
        echo "=== $name already at pinned commit in gpu-build-src/$name ==="
        return
    fi

    echo "=== Fetching $name @ $commit ==="
    rm -rf "$dest"
    git clone --branch "$branch" --depth 1 "$url" "$dest"
    if [ "$(git -C "$dest" rev-parse HEAD)" != "$commit" ]; then
        echo "  branch tip has moved past the pin — fetching the exact commit explicitly..."
        git -C "$dest" fetch --depth 1 origin "$commit"
        git -C "$dest" checkout "$commit"
    fi
    # .git is kept (small, shallow) so re-running this script can skip the
    # clone entirely once already at the pinned commit -- see the check above.
    echo "  staged $name @ $commit"
}

fetch_pinned_source ucx "$UCX_URL" "$UCX_BRANCH" "$UCX_COMMIT"
fetch_pinned_source nixl "$NIXL_URL" "$NIXL_BRANCH" "$NIXL_COMMIT"

echo "=== Building $FULL_IMAGE ==="
echo "    vllm:        $VLLM_VERSION"
echo "    UCX:         $UCX_BRANCH @ $UCX_COMMIT"
echo "    NIXL:        $NIXL_BRANCH @ $NIXL_COMMIT"
echo "    Dynamo:      $DYNAMO_VERSION"
echo "    LMCache:     $LMCACHE_VERSION"
echo ""

# docker/gpu/Dockerfile's ARGs have no defaults — every pin must come from here
# (config/versions.env), or the build fails loudly instead of silently
# using a stale value baked into the Dockerfile. UCX/NIXL are NOT passed as
# build-args (unlike VLLM/DYNAMO/LMCACHE below) -- their pin is already baked
# into the gpu-build-src/ checkout staged above; the Dockerfile just COPYs it.
BUILD_FLAGS=()
if [ "$NO_CACHE" = "true" ]; then
    echo "    (--no-cache: forcing a cold rebuild, ignoring any cached layers)"
    BUILD_FLAGS+=(--no-cache)
fi
sudo -g docker /usr/bin/docker-wrapper build \
    "${BUILD_FLAGS[@]}" \
    --build-arg VLLM_VERSION="v$VLLM_VERSION" \
    --build-arg DYNAMO_VERSION="$DYNAMO_VERSION" \
    --build-arg LMCACHE_VERSION="$LMCACHE_VERSION" \
    -t "$FULL_IMAGE" \
    -f "$REPO_ROOT/docker/gpu/Dockerfile" \
    "$REPO_ROOT"

echo ""
echo "=== Build complete: $FULL_IMAGE ==="

if [ "$PUSH" = "true" ]; then
    echo "=== Pushing to registry ==="
    sudo -g docker /usr/bin/docker-wrapper push "$FULL_IMAGE"
    echo "=== Done: $FULL_IMAGE ==="
else
    echo "(Skipped push — run 'sudo -g docker /usr/bin/docker-wrapper push $FULL_IMAGE' to push)"
fi
