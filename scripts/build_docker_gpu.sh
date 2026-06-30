#!/usr/bin/env bash
# Build and push the GPU prefill Docker image.
# Run from login node (sc-vnc9) — no GPU required.
#
# Usage:
#   bash scripts/build_docker_gpu.sh           # builds + pushes with default tag
#   bash scripts/build_docker_gpu.sh --no-push # build only, skip push
#
# Image is tagged: $REGISTRY/$IMAGE_NAME:v${VLLM_VERSION}-rdu-hdi.${PATCH}
# where PATCH is an incrementing integer per rdu-hdi change above vllm.
# Update PATCH when UCX/NIXL commits or the vllm patch changes.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"

REGISTRY=sc-artifacts2.sambanovasystems.com/sw-docker-scratch
IMAGE_NAME=rdu-hdi-gpu-prefill
PUSH=true

# Parse args
for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=false ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:$GPU_IMAGE_TAG"

echo "=== Building $FULL_IMAGE ==="
echo "    vllm:    $VLLM_VERSION"
echo "    UCX:     $UCX_COMMIT"
echo "    NIXL:    $NIXL_COMMIT"
echo "    Dynamo:  $DYNAMO_VERSION"
echo ""

sudo -g docker /usr/bin/docker-wrapper build \
    --build-arg VLLM_VERSION="$VLLM_VERSION" \
    --build-arg UCX_COMMIT="$UCX_COMMIT" \
    --build-arg NIXL_COMMIT="$NIXL_COMMIT" \
    --build-arg DYNAMO_VERSION="$DYNAMO_VERSION" \
    -t "$FULL_IMAGE" \
    -f "$REPO_ROOT/Dockerfile.gpu" \
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
