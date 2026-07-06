#!/usr/bin/env bash
# Build and push the control-plane Docker image (etcd + NATS + Dynamo frontend).
# Run from login node — no GPU/RDU required, zero NFS dependency.
#
# Usage:
#   bash docker/control-plane/build.sh             # builds + pushes with default tag
#   bash docker/control-plane/build.sh --no-push   # build only, skip push
#   bash docker/control-plane/build.sh --no-cache  # force a cache-free rebuild
#
# Image is tagged: $REGISTRY/$IMAGE_NAME:v${DYNAMO_VERSION}-rdu-hdi.${PATCH}
# where PATCH is an incrementing integer per rdu-hdi change above dynamo.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"

REGISTRY=sc-artifacts2.sambanovasystems.com/sw-docker-scratch
IMAGE_NAME=rdu-hdi-control-plane
PUSH=true
NO_CACHE=false

for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=false ;;
        --no-cache) NO_CACHE=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:${CONTROL_PLANE_IMAGE_TAG:-v${DYNAMO_VERSION}-rdu-hdi.1}"

echo "=== Building $FULL_IMAGE ==="
echo "    etcd:   $ETCD_VERSION"
echo "    NATS:   $NATS_VERSION"
echo "    Dynamo: $DYNAMO_VERSION"
echo ""

BUILD_FLAGS=()
if [ "$NO_CACHE" = "true" ]; then
    echo "    (--no-cache: forcing a cold rebuild, ignoring any cached layers)"
    BUILD_FLAGS+=(--no-cache)
fi
sudo -g docker /usr/bin/docker-wrapper build \
    "${BUILD_FLAGS[@]}" \
    --build-arg ETCD_VERSION="$ETCD_VERSION" \
    --build-arg ETCD_SHA256="$ETCD_SHA256" \
    --build-arg NATS_VERSION="$NATS_VERSION" \
    --build-arg NATS_SHA256="$NATS_SHA256" \
    --build-arg DYNAMO_VERSION="$DYNAMO_VERSION" \
    -t "$FULL_IMAGE" \
    -f "$REPO_ROOT/docker/control-plane/Dockerfile" \
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
