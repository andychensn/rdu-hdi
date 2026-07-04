#!/usr/bin/env bash
# Build and push the RDU decode Docker image.
# Run from login node — the FROM base image (rhel810-dev) brings
# /opt/sambanova along, same principle as Dockerfile.gpu bringing CUDA
# along via FROM vllm/vllm-openai, so no snrdu/RDU node is needed to BUILD
# this image (only to RUN it — see docker-run-wrapper's auto-mount of
# /import/scratch for the BAR2_* NFS paths at container runtime).
#
# Prerequisite: rdu-ucx-install/ and wheelhouse/{vllm+cpu,ai_dynamo*,
# nixl_cu12,...}.whl must already exist (built via
# scripts/build_rdu_env.sh --fetch-only / --build-only) — this script does
# NOT build those from source, it only bakes the already-built artifacts
# into the image.
#
# Usage:
#   bash scripts/build_docker_rdu.sh             # builds + pushes with default tag
#   bash scripts/build_docker_rdu.sh --no-push   # build only, skip push
#   bash scripts/build_docker_rdu.sh --no-cache  # force a cache-free rebuild
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"

REGISTRY=sc-artifacts2.sambanovasystems.com/sw-docker-scratch
IMAGE_NAME=rdu-hdi-rdu-decode
PUSH=true
NO_CACHE=false

for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=false ;;
        --no-cache) NO_CACHE=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

[ -d "$REPO_ROOT/rdu-ucx-install/lib" ] || { echo "ERROR: rdu-ucx-install/ not found — run scripts/build_rdu_env.sh first"; exit 1; }
[ -n "$(find "$REPO_ROOT/wheelhouse" -name 'vllm-*+cpu-cp311*.whl' 2>/dev/null)" ] || { echo "ERROR: no vllm +cpu wheel in wheelhouse/ — run scripts/build_rdu_env.sh first"; exit 1; }
[ -d "$REPO_ROOT/fast-coe/server/vllm-rdu" ] || { echo "ERROR: fast-coe/ not found — run scripts/build_rdu_env.sh --fetch-only first"; exit 1; }
# .dockerignore excludes fast-coe/.git/ from the Docker build context (keeps
# it small), so the pinned-commit check has to happen here instead of inside
# the Dockerfile.
FAST_COE_ACTUAL=$(git -C "$REPO_ROOT/fast-coe" rev-parse HEAD)
[ "$FAST_COE_ACTUAL" = "$FAST_COE_COMMIT" ] || { echo "ERROR: fast-coe/ is at $FAST_COE_ACTUAL, expected $FAST_COE_COMMIT (config/versions.env)"; exit 1; }

FULL_IMAGE="$REGISTRY/$IMAGE_NAME:${RDU_IMAGE_TAG:-v${DYNAMO_VERSION}-rdu-hdi.1}"

echo "=== Building $FULL_IMAGE ==="
echo "    base:          $RHEL810_DEV_IMAGE"
echo "    fast-coe:      $FAST_COE_COMMIT"
echo "    transformers:  $RDU_TRANSFORMERS_VERSION"
echo ""

BUILD_FLAGS=()
if [ "$NO_CACHE" = "true" ]; then
    echo "    (--no-cache: forcing a cold rebuild, ignoring any cached layers)"
    BUILD_FLAGS+=(--no-cache)
fi
sudo -g docker /usr/bin/docker-wrapper build \
    "${BUILD_FLAGS[@]}" \
    --build-arg RHEL810_DEV_IMAGE="$RHEL810_DEV_IMAGE" \
    --build-arg FAST_COE_COMMIT="$FAST_COE_COMMIT" \
    --build-arg RDU_TRANSFORMERS_VERSION="$RDU_TRANSFORMERS_VERSION" \
    -t "$FULL_IMAGE" \
    -f "$REPO_ROOT/Dockerfile.rdu" \
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
