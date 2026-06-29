#!/usr/bin/env bash
# Fetch Python wheels needed for the RDU venv into wheelhouse/.
# Run from login node (has internet access).
#
# Note: nixl-pathb wheel is no longer fetched here — build it from source instead:
#   bash scripts/build_rdu_ucx_nixl.sh  (run on s339 via snrdu)
# This is preferred over the NFS copy because it's reproducible and pinned to
# andychensn/nixl@$NIXL_COMMIT in versions.env.
#
# Usage: bash scripts/fetch_rdu_wheels.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
WHEELHOUSE="$REPO_ROOT/wheelhouse"
mkdir -p "$WHEELHOUSE"

echo "=== Fetching RDU wheels into $WHEELHOUSE ==="

# 1. vllm 0.16.0 ABI3 wheel (generic linux — bypasses glibc 2.31 check on s339/RHEL8)
echo "--- vllm $VLLM_VERSION (ABI3 wheel, renamed to linux_x86_64 for RHEL8 compat) ---"
VLLM_MANYLINUX="$WHEELHOUSE/vllm-${VLLM_VERSION}-cp38-abi3-manylinux_2_31_x86_64.whl"
VLLM_GENERIC="$WHEELHOUSE/vllm-${VLLM_VERSION}-cp38-abi3-linux_x86_64.whl"
if [ ! -f "$VLLM_GENERIC" ]; then
    pip download "vllm==$VLLM_VERSION" --no-deps --dest "$WHEELHOUSE" \
        --python-version 311 --platform manylinux_2_31_x86_64 2>&1 | tail -3
    [ -f "$VLLM_MANYLINUX" ] && cp "$VLLM_MANYLINUX" "$VLLM_GENERIC"
fi
echo "  vllm wheel: $VLLM_GENERIC"

# 2. ai-dynamo-runtime wheel (from wheelhouse in original dynamo_hdi repo)
echo "--- ai-dynamo-runtime $DYNAMO_VERSION ---"
if ! find "$WHEELHOUSE" -name "ai_dynamo_runtime-*.whl" | grep -q .; then
    # Try PyPI first (may not be available)
    pip download "ai-dynamo-runtime==$DYNAMO_VERSION" --only-binary=:all: \
        --dest "$WHEELHOUSE" 2>/dev/null || \
    # Fallback: copy from existing installation on this cluster
    find /import/snvm-sc-scratch1/andyc -name "ai_dynamo_runtime-${DYNAMO_VERSION}-*.whl" \
        2>/dev/null | head -1 | xargs -I{} cp {} "$WHEELHOUSE/" 2>/dev/null || \
    echo "  WARNING: ai-dynamo-runtime wheel not found — install from PyPI when available"
fi
find "$WHEELHOUSE" -name "ai_dynamo_runtime-*.whl" -exec echo "  dynamo wheel: {}" \;

# 3. nixl-pathb — now built from source (see scripts/build_rdu_ucx_nixl.sh)
echo "--- nixl-pathb (built from source by build_rdu_ucx_nixl.sh) ---"
NIXL_WHL=$(find "$WHEELHOUSE" -name "nixl*pathb*.whl" -o -name "nixl_cu12*cp311*.whl" 2>/dev/null | head -1 || true)
if [ -n "$NIXL_WHL" ]; then
    echo "  Already present: $NIXL_WHL"
else
    echo "  Not yet built — run: bash scripts/build_rdu_ucx_nixl.sh (on s339 via snrdu)"
    # Fallback to NFS copy while build_rdu_ucx_nixl.sh hasn't been run yet
    GUOYAOF_NIXL=/import/snvm-sc-scratch1/guoyaof/wheels/nixl-1.0.0+pathb-cp311-cp311-linux_x86_64.whl
    if [ -f "$GUOYAOF_NIXL" ]; then
        cp "$GUOYAOF_NIXL" "$WHEELHOUSE/"
        echo "  Fallback: copied from guoyaof NFS (replace with source build when ready)"
    fi
fi

echo ""
echo "=== Wheelhouse contents ==="
ls -lh "$WHEELHOUSE"/*.whl 2>/dev/null || echo "No wheels found"
