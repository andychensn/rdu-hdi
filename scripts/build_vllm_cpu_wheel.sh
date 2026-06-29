#!/usr/bin/env bash
# Build a vllm CPU-only wheel compatible with torch 2.2.x (on s339 RDU nodes).
#
# The official PyPI vllm 0.16.0 wheel uses torch 2.4+ APIs incompatible with
# torch 2.2.0+sn on s339. This builds a stripped CPU wheel from the official
# vllm source that avoids those APIs.
#
# Must run ON sc3-s339 (has python3.11 + access to system torch 2.2.0+sn).
# Output: $REPO_ROOT/wheelhouse/vllm-0.16.0+cpu-cp311-*.whl
#
# Usage (from login node):
#   source config/cluster.env
#   snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
#       --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:20:00 \
#       -o logs/build_vllm_cpu_wheel.log \
#       -- bash scripts/build_vllm_cpu_wheel.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

WHEEL_OUT="$REPO_ROOT/wheelhouse"
BUILD_TMP=$(mktemp -d /tmp/vllm-cpu-build-XXXX)
trap "rm -rf $BUILD_TMP" EXIT
PY=/opt/sambanova/bin/python3.11

echo "=== Building vllm $VLLM_VERSION CPU wheel on $(hostname) $(date) ==="
[ -x "$PY" ] || { echo "ERROR: $PY not found — must run on s339"; exit 1; }

mkdir -p "$WHEEL_OUT"

# setuptools<77 required: newer versions enforce strict SPDX license format
# that vllm's pyproject.toml doesn't comply with, causing metadata-generation-failed
"$PY" -m pip install --user --break-system-packages "setuptools<77" wheel 2>&1 | tail -3

echo "=== Cloning vllm v$VLLM_VERSION ==="
git clone --depth=1 --branch "v$VLLM_VERSION" \
    https://github.com/vllm-project/vllm.git "$BUILD_TMP/vllm"

echo "=== Building CPU-only wheel ==="
cd "$BUILD_TMP/vllm"
VLLM_TARGET_DEVICE=cpu \
SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}+cpu" \
    "$PY" -m pip wheel . \
    --no-deps \
    --no-build-isolation \
    --wheel-dir "$WHEEL_OUT" 2>&1 | tail -10

WHL=$(find "$WHEEL_OUT" -name "vllm-*cp311*.whl" -newer "$BUILD_TMP" | head -1 || true)
echo "=== Done: ${WHL:-no wheel found — check build output} ==="
echo "Update wheelhouse/ and rebuild RDU venv with: bash scripts/build_rdu_venv.sh"
