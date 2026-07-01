#!/bin/bash
# Try building vllm-0.16.0+cpu with protobuf faked out via CMAKE_ARGS.
# Run via snrdu on s339.
set -uo pipefail
export HOME=/tmp
export PYTHONNOUSERSITE=1

PY=/opt/sambanova/bin/python3.11
REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
VLLM_SRC="$REPO_ROOT/rdu-build-src/vllm-patched"
WHEEL_OUT="$REPO_ROOT/wheelhouse"

echo "=== $(hostname) $(date) ==="
"$PY" --version
cmake --version | head -1

# Remove stale cmake cache from previous attempts
rm -rf "$VLLM_SRC/build/"
echo "Cleared stale cmake cache"

echo ""
echo "=== Building with protobuf bypassed ==="
cd "$VLLM_SRC"

# SambaNova ships its own protobuf in /opt/sambanova/lib/.
# Point cmake to it via PKG_CONFIG_PATH + CMAKE_PREFIX_PATH.
export PKG_CONFIG_PATH="/opt/sambanova/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# /opt/sambanova has protobuf + other SambaNova libraries
# VLLM_NUMA_DISABLED: numa.h (numactl-devel) not installed; skip numa usage
# CMAKE_PREFIX_PATH: find SambaNova's bundled protobuf (required by Caffe2Config)
# VLLM_NUMA_DISABLED passed as CXX define (not cmake var) so #ifndef check works
# ENABLE_NUMA=OFF: skips -lnuma link + adds -DVLLM_NUMA_DISABLED compile flag automatically
# numactl-devel not installed on s339 (libnuma.so.1 exists but no unversioned .so)
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/sambanova -DENABLE_NUMA=OFF"

VLLM_TARGET_DEVICE=cpu \
SETUPTOOLS_SCM_PRETEND_VERSION=0.16.0+cpu \
    "$PY" -m pip wheel . \
    --no-deps \
    --no-build-isolation \
    --no-cache-dir \
    --wheel-dir "$WHEEL_OUT" 2>&1

BUILD_EXIT=$?
echo "pip exit code: $BUILD_EXIT"

echo ""
echo "=== cmake errors (if any) ==="
find "$VLLM_SRC/build" -name "CMakeError.log" 2>/dev/null | xargs cat 2>/dev/null | tail -20 || true

echo ""
echo "=== Result ==="
ls -lh "$WHEEL_OUT"/vllm-*+cpu*.whl 2>/dev/null || echo "no +cpu wheel produced"
