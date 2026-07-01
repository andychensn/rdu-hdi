#!/bin/bash
# Build vllm-0.16.0+cpu wheel natively on s339 using VLLM_TARGET_DEVICE=cpu.
# Run via snrdu. Captures full cmake output to diagnose failures.
set -uo pipefail
export HOME=/tmp
export PYTHONNOUSERSITE=1
PY=/opt/sambanova/bin/python3.11
REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
VLLM_SRC="$REPO_ROOT/rdu-build-src/vllm-patched"
WHEEL_OUT="$REPO_ROOT/wheelhouse"

echo "=== Build environment on $(hostname) $(date) ==="
cmake --version
"$PY" --version
"$PY" -c "import torch; print('torch:', torch.__version__)"

echo ""
echo "=== VLLM_TARGET_DEVICE=cpu build (full output) ==="
cd "$VLLM_SRC"

# SambaNova ships protobuf in /opt/sambanova/lib/
export PKG_CONFIG_PATH="/opt/sambanova/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# ENABLE_NUMA=OFF: cmake controls NUMA; source-level define also added to utils.cpp
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/sambanova -DENABLE_NUMA=OFF"

VLLM_TARGET_DEVICE=cpu SETUPTOOLS_SCM_PRETEND_VERSION=0.16.0+cpu \
    "$PY" -m pip wheel . \
    --no-deps \
    --no-build-isolation \
    --no-cache-dir \
    --wheel-dir "$WHEEL_OUT" 2>&1

BUILD_EXIT=$?
echo ""
echo "=== pip exit code: $BUILD_EXIT ==="

echo ""
echo "=== CMakeError.log (actual cmake failures) ==="
find /tmp -name "CMakeError.log" 2>/dev/null | head -3 | while read f; do
    echo "--- $f ---"
    cat "$f" | tail -40
done

echo ""
echo "=== Result ==="
ls -lh "$WHEEL_OUT"/vllm-*+cpu*.whl 2>/dev/null || echo "no +cpu wheel produced"
