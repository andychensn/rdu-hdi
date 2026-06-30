#!/usr/bin/env bash
# Build the GPU venv for rdu-hdi from scratch.
# Must run on an H200 GPU node (sc3-c127 or sc3-c129) with CUDA 13.x.
# Usage: srun -p gpuonly -w sc3-c127 --gres=gpu:4 -c 16 --mem=65536 -t 01:30:00 \
#            bash scripts/build_gpu_venv.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

UCX_INSTALL=$REPO_ROOT/ucx-install
NIXL_WHEEL_DIR=$REPO_ROOT/wheelhouse
VENV=$REPO_ROOT/.venv_gpu
BUILD_TMP=${BUILD_TMP:-$(mktemp -d /tmp/rdu-hdi-build-XXXX)}
NPROC=$(nproc 2>/dev/null || echo 8)

# CUDA detection
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
if command -v nvcc >/dev/null; then
    CUDA_BIN=$(dirname "$(command -v nvcc)")
elif [ -x "$CUDA_HOME/bin/nvcc" ]; then
    CUDA_BIN="$CUDA_HOME/bin"
else
    echo "ERROR: nvcc not found. Set CUDA_HOME."; exit 1
fi
export PATH="$CUDA_BIN:$PATH"
export CUDACXX="$CUDA_BIN/nvcc"
CUDA_ARG="--with-cuda=$CUDA_HOME"

echo "=== rdu-hdi GPU venv build on $(hostname) $(date) ==="
echo "    REPO=$REPO_ROOT  BUILD=$BUILD_TMP"
echo "    CUDA: $(nvcc --version | grep release | head -1)"
echo "    Python: $(python3.12 --version)"

# Prerequisites
for cmd in gcc make autoreconf libtoolize nvcc python3.12; do
    command -v "$cmd" >/dev/null || { echo "ERROR: $cmd not found"; exit 1; }
done
echo "Prerequisites: OK"

# ── Step 1: Build UCX 1.22 ──────────────────────────────────────────────────
if [ -f "$UCX_INSTALL/include/ucp/api/ucp.h" ] && [ "${SKIP_UCX:-0}" = "1" ]; then
    echo "=== Step 1: UCX already built (SKIP_UCX=1) ==="
else
    echo "=== Step 1: Building UCX from andychensn/ucx@$UCX_BRANCH ==="
    UCX_SRC="$BUILD_TMP/ucx"
    rm -rf "$UCX_SRC"
    git clone --depth=200 --branch "$UCX_BRANCH" \
        https://github.com/andychensn/ucx.git "$UCX_SRC"
    git -C "$UCX_SRC" checkout "$UCX_COMMIT"

    # Strip GDAKI (DOCA/mlx5 GPU-Direct-Async) and RDU modules — not needed on GPU host
    sed -i 's/^SUBDIRS = \. gdaki/SUBDIRS = ./' "$UCX_SRC/src/uct/ib/mlx5/Makefile.am" 2>/dev/null || true
    sed -i '\#m4_include(\[src/uct/ib/mlx5/gdaki/configure.m4\])#d' "$UCX_SRC/src/uct/ib/mlx5/configure.m4" 2>/dev/null || true
    rm -rf "$UCX_SRC/src/uct/ib/mlx5/gdaki"
    sed -i '/^SUBDIRS = / s/ rdu / /' "$UCX_SRC/src/uct/Makefile.am" 2>/dev/null || true
    sed -i '\#m4_include(\[src/uct/rdu/configure.m4\])#d' "$UCX_SRC/src/uct/configure.m4" 2>/dev/null || true

    ( cd "$UCX_SRC"
      autoreconf -fiv 2>&1 | grep -E "error:|warning:|Leaving" | head -5 || autoreconf -fiv
      ./configure \
          --prefix="$UCX_INSTALL" \
          --enable-shared --disable-static --enable-mt \
          $CUDA_ARG \
          --without-java --without-go --without-rocm \
          --without-gdrcopy --without-valgrind \
          --without-knem --without-efa --without-mpi \
          --disable-doxygen-doc --enable-optimizations \
          MPICC= 2>&1 | grep -E "UCT modules|UCM modules|IB modules|error:|configure:" | head -10
      make -j"$NPROC" install 2>&1 | grep -E "error:|warning:|libuct" | head -10 || make -j"$NPROC" install
    )
    echo "UCX CUDA transport: $(ls $UCX_INSTALL/lib/ucx/libuct_cuda*.so 2>/dev/null | wc -l) .so files"
fi

# ── Step 2: Build NIXL wheel ─────────────────────────────────────────────────
NIXL_WHL=$(find "$NIXL_WHEEL_DIR" -name "nixl_cu12-*.whl" 2>/dev/null | head -1 || true)
if [ -n "$NIXL_WHL" ] && [ "${SKIP_NIXL:-0}" = "1" ]; then
    echo "=== Step 2: NIXL wheel exists (SKIP_NIXL=1): $NIXL_WHL ==="
else
    echo "=== Step 2: Building NIXL wheel from andychensn/nixl@$NIXL_BRANCH ==="
    NIXL_SRC="$BUILD_TMP/nixl"
    rm -rf "$NIXL_SRC"
    git clone --depth=50 --branch "$NIXL_BRANCH" \
        https://github.com/andychensn/nixl.git "$NIXL_SRC"
    git -C "$NIXL_SRC" checkout "$NIXL_COMMIT"

    mkdir -p "$NIXL_WHEEL_DIR"
    echo "  Installing meson build deps..."
    python3.12 -m pip install --user --break-system-packages \
        meson-python pybind11 patchelf pyyaml types-PyYAML setuptools build wheel \
        2>&1 | tail -3

    export LIBRARY_PATH="$UCX_INSTALL/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$UCX_INSTALL/lib:${LD_LIBRARY_PATH:-}"
    export PKG_CONFIG_PATH="$UCX_INSTALL/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

    echo "  Building NIXL wheel..."
    cd "$NIXL_SRC"
    python3.12 -m pip wheel . --no-deps -w "$NIXL_WHEEL_DIR" \
        --config-settings=setup-args="-Ducx_path=$UCX_INSTALL" \
        --config-settings=setup-args="-Denable_plugins=UCX" \
        --config-settings=setup-args="-Dcuda_path=$CUDA_HOME"
    cd "$REPO_ROOT"

    NIXL_WHL=$(ls "$NIXL_WHEEL_DIR"/nixl_cu12-*.whl 2>/dev/null | head -1)
    echo "NIXL wheel: $NIXL_WHL"
fi

# ── Step 3: Create Python venv ───────────────────────────────────────────────
echo "=== Step 3: Creating venv at $VENV ==="
python3.12 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

# ── Step 4: Install Python packages ─────────────────────────────────────────
echo "=== Step 4: torch $TORCH_VERSION+$TORCH_CUDA ==="
pip install -q "torch==$TORCH_VERSION" torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo "=== Step 4: vllm $VLLM_VERSION ==="
VLLM_USE_PRECOMPILED=1 pip install -q "vllm==$VLLM_VERSION"

echo "=== Step 4: sn_vllm patch check ==="
# Official vllm 0.16.0 already includes REGISTER_CONSUMER_MSG (verified 2026-06-28)
NIXL_PY=$(find "$VENV" -name "nixl_connector.py" -path "*/kv_connector*" 2>/dev/null | head -1)
if grep -q "REGISTER_CONSUMER_MSG" "$NIXL_PY" 2>/dev/null; then
    echo "  REGISTER_CONSUMER_MSG present in official vllm 0.16.0 ✅"
else
    echo "ERROR: REGISTER_CONSUMER_MSG not found in nixl_connector.py"
    echo "  Expected vllm 0.16.0 — check VLLM_VERSION in versions.env"
    exit 1
fi

echo "=== Step 4: nixl (custom UCX build) ==="
export LD_LIBRARY_PATH="$UCX_INSTALL/lib:${LD_LIBRARY_PATH:-}"
pip install -q "$NIXL_WHL"
# Install stub so 'import nixl' resolves without symlink
pip install -q --no-deps nixl==1.0.0 2>/dev/null || true

echo "=== Step 4: ai-dynamo[vllm] $DYNAMO_VERSION ==="
pip install -q "ai-dynamo[vllm]==$DYNAMO_VERSION"

echo "=== Step 4: flashinfer $FLASHINFER_VERSION ==="
pip install -q "flashinfer-python==$FLASHINFER_VERSION" \
    --find-links "https://flashinfer.ai/whl/cu130/torch2.9/" 2>/dev/null || \
pip install -q "flashinfer-python==$FLASHINFER_VERSION"

echo "=== Step 4: deep-gemm @$DEEPGEMM_COMMIT ==="
# --no-build-isolation: setup.py imports torch; it must find the venv's torch
pip install -q --no-build-isolation \
    "deep-gemm @ git+https://github.com/deepseek-ai/DeepGEMM.git@$DEEPGEMM_COMMIT"

# ── Step 5: Validate ─────────────────────────────────────────────────────────
echo ""
echo "=== Step 5: Validating ==="
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$UCX_INSTALL/lib:${LD_LIBRARY_PATH:-}"
python -c "import torch; print(f'torch: {torch.__version__} | cuda: {torch.version.cuda}')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import deep_gemm; print(f'deep_gemm: {deep_gemm.__version__} ✅')"
# is_deep_gemm_supported() needs GPU runtime — skip at build time, check at launch
python -c "
import pathlib, vllm
nc = pathlib.Path(vllm.__file__).parent / 'distributed/kv_transfer/kv_connector/v1/nixl_connector.py'
assert 'REGISTER_CONSUMER_MSG' in nc.read_text(), 'REGISTER_CONSUMER_MSG not found in nixl_connector.py'
print('nixl_connector REGISTER_CONSUMER_MSG: ✅')
"
python -c "from nixl._api import nixl_agent; print('nixl: ✅')"
echo "Note: is_deep_gemm_supported() requires GPU at runtime — verify at launch with VLLM logs"

echo ""
echo "=== GPU venv build COMPLETE $(date) ==="
echo "    Venv: $VENV"
echo "    UCX:  $UCX_INSTALL"
echo "    NIXL: $NIXL_WHL"
