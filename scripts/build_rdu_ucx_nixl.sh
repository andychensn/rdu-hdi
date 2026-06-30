#!/usr/bin/env bash
# Build UCX 1.22 (no CUDA) and NIXL pathb wheel for the RDU decode side.
#
# The RDU node has no internet access, so this script runs in two phases:
#
# Phase 1 (login node — needs internet):
#   bash scripts/build_rdu_ucx_nixl.sh --fetch-only
#
# Phase 2 (RDU node via snrdu — uses NFS clone, no internet needed):
#   source config/cluster.env config/model.env
#   snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
#       --allow-local-lib-python --reservation "$RDU_RESERVATION" \
#       --pef "$PEF" --timeout 00:30:00 \
#       -o logs/build_rdu_ucx_nixl.log \
#       -- bash scripts/build_rdu_ucx_nixl.sh --build-only
#
# Outputs:
#   $REPO_ROOT/rdu-ucx-install/    — UCX (CPU-only, bnxt_re verbs, no CUDA)
#   $REPO_ROOT/wheelhouse/nixl_cu12-*cp311*.whl  — NIXL wheel (RDU python3.11)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
# cluster.env + model.env needed for printed snrdu instructions (RDU_NODE, PEF, etc.)
[ -f "$REPO_ROOT/config/cluster.env" ] && source "$REPO_ROOT/config/cluster.env" || true
[ -f "$REPO_ROOT/config/model.env"   ] && source "$REPO_ROOT/config/model.env"   || true

SRC_DIR="$REPO_ROOT/rdu-build-src"
UCX_INSTALL="$REPO_ROOT/rdu-ucx-install"
WHEEL_OUT="$REPO_ROOT/wheelhouse"
NPROC=$(nproc 2>/dev/null || echo 4)
PY=/opt/sambanova/bin/python3.11

MODE="${1:-both}"   # --fetch-only | --build-only | both

# ── Phase 1: Fetch all internet resources (login node) ───────────────────────
fetch_sources() {
    echo "=== Phase 1: Fetching sources and wheels (login node, needs internet) ==="
    mkdir -p "$SRC_DIR" "$WHEEL_OUT"

    # UCX source
    if [ ! -d "$SRC_DIR/ucx/.git" ]; then
        echo "  Cloning andychensn/ucx@$UCX_COMMIT..."
        git clone --depth=200 --branch "$UCX_BRANCH" \
            https://github.com/andychensn/ucx.git "$SRC_DIR/ucx"
        git -C "$SRC_DIR/ucx" checkout "$UCX_COMMIT"
        echo "  UCX cloned ✅"
    else
        echo "  UCX already present"
    fi

    # NIXL source
    if [ ! -d "$SRC_DIR/nixl/.git" ]; then
        echo "  Cloning andychensn/nixl@$NIXL_COMMIT..."
        git clone --depth=50 --branch "$NIXL_BRANCH" \
            https://github.com/andychensn/nixl.git "$SRC_DIR/nixl"
        git -C "$SRC_DIR/nixl" checkout "$NIXL_COMMIT"
        echo "  NIXL cloned ✅"
    else
        echo "  NIXL already present"
    fi

    # vllm CPU wheel — downloaded on login node, installed on s339 (no internet there)
    # Rename manylinux→linux_x86_64 to bypass glibc 2.31 check on RHEL8/s339
    VLLM_WHL=$(find "$WHEEL_OUT" -name "vllm-*linux_x86_64.whl" 2>/dev/null | head -1 || true)
    if [ -z "$VLLM_WHL" ]; then
        echo "  Downloading vllm==$VLLM_VERSION wheel..."
        python3.12 -m pip download "vllm==$VLLM_VERSION" --no-deps --dest "$WHEEL_OUT" \
            --python-version 311 --platform manylinux_2_31_x86_64 2>&1 | tail -3
        MANYLINUX="$WHEEL_OUT/vllm-${VLLM_VERSION}-cp38-abi3-manylinux_2_31_x86_64.whl"
        [ -f "$MANYLINUX" ] && mv "$MANYLINUX" "$WHEEL_OUT/vllm-${VLLM_VERSION}-cp38-abi3-linux_x86_64.whl"
        echo "  vllm wheel ✅"
    else
        echo "  vllm wheel already present: $VLLM_WHL"
    fi

    # ai-dynamo-runtime (Rust compiled bindings) + ai-dynamo (Python app code)
    # Both downloaded with --no-deps; installed separately on s339 to avoid
    # pulling vllm/torch as pip dependencies (already installed from wheelhouse).
    if ! find "$WHEEL_OUT" -name "ai_dynamo_runtime-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading ai-dynamo-runtime==$DYNAMO_VERSION..."
        python3.12 -m pip download "ai-dynamo-runtime==$DYNAMO_VERSION" --only-binary=:all: --no-deps \
            --dest "$WHEEL_OUT" 2>&1 | tail -2
        echo "  ai-dynamo-runtime wheel ✅"
    else
        echo "  ai-dynamo-runtime wheel already present"
    fi
    if ! find "$WHEEL_OUT" -name "ai_dynamo-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading ai-dynamo==$DYNAMO_VERSION..."
        python3.12 -m pip download "ai-dynamo==$DYNAMO_VERSION" --only-binary=:all: --no-deps \
            --dest "$WHEEL_OUT" 2>&1 | tail -2
        echo "  ai-dynamo wheel ✅"
    else
        echo "  ai-dynamo wheel already present"
    fi
}

# ── Phase 2: Build on RDU node (uses NFS clone, no internet needed) ──────────
build_on_rdu_node() {
    echo "=== Phase 2: Building UCX + NIXL on $(hostname) $(date) ==="
    [ -x "$PY" ] || { echo "ERROR: $PY not found — must run on RDU node via snrdu"; exit 1; }
    [ -d "$SRC_DIR/ucx/.git" ] || { echo "ERROR: sources not fetched — run with --fetch-only from login node first"; exit 1; }
    for cmd in gcc make autoreconf libtoolize; do
        command -v "$cmd" >/dev/null || { echo "ERROR: $cmd not found"; exit 1; }
    done

    BUILD_TMP=$(mktemp -d /tmp/rdu-ucx-build-XXXX)
    trap "rm -rf $BUILD_TMP" EXIT

    # Step A: Build UCX (no CUDA, verbs for bnxt_re)
    if [ -f "$UCX_INSTALL/include/ucp/api/ucp.h" ] && [ "${SKIP_UCX:-0}" = "1" ]; then
        echo "=== UCX already built (SKIP_UCX=1) ==="
    else
        echo "=== Building UCX from $SRC_DIR/ucx ==="
        UCX_SRC="$BUILD_TMP/ucx"
        cp -r "$SRC_DIR/ucx" "$UCX_SRC"

        # Strip GPU-only modules
        sed -i 's/^SUBDIRS = \. gdaki/SUBDIRS = ./' "$UCX_SRC/src/uct/ib/mlx5/Makefile.am" 2>/dev/null || true
        sed -i '\#m4_include(\[src/uct/ib/mlx5/gdaki/configure.m4\])#d' "$UCX_SRC/src/uct/ib/mlx5/configure.m4" 2>/dev/null || true
        rm -rf "$UCX_SRC/src/uct/ib/mlx5/gdaki"
        sed -i '/^SUBDIRS = / s/ rdu / /' "$UCX_SRC/src/uct/Makefile.am" 2>/dev/null || true
        sed -i '\#m4_include(\[src/uct/rdu/configure.m4\])#d' "$UCX_SRC/src/uct/configure.m4" 2>/dev/null || true

        # rdma-core-devel (libibverbs headers + .so symlink) is required for IB/RoCE support.
        # On RDU nodes, only the runtime lib exists (.so.1) — not the devel package.
        # Create local symlinks so UCX configure can find them without sudo.
        VERBS_TMP="$BUILD_TMP/verbs_devel"
        mkdir -p "$VERBS_TMP"
        VERBS_EXTRA_FLAGS=""
        if [ ! -f /usr/include/infiniband/verbs.h ]; then
            # Try to find headers from an NFS rdma-core-devel install or guoyaof's build
            RDMA_HEADERS=$(find /import -name "verbs.h" -path "*/infiniband/*" 2>/dev/null | head -1)
            if [ -n "$RDMA_HEADERS" ]; then
                RDMA_INC=$(dirname "$(dirname "$RDMA_HEADERS")")
                echo "  Using rdma headers from $RDMA_INC"
                VERBS_EXTRA_FLAGS="CPPFLAGS=-I$RDMA_INC"
            else
                echo "  WARNING: infiniband/verbs.h not found — IB transport will be disabled"
            fi
        fi
        # Create .so symlink if only .so.1 exists (needed by configure's -libverbs link test)
        for LIB in libibverbs librdmacm; do
            if [ ! -f /usr/lib64/${LIB}.so ] && [ -f /usr/lib64/${LIB}.so.1 ]; then
                ln -sf /usr/lib64/${LIB}.so.1 "$VERBS_TMP/${LIB}.so"
            fi
        done
        [ -n "$(ls $VERBS_TMP/*.so 2>/dev/null)" ] && \
            VERBS_EXTRA_FLAGS="${VERBS_EXTRA_FLAGS} LDFLAGS=-L$VERBS_TMP"

        ( cd "$UCX_SRC"
          autoreconf -fiv 2>&1 | tail -3
          ./configure \
              --prefix="$UCX_INSTALL" \
              --enable-shared --disable-static --enable-mt \
              --without-cuda \
              --without-java --without-go --without-rocm \
              --without-gdrcopy --without-valgrind \
              --without-knem --without-efa --without-mpi \
              --disable-doxygen-doc --enable-optimizations \
              MPICC= $VERBS_EXTRA_FLAGS 2>&1 | tail -5
          make -j"$NPROC" install 2>&1 | tail -3
        )
        IB_COUNT=$(ls $UCX_INSTALL/lib/ucx/libuct_ib*.so 2>/dev/null | wc -l)
        echo "UCX IB transports: $IB_COUNT"
        [ "$IB_COUNT" -eq 0 ] && echo "WARNING: No IB transports — RDMA/RoCE will not work. Install rdma-core-devel on the RDU node."
    fi

    # Step B: Build NIXL pathb wheel
    NIXL_WHL=$(find "$WHEEL_OUT" -name "nixl_cu12*cp311*.whl" 2>/dev/null | head -1 || true)
    if [ -n "$NIXL_WHL" ] && [ "${SKIP_NIXL:-0}" = "1" ]; then
        echo "=== NIXL wheel exists (SKIP_NIXL=1): $NIXL_WHL ==="
    else
        echo "=== Building NIXL wheel from $SRC_DIR/nixl ==="
        NIXL_SRC="$BUILD_TMP/nixl"
        cp -r "$SRC_DIR/nixl" "$NIXL_SRC"
        mkdir -p "$WHEEL_OUT"

        # Install build tools into a local prefix we control (avoids HOME dependency)
        BUILD_TOOLS_DIR="$BUILD_TMP/build-tools"
        "$PY" -m pip install --prefix="$BUILD_TOOLS_DIR" \
            meson-python pybind11 patchelf pyyaml types-PyYAML setuptools build wheel ninja
        # Locate meson/ninja — pip may install to prefix or user base (e.g. ~/.local)
        USER_BASE=$("$PY" -c "import site; print(site.getuserbase())" 2>/dev/null || echo "")
        MESON_BIN=$(find "$BUILD_TOOLS_DIR" "$USER_BASE" -name "meson" -type f 2>/dev/null | head -1 || \
                    command -v meson 2>/dev/null || true)
        NINJA_BIN=$(find "$BUILD_TOOLS_DIR" "$USER_BASE" -name "ninja" -type f 2>/dev/null | head -1 || \
                    command -v ninja 2>/dev/null || true)
        [ -n "$MESON_BIN" ] && export PATH="$(dirname "$MESON_BIN"):$PATH" || { echo "ERROR: meson not found in $BUILD_TOOLS_DIR or $USER_BASE"; exit 1; }
        [ -n "$NINJA_BIN" ] && export PATH="$(dirname "$NINJA_BIN"):$PATH" || true
        # Add site-packages from prefix so meson-python module is importable
        SITE_PKG=$(find "$BUILD_TOOLS_DIR" -type d -name "site-packages" 2>/dev/null | head -1)
        [ -n "$SITE_PKG" ] && export PYTHONPATH="$SITE_PKG:${PYTHONPATH:-}"
        echo "  meson: $MESON_BIN  ninja: $NINJA_BIN"

        export LIBRARY_PATH="$UCX_INSTALL/lib:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="$UCX_INSTALL/lib:${LD_LIBRARY_PATH:-}"
        export PKG_CONFIG_PATH="$UCX_INSTALL/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

        cd "$NIXL_SRC"
        "$PY" -m pip wheel . --no-deps --no-build-isolation -w "$WHEEL_OUT" \
            --config-settings=setup-args="-Ducx_path=$UCX_INSTALL" \
            --config-settings=setup-args="-Denable_plugins=UCX" \
            --config-settings=setup-args="-Dcpp_args=-Wno-attributes"
        cd "$REPO_ROOT"

        NIXL_WHL=$(find "$WHEEL_OUT" -name "nixl*.whl" -newer "$BUILD_TMP" | head -1 || true)
        echo "NIXL wheel: $NIXL_WHL"
    fi

    echo ""
    echo "=== Done $(date) ==="
    echo "    UCX:  $UCX_INSTALL"
    echo "    NIXL: $NIXL_WHL"
    echo ""
    echo "Next: re-run scripts/build_rdu_venv.sh to pick up the new wheel"
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "$MODE" in
    --fetch-only)
        fetch_sources ;;
    --build-only)
        build_on_rdu_node ;;
    both|"")
        fetch_sources
        echo ""
        echo "Sources and wheels fetched. Now submit the build job on the RDU node:"
        echo ""
        echo "  snrdu run -sp ${RDU_PARTITION:-zd3} --qos ${RDU_QOS:-5} --nodelist ${RDU_NODE:-<RDU_NODE>} \\"
        echo "      --allow-local-lib-python ${RDU_RESERVATION:+--reservation $RDU_RESERVATION} \\"
        echo "      --pef ${PEF:-<PEF>} --timeout 00:30:00 \\"
        echo "      -o logs/build_rdu_ucx_nixl.log \\"
        echo "      -- bash $REPO_ROOT/scripts/build_rdu_ucx_nixl.sh --build-only"
        ;;
    *)
        echo "Usage: $0 [--fetch-only | --build-only | (no arg = print instructions)]"
        exit 1 ;;
esac
