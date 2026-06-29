#!/usr/bin/env bash
# Build UCX 1.22 (no CUDA) and NIXL pathb wheel for the RDU decode side.
#
# s339 has no internet access, so this script must be run in two phases:
#
# Phase 1 (login node — needs internet):
#   bash scripts/build_rdu_ucx_nixl.sh --fetch-only
#   Clones UCX and NIXL sources to $REPO_ROOT/rdu-build-src/ on NFS.
#
# Phase 2 (sc3-s339 via snrdu — uses NFS clone, no internet needed):
#   source config/cluster.env
#   snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
#       --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:30:00 \
#       -o logs/build_rdu_ucx_nixl.log \
#       -- bash scripts/build_rdu_ucx_nixl.sh --build-only
#
# Or run both phases from the login node (Phase 1 runs inline, Phase 2 via snrdu):
#   source config/cluster.env
#   bash scripts/build_rdu_ucx_nixl.sh
#
# Outputs:
#   $REPO_ROOT/rdu-ucx-install/    — UCX (CPU-only, bnxt_re verbs, no CUDA)
#   $REPO_ROOT/wheelhouse/nixl_cu12-*cp311*.whl  — NIXL wheel for s339 python3.11
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

SRC_DIR="$REPO_ROOT/rdu-build-src"
UCX_INSTALL="$REPO_ROOT/rdu-ucx-install"
WHEEL_OUT="$REPO_ROOT/wheelhouse"
NPROC=$(nproc 2>/dev/null || echo 4)
PY=/opt/sambanova/bin/python3.11

MODE="${1:-both}"   # --fetch-only | --build-only | both

# ── Phase 1: Clone sources to NFS (login node, needs internet) ───────────────
fetch_sources() {
    echo "=== Phase 1: Fetching UCX and NIXL sources to $SRC_DIR ==="
    mkdir -p "$SRC_DIR"

    if [ ! -d "$SRC_DIR/ucx/.git" ]; then
        echo "  Cloning andychensn/ucx@$UCX_COMMIT..."
        git clone --depth=200 --branch "$UCX_BRANCH" \
            https://github.com/andychensn/ucx.git "$SRC_DIR/ucx"
        git -C "$SRC_DIR/ucx" checkout "$UCX_COMMIT"
        echo "  UCX cloned ✅"
    else
        echo "  UCX already present in $SRC_DIR/ucx"
    fi

    if [ ! -d "$SRC_DIR/nixl/.git" ]; then
        echo "  Cloning andychensn/nixl@$NIXL_COMMIT..."
        git clone --depth=50 --branch "$NIXL_BRANCH" \
            https://github.com/andychensn/nixl.git "$SRC_DIR/nixl"
        git -C "$SRC_DIR/nixl" checkout "$NIXL_COMMIT"
        echo "  NIXL cloned ✅"
    else
        echo "  NIXL already present in $SRC_DIR/nixl"
    fi
}

# ── Phase 2: Build on s339 (uses NFS clone, no internet needed) ──────────────
build_on_s339() {
    echo "=== Phase 2: Building UCX + NIXL on $(hostname) $(date) ==="
    [ -x "$PY" ] || { echo "ERROR: $PY not found — must run on s339"; exit 1; }
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
              MPICC= 2>&1 | tail -5
          make -j"$NPROC" install 2>&1 | tail -3
        )
        echo "UCX IB transports: $(ls $UCX_INSTALL/lib/ucx/libuct_ib*.so 2>/dev/null | wc -l)"
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

        "$PY" -m pip install --user --break-system-packages \
            meson-python pybind11 patchelf pyyaml types-PyYAML setuptools build wheel 2>&1 | tail -3

        export LIBRARY_PATH="$UCX_INSTALL/lib:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="$UCX_INSTALL/lib:${LD_LIBRARY_PATH:-}"
        export PKG_CONFIG_PATH="$UCX_INSTALL/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

        cd "$NIXL_SRC"
        "$PY" -m pip wheel . --no-deps -w "$WHEEL_OUT" \
            --config-settings=setup-args="-Ducx_path=$UCX_INSTALL" \
            --config-settings=setup-args="-Denable_plugins=UCX" 2>&1 | tail -10
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
        build_on_s339 ;;
    both|"")
        fetch_sources
        echo ""
        echo "Sources fetched to $SRC_DIR. Now submit the build job on s339:"
        echo ""
        echo "  source config/cluster.env"
        echo "  snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \\"
        echo "      --reservation no_sf_catchup_demos --pef \"\$PEF\" --timeout 00:30:00 \\"
        echo "      -o logs/build_rdu_ucx_nixl.log \\"
        echo "      -- bash $REPO_ROOT/scripts/build_rdu_ucx_nixl.sh --build-only"
        ;;
    *)
        echo "Usage: $0 [--fetch-only | --build-only | (no arg = print instructions)]"
        exit 1 ;;
esac
