#!/usr/bin/env bash
# Build UCX 1.22 (no CUDA) and NIXL pathb wheel for the RDU decode side.
#
# This script runs in two phases:
#
# (Note 2026-07-03: empirically, sc3-s339 DOES have internet access — `curl
# https://pypi.org`, `pip install`, and `git clone` all work fine from it.
# The two-phase split below predates that finding and isn't a hard
# requirement anymore, but it's a proven-safe pattern already in place —
# kept as-is rather than churned for no functional benefit. Don't assume
# "no internet" as a hard constraint elsewhere without checking first.)
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

    # gguf + regex — needed by vllm.transformers_utils.gguf_utils (loaded at import time).
    # gguf is a pure-Python GGUF model reader; regex is its C-ext string library.
    # Neither is in the SambaNova system Python. Not used at runtime for HF models,
    # but the module-level import in vllm 0.16.0 means both must be present.
    if ! find "$WHEEL_OUT" -name "gguf-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading gguf..."
        python3.12 -m pip download "gguf>=0.17.0" --only-binary=:all: --no-deps --dest "$WHEEL_OUT" 2>&1 | tail -2
        echo "  gguf wheel ✅"
    else
        echo "  gguf wheel already present"
    fi
    if ! find "$WHEEL_OUT" -name "regex-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading regex..."
        python3.12 -m pip download "regex" --only-binary=:all: --no-deps \
            --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEEL_OUT" 2>&1 | tail -2
        echo "  regex wheel ✅"
    else
        echo "  regex wheel already present"
    fi

    # av (PyAV) — needed by fast-coe's rdu_manifest.vlm_pipeline (imported
    # unconditionally at module load, even for text-only models). C-extension
    # package (bundles ffmpeg libs), so must be a prebuilt wheel, not vendored
    # source. Not in the SambaNova system Python.
    if ! find "$WHEEL_OUT" -name "av-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading av..."
        python3.12 -m pip download "av==12.3.0" --only-binary=:all: --no-deps \
            --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEEL_OUT" 2>&1 | tail -2
        echo "  av wheel ✅"
    else
        echo "  av wheel already present"
    fi

    # numpy pinned wheel — system s339 may have newer numpy (2.x) compiled against
    # incompatible ABI; we install 1.26.4 into the venv to override it.
    if ! find "$WHEEL_OUT" -name "numpy-${RDU_NUMPY_VERSION}-cp311-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading numpy==$RDU_NUMPY_VERSION (cp311 wheel for s339)..."
        python3.12 -m pip download "numpy==$RDU_NUMPY_VERSION" --only-binary=:all: --no-deps \
            --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEEL_OUT" 2>&1 | tail -2
        echo "  numpy wheel ✅"
    else
        echo "  numpy wheel already present"
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

    # rdma-core devel headers — RDU nodes ship libibverbs/librdmacm runtime
    # .so.1 (confirmed matching this exact version via SONAME 1.14.48.0 /
    # 1.3.48.0) but not the -devel headers, and we can't install system
    # packages there. Only public headers are needed to compile UCX's verbs
    # transport against (internal/provider headers are not required by
    # external consumers of the library). Fetched here instead of searched
    # for on the RDU node — searching the whole /import NFS tree for
    # verbs.h took well over 90s just for two subdirectories in testing
    # and was the actual root cause of hours-long "hangs" during --build-only.
    RDMA_HDRS="$SRC_DIR/rdma-core-headers"
    if [ ! -f "$RDMA_HDRS/infiniband/verbs.h" ]; then
        echo "  Downloading rdma-core $RDMA_CORE_VERSION headers..."
        RDMA_TGZ="$SRC_DIR/rdma-core-${RDMA_CORE_VERSION}.tar.gz"
        curl -sL -o "$RDMA_TGZ" "$RDMA_CORE_URL"
        echo "$RDMA_CORE_SHA256  $RDMA_TGZ" | sha256sum -c - || { echo "ERROR: rdma-core tarball checksum mismatch"; exit 1; }
        mkdir -p "$RDMA_HDRS/infiniband" "$RDMA_HDRS/rdma"
        tar xzf "$RDMA_TGZ" --strip-components=2 -C "$RDMA_HDRS/infiniband" \
            "rdma-core-${RDMA_CORE_VERSION}/libibverbs/arch.h" \
            "rdma-core-${RDMA_CORE_VERSION}/libibverbs/opcode.h" \
            "rdma-core-${RDMA_CORE_VERSION}/libibverbs/sa-kern-abi.h" \
            "rdma-core-${RDMA_CORE_VERSION}/libibverbs/sa.h" \
            "rdma-core-${RDMA_CORE_VERSION}/libibverbs/verbs.h" \
            "rdma-core-${RDMA_CORE_VERSION}/libibverbs/verbs_api.h" \
            "rdma-core-${RDMA_CORE_VERSION}/libibverbs/tm_types.h"
        tar xzf "$RDMA_TGZ" --strip-components=2 -C "$RDMA_HDRS/infiniband" \
            "rdma-core-${RDMA_CORE_VERSION}/librdmacm/acm.h" \
            "rdma-core-${RDMA_CORE_VERSION}/librdmacm/ib.h"
        tar xzf "$RDMA_TGZ" --strip-components=3 -C "$RDMA_HDRS/infiniband" \
            "rdma-core-${RDMA_CORE_VERSION}/kernel-headers/rdma/ib_user_ioctl_verbs.h"
        tar xzf "$RDMA_TGZ" --strip-components=2 -C "$RDMA_HDRS/rdma" \
            "rdma-core-${RDMA_CORE_VERSION}/librdmacm/rdma_cma.h" \
            "rdma-core-${RDMA_CORE_VERSION}/librdmacm/rdma_cma_abi.h" \
            "rdma-core-${RDMA_CORE_VERSION}/librdmacm/rdma_verbs.h" \
            "rdma-core-${RDMA_CORE_VERSION}/librdmacm/rsocket.h"
        rm -f "$RDMA_TGZ"
        echo "  rdma-core headers ✅"
    else
        echo "  rdma-core headers already present"
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
            # Headers pre-fetched by --fetch-only (see fetch_sources) — NOT
            # searched for here. A blind `find /import -name verbs.h` over
            # the whole multi-TB NFS scratch tree was measured taking well
            # over 90s for just two subdirectories and was the actual root
            # cause of hours-long apparent "hangs" in this build step.
            RDMA_HDRS="$SRC_DIR/rdma-core-headers"
            if [ -f "$RDMA_HDRS/infiniband/verbs.h" ]; then
                echo "  Using rdma-core headers from $RDMA_HDRS"
                VERBS_EXTRA_FLAGS="CPPFLAGS=-I$RDMA_HDRS"
            else
                echo "  WARNING: $RDMA_HDRS/infiniband/verbs.h not found (run --fetch-only first) — IB transport will be disabled"
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

        # Unbuffered, timestamped output — a plain `| tail -N` here fully
        # buffers all output until each step's EOF, so SLURM logs showed zero
        # progress for the entire step no matter how long it took (measured:
        # autoreconf ~12s, configure ~11s, full `make -j8 install` well under
        # 2 minutes — none of these are actually slow; only their output was hidden).
        ( cd "$UCX_SRC"
          stdbuf -oL -eL autoreconf -fiv 2>&1 | stdbuf -oL awk '{ print strftime("[%H:%M:%S]"), $0; fflush(); }'
          stdbuf -oL -eL ./configure \
              --prefix="$UCX_INSTALL" \
              --enable-shared --disable-static --enable-mt \
              --without-cuda \
              --without-java --without-go --without-rocm \
              --without-gdrcopy --without-valgrind \
              --without-knem --without-efa --without-mpi \
              --disable-doxygen-doc --enable-optimizations \
              MPICC= $VERBS_EXTRA_FLAGS 2>&1 | stdbuf -oL awk '{ print strftime("[%H:%M:%S]"), $0; fflush(); }'
          stdbuf -oL -eL make -j"$NPROC" install 2>&1 | stdbuf -oL awk '{ print strftime("[%H:%M:%S]"), $0; fflush(); }'
        )
        IB_COUNT=$(ls $UCX_INSTALL/lib/ucx/libuct_ib*.so 2>/dev/null | wc -l)
        echo "UCX IB transports: $IB_COUNT"
        [ "$IB_COUNT" -eq 0 ] && echo "WARNING: No IB transports — RDMA/RoCE will not work. Run '$0 --fetch-only' first to fetch rdma-core headers (there's no package manager access on the RDU node to install rdma-core-devel directly)."
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
        # GCC 8 (RHEL8) needs:
        #   -Wno-attributes: [[likely]]/[[unlikely]] are C++20, treated as errors by -Werror
        #   -lstdc++fs: std::filesystem is a separate lib before GCC 9
        # Meson picks these up from env vars (multiple --config-settings=setup-args= overrides each other)
        CXXFLAGS="-Wno-attributes" \
        LDFLAGS="-lstdc++fs" \
        "$PY" -m pip wheel . --no-deps --no-build-isolation -w "$WHEEL_OUT" \
            --config-settings=setup-args="-Ducx_path=$UCX_INSTALL" \
            --config-settings=setup-args="-Denable_plugins=UCX"
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
