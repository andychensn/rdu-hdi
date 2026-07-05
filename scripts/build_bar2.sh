#!/usr/bin/env bash
# Produces the wheels under wheelhouse/ and the libs under
# rdu-runtime-install/ that docker/rdu-decode-install-deps.sh and
# Dockerfile.rdu bake into the RDU decode image.
#
# Build coe_api/rdu_engine (Python wheel) and the runtime connector libs
# (libc_samba_runtime.so/libcpp_samba_runtime.so) from a pinned commit of
# SambaNova/software (config/versions.env's SOFTWARE_REPO_* vars),
# replacing the ad-hoc NFS trees BAR2_INSTALL/BAR2_RUNTIME_LIBS/BAR2_PRELOAD
# used to point at.
#
# Two phases, matching every other RDU build script in this repo:
#
# Phase 1 (login node — needs internet):
#   bash scripts/build_bar2.sh --fetch-only
#
# Phase 2 (RDU node via snrdu, INSIDE the rhel810-dev container — see below
# for why bare metal doesn't work):
#   source config/cluster.env; source config/model.env
#   snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
#       --allow-local-lib-python --reservation "$RDU_RESERVATION" \
#       --pef "$PEF" --timeout "$RDU_TIMEOUT" \
#       -o logs/build_bar2.log \
#       -- bash scripts/build_bar2.sh --build-only
#
# NOTE: `source a.env b.env` only sources a.env — bash's `source` treats
# extra args as $1.. for the sourced script, not additional files. Always
# source config files as separate statements.
#
# Why the container and not bare metal: this repo's Bazel cc_toolchain
# hardcodes gcc-toolset-13 for libstdc++ headers (bazel/cc/sn_cc_rules.bzl:
# GCC_TOOLSET_VERSION = "13", needed for C++20 <format> support specifically
# — confirmed missing from gcc-toolset-12 by direct test, not just a wrong
# search path). RDU nodes we've checked (e.g. sc3-s339) only have
# gcc-toolset-12 installed on bare metal and installing 13 needs root we
# don't have — but SambaNova's own `rhel810-dev` dev container already has
# gcc-toolset-13 (confirmed by direct inspection), so --build-only re-execs
# itself inside that container automatically. `docker-run-wrapper` (present
# on RDU nodes, unlike the login node's build/push-only `docker-wrapper`)
# auto-mounts /import and /scratch at identical paths, so no explicit -v
# mount is needed (and is in fact blocked by the wrapper for security).
# Outputs:
#   $REPO_ROOT/bar2-build-src/software/   — pinned SambaNova/software checkout
#   $REPO_ROOT/wheelhouse/                — sambanova_rdu_engine_api-*.whl (rdu_engine)
#                                            and sambanova_coe_api-*.whl (coe_api compat shim)
#   $REPO_ROOT/rdu-runtime-install/lib/   — libc_samba_runtime.so, libcpp_samba_runtime.so,
#                                            libLlvm21.so (coe_api's runtime — not build-time — dep)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

SRC_DIR="$REPO_ROOT/bar2-build-src"
SOFTWARE_SRC="$SRC_DIR/software"
WHEELHOUSE="$REPO_ROOT/wheelhouse"
RUNTIME_INSTALL="$REPO_ROOT/rdu-runtime-install"
BAZEL_OUTPUT_BASE="/scratch/$USER/bazel-cache-rdu-hdi"
BAZELISK_BIN="$REPO_ROOT/vendor/bin/bazelisk"

MODE="${1:-both}"   # --fetch-only | --build-only | both

# Neither the login node's nor any RDU node's default PATH has `bazel` — and
# even where a personal one exists (e.g. a login-node user's own ~/bin), it's
# node-local and won't be there under snrdu on a different node. bazelisk
# reads the software repo's own .bazelversion and downloads/caches the exact
# matching bazel release, so this is fetched once and reused across nodes via
# this repo's own NFS-shared vendor/ dir.
ensure_bazelisk() {
    if [ -x "$BAZELISK_BIN" ]; then
        return
    fi
    echo "  fetching bazelisk $BAZELISK_VERSION (bazel itself isn't on PATH on this node)..."
    mkdir -p "$(dirname "$BAZELISK_BIN")"
    curl -sL -o "$BAZELISK_BIN" "$BAZELISK_URL"
    echo "$BAZELISK_SHA256  $BAZELISK_BIN" | sha256sum -c - || { echo "ERROR: bazelisk checksum mismatch"; exit 1; }
    chmod +x "$BAZELISK_BIN"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: fetch the pinned commit (login node, needs internet)
# ═══════════════════════════════════════════════════════════════════════════
fetch_sources() {
    echo "=== Phase 1: Fetching SambaNova/software @ $SOFTWARE_REPO_COMMIT ==="
    mkdir -p "$SRC_DIR"

    NEED_CLONE=1
    if [ -d "$SOFTWARE_SRC/.git" ]; then
        CURRENT_SHA=$(git -C "$SOFTWARE_SRC" rev-parse HEAD)
        if [ "$CURRENT_SHA" = "$SOFTWARE_REPO_COMMIT" ]; then
            echo "  already at pinned commit, skipping clone"
            NEED_CLONE=0
        else
            echo "  present but at wrong commit ($CURRENT_SHA) — re-fetching"
        fi
    fi

    if [ "$NEED_CLONE" = "1" ]; then
        rm -rf "$SOFTWARE_SRC"
        git clone --branch "$SOFTWARE_REPO_BRANCH" --depth 1 "$SOFTWARE_REPO_URL" "$SOFTWARE_SRC"
        ACTUAL_SHA=$(git -C "$SOFTWARE_SRC" rev-parse HEAD)
        if [ "$ACTUAL_SHA" != "$SOFTWARE_REPO_COMMIT" ]; then
            echo "  branch tip has moved: expected $SOFTWARE_REPO_COMMIT, got $ACTUAL_SHA"
            echo "  fetching the exact pinned commit explicitly..."
            git -C "$SOFTWARE_SRC" fetch --depth 1 origin "$SOFTWARE_REPO_COMMIT"
            git -C "$SOFTWARE_SRC" checkout "$SOFTWARE_REPO_COMMIT"
        fi
        echo "  checked out $(git -C "$SOFTWARE_SRC" rev-parse HEAD)"
    fi

    apply_local_patches
}

# The pinned commit needs one additional local patch (adds
# CoETensor/RDUTensor.dtype -- see config/versions.env's SOFTWARE_REPO_*
# comment for why this matters: fast-coe's pipeline.py was validated against
# a coe_api build WITH this attribute), captured as our own patch file so
# the build doesn't depend on any one engineer's personal checkout surviving.
apply_local_patches() {
    local PATCH="$REPO_ROOT/patches/software-repo/coe_api_rdutensor_dtype.patch"
    [ -f "$PATCH" ] || { echo "ERROR: $PATCH not found"; exit 1; }
    if git -C "$SOFTWARE_SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "  coe_api_rdutensor_dtype.patch already applied, skipping"
        return
    fi
    git -C "$SOFTWARE_SRC" apply "$PATCH"
    echo "  applied patches/software-repo/coe_api_rdutensor_dtype.patch"
}

RHEL810_DEV_IMAGE="artifacts.sambanovasystems.com/sw-docker/rhel810-dev:latest"

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: build. Bare RDU-node metal has /opt/sambanova (mpich/llvm19/torch)
# but only gcc-toolset-12, not the gcc-toolset-13 this repo's Bazel config
# requires — so build_on_rdu_node() re-execs itself inside rhel810-dev
# automatically when gcc-toolset-13 isn't found on the host.
# ═══════════════════════════════════════════════════════════════════════════
build_coe_api_wheel() {
    echo "=== Building coe_api/rdu_engine wheel (Bazel) $(date) ==="
    [ -d "$SOFTWARE_SRC/.git" ] || { echo "ERROR: $SOFTWARE_SRC not found — run --fetch-only first"; exit 1; }
    [ -d /opt/rh/gcc-toolset-13 ] || { echo "ERROR: /opt/rh/gcc-toolset-13 not found — must run inside rhel810-dev (see build_on_rdu_node)"; exit 1; }
    ensure_bazelisk

    mkdir -p "$BAZEL_OUTPUT_BASE" "$WHEELHOUSE"
    # bazelisk's own download cache (the fetched bazel binary itself, separate
    # from --output_base above which is bazel's build/action cache) defaults
    # under $HOME, which is NFS here — same NFS-cache-is-slow lesson learned
    # earlier this project with bazel's own output_base. Force it to scratch.
    export XDG_CACHE_HOME="/scratch/$USER/.cache"
    cd "$SOFTWARE_SRC"
    # bazelisk auto-detects the required version from this repo's own
    # .bazelversion (8.5.1) and downloads/caches that exact bazel release.
    #
    # --strategy=CppCompile=local: kept as a defensive default even though
    # the actual root cause of this build's earlier failures (bare RDU nodes
    # only having gcc-toolset-12, not the gcc-toolset-13 this repo's
    # bazel/cc/sn_cc_rules.bzl hardcodes for C++20 <format> support — see
    # build_on_rdu_node's re-exec-into-rhel810-dev logic above) turned out to
    # be unrelated to sandboxing at all. Harmless to leave on: we're building
    # on one known, controlled environment, not chasing cross-machine Bazel
    # sandbox hermeticity.
    # rdu_engine_py311_wheel: the real pybind module (rdu_engine.*.so).
    # coe_api_py311_wheel: a *separate* Bazel target — a thin backward-compat
    # shim (coe_api.*.so, built from py_coe_api_compat.cpp) that re-exports
    # rdu_engine under the legacy `coe_api` name. All of fast-coe/vllm-rdu's
    # production code imports `coe_api`, not `rdu_engine` — known-working
    # BAR2_INSTALL trees ship both .so files side by side, so both must be
    # built here too or `import coe_api` silently falls back to whatever
    # ambient system install happens to be on the node.
    "$BAZELISK_BIN" --output_base="$BAZEL_OUTPUT_BASE" build -c opt \
        --strategy=CppCompile=local \
        //frontend/nova/coe_api:rdu_engine_py311_wheel \
        //frontend/nova/coe_api:coe_api_py311_wheel

    for target in //frontend/nova/coe_api:rdu_engine_py311_wheel //frontend/nova/coe_api:coe_api_py311_wheel; do
        WHEEL=$("$BAZELISK_BIN" --output_base="$BAZEL_OUTPUT_BASE" cquery --output=files \
            "$target" 2>/dev/null | tail -1)
        [ -f "$WHEEL" ] || { echo "ERROR: expected wheel not found for $target at $WHEEL"; exit 1; }
        # -f: bazel's own build outputs are read-only, and a re-run (e.g. from a
        # different container invocation, possibly a different UID) can't
        # overwrite a previous run's copy without forcing it.
        rm -f "$WHEELHOUSE/$(basename "$WHEEL")"
        cp "$WHEEL" "$WHEELHOUSE/"
        echo "  copied $(basename "$WHEEL") to $WHEELHOUSE/"
    done
}

# rdu_engine.so dynamically loads these at IMPORT time — not needed at
# build time, which is why the earlier "does coe_api need the full compiler"
# analysis in DOCKERIZE_BAR2_PLAN.md missed them (that analysis only traced
# *build-time* deps). Discovered by actually importing the built wheel and
# iterating on ImportErrors, then getting the complete list at once via
# `ldd rdu_engine.cpython-311-x86_64-linux-gnu.so | grep "not found"`
# instead of continuing one error at a time. None of these are bundled by
# coe_api's own wheel rule — they're separate repo-wide "export bundle"
# cc_shared_library targets other binaries dynamic-link against.
EXTRA_RUNTIME_SHARED_LIBS=(
    "//bazel/third_party:llvm-21-so"             # libLlvm21.so
    "//bazel/third_party:abseil-so"              # libAbseil.so
    "//common/pef/src:lib-jit-function-so"       # libJITFunction.so
    "//common/pef/src:pef-bitfile-patching-so"   # libPefBitfilePatching.so
    "//common/pin:pin-compiler-so"               # libPinCompiler.so
)

build_extra_runtime_shared_libs() {
    echo "=== Building extra runtime-only shared libs (Bazel) $(date) ==="
    ensure_bazelisk
    mkdir -p "$BAZEL_OUTPUT_BASE" "$RUNTIME_INSTALL/lib"
    export XDG_CACHE_HOME="/scratch/$USER/.cache"
    cd "$SOFTWARE_SRC"
    "$BAZELISK_BIN" --output_base="$BAZEL_OUTPUT_BASE" build -c opt \
        --strategy=CppCompile=local \
        "${EXTRA_RUNTIME_SHARED_LIBS[@]}"

    for target in "${EXTRA_RUNTIME_SHARED_LIBS[@]}"; do
        LIB=$("$BAZELISK_BIN" --output_base="$BAZEL_OUTPUT_BASE" cquery --output=files \
            "$target" 2>/dev/null | tail -1)
        [ -f "$LIB" ] || { echo "ERROR: expected output not found for $target at $LIB"; exit 1; }
        rm -f "$RUNTIME_INSTALL/lib/$(basename "$LIB")"
        cp "$LIB" "$RUNTIME_INSTALL/lib/"
        echo "  copied $(basename "$LIB") to $RUNTIME_INSTALL/lib/"
    done
}

build_runtime_graph_libs() {
    echo "=== Building runtime 'graph' target group (CMake) $(date) ==="
    [ -d "$SOFTWARE_SRC/runtime" ] || { echo "ERROR: $SOFTWARE_SRC/runtime not found — run --fetch-only first"; exit 1; }

    # runtime/build.py's `#!/usr/bin/env python3` shebang resolves to the
    # ancient system python3 (3.6.8) on both bare RDU nodes AND inside
    # rhel810-dev if run directly. Also, build.py internally subprocess.Popen's
    # a bare "cmake" — same story: /opt/sambanova/bin/{python3.11,cmake} both
    # exist in both environments but aren't on PATH by default. Prepending to
    # PATH (rather than hardcoding one absolute path) fixes both invocation
    # styles at once. Needs the `distro` package too (not in the base install).
    export PATH="/opt/sambanova/bin:$PATH"
    PY311=/opt/sambanova/bin/python3.11
    [ -x "$PY311" ] || { echo "ERROR: $PY311 not found"; exit 1; }
    "$PY311" -m pip install --user -q distro

    cd "$SOFTWARE_SRC/runtime"
    # -rv ts16: this cluster's RDU hardware is Taurus16 (chip codename "sn40+").
    # Without -rv, build.py defaults to -rv gm (Gemini/sn20) and only builds
    # hal_snlib_sn20.so — silently wrong-architecture output that still links
    # and imports fine, but fails at actual RDU session creation with
    # `Unable to open low-level dynamic lib: libhal_snlib_sn40.so`
    # (hal_platform.c's dlopen() is runtime chip-detected, not build-time
    # gated). Known-working trees ship libhal_snlib_sn40+.so (WITH the plus,
    # confirming ts16/TAURUS16, not plain ts/TAURUS which would produce
    # sn40 without a plus — see runtime/src/lib/hal/hal_platform.c and
    # runtime/python/utils.py's
    # rdu_ver_to_sn_ver map).
    "$PY311" build.py -b graph -bt Release -rv ts16

    # Copy the entire build/graph/lib/ output, not a hand-picked subset --
    # `ldd` on libcpp_samba_runtime.so.4.13 shows it dynamically links
    # against ~20 sibling libraries also produced by this same "graph" CMake
    # target group (libsamba_ccl.so, libsn_lib.so, librduconnect.so,
    # libsamba_connector.so, libtransport.so, libdyn_comm_lib.so, etc.) --
    # missing any of them lets LD_LIBRARY_PATH silently fall through to a
    # mismatched version elsewhere instead of failing loudly.
    #
    # NOTE: do NOT `rm -rf "$RUNTIME_INSTALL/lib"` here -- build_on_rdu_node()
    # calls build_extra_runtime_shared_libs() (libAbseil.so/libLlvm21.so/etc.)
    # BEFORE this function, into this SAME directory. Wiping the whole dir
    # would delete those. Only remove this function's OWN prior output before
    # re-copying, so it's still idempotent across re-runs without clobbering
    # the other function's files.
    mkdir -p "$RUNTIME_INSTALL/lib"
    if [ -d build/graph/lib ]; then
        find build/graph/lib -maxdepth 1 -mindepth 1 -printf '%f\n' | while read -r f; do
            rm -rf "$RUNTIME_INSTALL/lib/$f"
        done
    fi
    cp -a build/graph/lib/. "$RUNTIME_INSTALL/lib/"
    echo "  copied $(ls build/graph/lib/ | wc -l) graph-group libs to $RUNTIME_INSTALL/lib/"
    ls "$RUNTIME_INSTALL/lib/"
}

# Bare-metal /opt/sambaflow's own libc_samba_runtime.so/libcpp_samba_runtime.so
# carry a baked-in DT_RPATH back to /opt/sambaflow -- RPATH beats
# LD_LIBRARY_PATH, so just adding our build to LD_LIBRARY_PATH is NOT enough
# to make it win. libNovaRuntime dlopen()s the UNVERSIONED name
# "libc_samba_runtime.so" (not the "libc_samba_runtime.so.4.13" this build
# actually produces), so the fix (same one hdi's own start_vllm_rdu_decode.sh
# uses) is: copy the .4.13 file, patch its own DT_SONAME to the unversioned
# name via patchelf, and force-load it via LD_PRELOAD -- LD_PRELOAD's own
# explicit unversioned name then wins the dlopen() regardless of RPATH.
build_preload_libs() {
    echo "=== Building LD_PRELOAD copies (SONAME-patched) $(date) ==="
    which patchelf >/dev/null 2>&1 || { echo "ERROR: patchelf not found (pip install --user patchelf, or dnf install patchelf)"; exit 1; }
    mkdir -p "$RUNTIME_INSTALL/preload"
    for base in libc_samba_runtime libcpp_samba_runtime; do
        SRC="$RUNTIME_INSTALL/lib/${base}.so.4.13"
        DST="$RUNTIME_INSTALL/preload/${base}.so"
        [ -f "$SRC" ] || { echo "ERROR: $SRC not found — run build_runtime_graph_libs first"; exit 1; }
        cp -f "$SRC" "$DST"
        patchelf --set-soname "${base}.so" "$DST"
        echo "  $DST: $(readelf -d "$DST" | grep SONAME)"
    done
}

build_on_rdu_node() {
    if [ ! -d /opt/rh/gcc-toolset-13 ] && [ -z "${BAR2_IN_CONTAINER:-}" ]; then
        echo "=== gcc-toolset-13 not on bare metal — re-executing inside $RHEL810_DEV_IMAGE ==="
        exec sudo -n -g docker /usr/bin/docker-run-wrapper --rm --entrypoint /bin/bash \
            -e "BAR2_IN_CONTAINER=1" \
            -e "HOME=/tmp" \
            "$RHEL810_DEV_IMAGE" \
            -c "cd '$REPO_ROOT' && bash scripts/build_bar2.sh --build-only"
    fi

    build_coe_api_wheel
    echo ""
    build_extra_runtime_shared_libs
    echo ""
    build_runtime_graph_libs
    echo ""
    build_preload_libs
    echo ""
    echo "=== BAR2 self-build COMPLETE $(date) ==="
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
        echo "Sources fetched. Now submit the build job on the RDU node:"
        echo ""
        echo "  source config/cluster.env; source config/model.env"
        echo "  snrdu run -sp \"\$RDU_PARTITION\" --qos \"\$RDU_QOS\" --nodelist \"\$RDU_NODE\" \\"
        echo "      --allow-local-lib-python --reservation \"\$RDU_RESERVATION\" \\"
        echo "      --pef \"\$PEF\" --timeout \"\$RDU_TIMEOUT\" \\"
        echo "      -o logs/build_bar2.log \\"
        echo "      -- bash scripts/build_bar2.sh --build-only"
        ;;
    *)
        echo "Usage: $0 [--fetch-only|--build-only]"; exit 1 ;;
esac
