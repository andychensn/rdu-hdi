#!/usr/bin/env bash
# Build the RDU-side artifacts rdu-hdi's Docker image needs: fast-coe
# source, UCX/NIXL (from source), and the vllm+cpu wheel.
#
# One script, not split across several — fetching fast-coe and building
# UCX/NIXL always run together in a fixed order, and splitting the "extra
# deps" step into its own script previously caused it to be silently
# forgotten before a launch.
#
# Two phases, matching every other RDU build script in this repo:
#
# Phase 1 (login node — needs internet):
#   bash build/rdu_env.sh --fetch-only
#
# Phase 2 (RDU node via snrdu — no internet needed once fetched):
#   source config/cluster.env config/model.env
#   snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
#       --allow-local-lib-python --reservation "$RDU_RESERVATION" \
#       --pef "$PEF" --timeout "$RDU_TIMEOUT" \
#       -o logs/build_rdu_env.log \
#       -- bash build/rdu_env.sh --build-only
#
# Outputs:
#   $REPO_ROOT/fast-coe/          — vllm-rdu source, pinned by commit
#   $REPO_ROOT/rdu-ucx-install/   — UCX (CPU-only, bnxt_re verbs, no CUDA)
#   $REPO_ROOT/wheelhouse/        — vllm+cpu, nixl, ai-dynamo(-runtime), and all
#                                   transitive-dep wheels
# All of the above get COPY'd into docker/rdu/Dockerfile by
# docker/rdu/install-deps.sh, which does its own install + patch +
# validation pass directly into the image — this script doesn't build or
# validate a venv itself.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
# cluster.env + model.env needed for printed snrdu instructions (RDU_NODE, PEF, etc.)
[ -f "$REPO_ROOT/config/cluster.env" ] && source "$REPO_ROOT/config/cluster.env" || true
[ -f "$REPO_ROOT/config/model.env"   ] && source "$REPO_ROOT/config/model.env"   || true

SRC_DIR="$REPO_ROOT/rdu-build-src"
UCX_INSTALL="$REPO_ROOT/rdu-ucx-install"
WHEELHOUSE="$REPO_ROOT/wheelhouse"
VLLM_SRC="$SRC_DIR/vllm-patched"
FAST_COE_SRC="$REPO_ROOT/fast-coe"
NPROC=$(nproc 2>/dev/null || echo 4)
PY=/opt/sambanova/bin/python3.11

MODE="${1:-both}"   # --fetch-only | --build-only | both

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: fetch everything (login node, needs internet)
#
# (Note: sc3-s339 does have internet access — curl/pip/git all work fine
# from it. The two-phase split below isn't a hard requirement, but it's a
# proven-safe pattern kept as-is. Don't assume "no internet" as a hard
# constraint on RDU nodes elsewhere without checking first.)
# ═══════════════════════════════════════════════════════════════════════════
fetch_sources() {
    echo "=== Phase 1: Fetching sources and wheels (login node, needs internet) ==="
    mkdir -p "$SRC_DIR" "$WHEELHOUSE"

    # ── fast-coe (vllm-rdu source) ────────────────────────────────────────────
    # server/vllm-rdu is the RDU decode connector/engine code
    # (rdu_hardware/connector_override.py: 4 independent per-NIC NIXL agents,
    # BAR2/DDR slab cache, cache-aware routing). This is a PYTHONPATH/
    # editable-install source tree, not a wheel — read directly off NFS at
    # build+run time, no separate build step needed.
    if [ -d "$FAST_COE_SRC/.git" ]; then
        CURRENT=$(git -C "$FAST_COE_SRC" rev-parse HEAD)
        if [ "$CURRENT" = "$FAST_COE_COMMIT" ]; then
            echo "  fast-coe already present at $FAST_COE_COMMIT ✅"
        else
            echo "ERROR: $FAST_COE_SRC exists but is at $CURRENT, expected $FAST_COE_COMMIT"
            echo "  Remove it and re-run to re-fetch, or checkout the pin manually."
            exit 1
        fi
    else
        echo "  Cloning sambanova/fast-coe@$FAST_COE_COMMIT..."
        git clone git@github.com:sambanova/fast-coe.git "$FAST_COE_SRC"
        git -C "$FAST_COE_SRC" checkout "$FAST_COE_COMMIT"
        echo "  fast-coe cloned + pinned ✅"
        for p in server/vllm-rdu/rdu_hardware/connector_override.py \
                 server/rdu_manifest \
                 server/inference-router/client-py/inference_router \
                 server/block_hash; do
            if [ -e "$FAST_COE_SRC/$p" ]; then
                echo "    OK: $p"
            else
                echo "    WARNING: missing expected path: $p"
            fi
        done
    fi

    # ── vllm source + torch 2.2.x compat patches ──────────────────────────────
    # PyPI vllm 0.16.0 targets torch 2.9.x and uses APIs unavailable in torch
    # 2.2.0+sn: _symmetric_memory, _unregister_process_group, custom_graph_pass,
    # infer_schema, etc. These patches (from patches/ in this repo) let a vllm
    # 0.16.0 source build run on s339's torch 2.2.0+sn.
    if [ ! -d "$VLLM_SRC/.git" ]; then
        echo "  Cloning vllm v$VLLM_VERSION..."
        mkdir -p "$(dirname "$VLLM_SRC")"
        git clone --depth=1 --branch "v$VLLM_VERSION" \
            https://github.com/vllm-project/vllm.git "$VLLM_SRC"
        echo "  vllm source cloned ✅"
    else
        echo "  vllm source already present at $VLLM_SRC"
    fi

    echo "  Applying torch 2.2.x compat patches to vllm source..."

    # Patch 1: env_override.py — replace minimal upstream version with comprehensive
    # torch 2.2.x compat shim that pre-registers stubs for _symmetric_memory,
    # _unregister_process_group, custom_graph_pass, and ~10 other torch 2.5+/2.9.x
    # symbols via sys.modules injection before any vllm submodule is imported.
    PATCH1="$REPO_ROOT/patches/rdu/vllm_env_override_torch22x.py"
    [ -f "$PATCH1" ] || { echo "ERROR: $PATCH1 not found"; exit 1; }
    cp "$PATCH1" "$VLLM_SRC/vllm/env_override.py"
    echo "    env_override.py: replaced with torch 2.2.x compat shim ✅"

    # Patch 2: torch_utils.py — add _infer_schema_available flag so
    # direct_register_custom_op skips op registration on torch 2.2.x
    # (infer_schema not available → schema_str="" → define() fails with bare op name).
    TORCH_UTILS="$VLLM_SRC/vllm/utils/torch_utils.py"
    if grep -q "^from torch.library import Library, infer_schema" "$TORCH_UTILS" 2>/dev/null; then
        sed -i 's/^from torch.library import Library, infer_schema$/try:\n    from torch.library import Library, infer_schema\n    _infer_schema_available = True\nexcept ImportError:\n    from torch.library import Library\n    _infer_schema_available = False\n    def infer_schema(func, *args, **kwargs): return ""/' "$TORCH_UTILS"
        echo "    torch_utils.py: _infer_schema_available flag added ✅"
    else
        echo "    torch_utils.py: infer_schema import pattern not found (already patched?)"
    fi

    # Add early-return guard to direct_register_custom_op if not already present.
    # NOTE: the guard check must be scoped to the function body specifically
    # (grep -A3 right after the def line) -- a whole-file `grep -q
    # _infer_schema_available` always matches because Patch 2 above already
    # added that flag's *declaration* earlier in the file, which silently
    # skipped ever adding this guard and left direct_register_custom_op
    # calling my_lib.define(op_name + "") on torch 2.2.x, crashing with
    # "RuntimeError: Tried to parse a function schema but only the operator
    # name was given" (found via a live Docker build failure).
    if grep -q "^def direct_register_custom_op" "$TORCH_UTILS" 2>/dev/null && \
       ! grep -A3 "^def direct_register_custom_op" "$TORCH_UTILS" | grep -q "_infer_schema_available"; then
        sed -i '/^def direct_register_custom_op/,/^    """/{/^    """/{i\    if not _infer_schema_available:\n        return
}}' "$TORCH_UTILS"
        echo "    direct_register_custom_op: early-return guard added ✅"
    else
        echo "    direct_register_custom_op: guard already present ✅"
    fi

    # Patch 3: csrc/cpu/utils.cpp — define VLLM_NUMA_DISABLED at source level.
    # cmake's add_compile_definitions(-DVLLM_NUMA_DISABLED) has a leading-dash bug;
    # the define never reaches the preprocessor. Adding it directly to the source
    # is more reliable and avoids the cmake bug entirely.
    UTILS_CPP="$VLLM_SRC/csrc/cpu/utils.cpp"
    if ! grep -q "^#define VLLM_NUMA_DISABLED" "$UTILS_CPP" 2>/dev/null; then
        sed -i '1s/^/#define VLLM_NUMA_DISABLED  \/\/ numactl-devel not installed on build node\n/' "$UTILS_CPP"
        echo "    csrc/cpu/utils.cpp: VLLM_NUMA_DISABLED defined ✅"
    fi

    # Patch 4: cmake/cpu_extension.cmake — auto-detect NUMA properly.
    # The original cmake hardcodes ENABLE_NUMA=TRUE and only disables for Apple Silicon.
    # s339 has libnuma.so.1 (runtime) but NOT libnuma.so (needs numactl-devel).
    # cmake's find_library(numa) correctly returns NOT FOUND for missing .so → NUMA disabled.
    #
    # NOTE: pass the path as an argument to the SAME invocation that reads
    # its script from the heredoc — a bare `python3 << 'PYEOF'` heredoc has
    # no argv, so a separate `sys.argv[1]`-reading invocation without the
    # heredoc attached would hang on stdin instead of receiving it.
    python3 - "$VLLM_SRC/cmake/cpu_extension.cmake" << 'PYEOF' || echo "    NUMA cmake patch: failed (may already be patched)"
import sys
content = open(sys.argv[1]).read()
old = '\nset (ENABLE_NUMA TRUE)\n'
new = '''
find_library(LIBNUMA_LIB NAMES numa PATHS /usr/lib64 /usr/lib)
if(LIBNUMA_LIB)
    set(ENABLE_NUMA TRUE)
else()
    set(ENABLE_NUMA FALSE)
    message(STATUS "NUMA: libnuma.so not found (numactl-devel not installed), disabling")
endif()
'''
if old not in content:
    print('    ENABLE_NUMA patch: already applied or pattern changed, skipping')
    sys.exit(0)
open(sys.argv[1], 'w').write(content.replace(old, new))
print('    cmake ENABLE_NUMA auto-detect: patched ✅')
PYEOF

    # Patch 5: cmake/cpu_extension.cmake — remove mla_decode.cpp (requires AVX-512,
    # s339 has AMD EPYC 7742 = AVX2 only). The RDU decode side only needs vllm._C
    # for model architecture inspection, not for CPU MLA inference.
    sed -i '/"csrc\/cpu\/mla_decode.cpp"/d' "$VLLM_SRC/cmake/cpu_extension.cmake" 2>/dev/null || true
    echo "    cmake/cpu_extension.cmake: mla_decode.cpp removed ✅"

    # Patch 6: csrc/cpu/utils.hpp — at::cpu::L2_cache_size() doesn't exist in
    # torch 2.2.0+sn (confirmed absent from its C++ headers entirely, not a
    # version regression — this vllm release's csrc has just never actually
    # been compiled against this specific torch before). Used only to pick a
    # cache-blocking size for a CPU attention micro-optimization we don't
    # need — real compute happens on RDU/GPU hardware, not vllm's own CPU
    # kernels. Hardcode a reasonable constant (1MB) instead of calling it.
    UTILS_HPP="$VLLM_SRC/csrc/cpu/utils.hpp"
    if grep -q "const uint32_t l2_cache_size = at::cpu::L2_cache_size();" "$UTILS_HPP" 2>/dev/null; then
        python3 - "$UTILS_HPP" <<'PYEOF'
import sys
path = sys.argv[1]
old = '''inline int64_t get_available_l2_size() {
  static int64_t size = []() {
    const uint32_t l2_cache_size = at::cpu::L2_cache_size();
    return l2_cache_size >> 1;  // use 50% of L2 cache
  }();
  return size;
}'''
new = '''inline int64_t get_available_l2_size() {
  // at::cpu::L2_cache_size() is not present in torch 2.2.0+sn's C++ API
  // (confirmed absent from its headers). Not performance-critical here —
  // this only sizes a CPU-attention cache-blocking optimization, and real
  // compute happens on RDU/GPU hardware. Hardcode a reasonable value
  // (1MB, half of a common 2MB per-core L2 size) instead.
  static int64_t size = 1048576 >> 1;
  return size;
}'''
text = open(path).read()
assert old in text, "utils.hpp: expected get_available_l2_size() body not found — upstream vllm may have changed"
open(path, "w").write(text.replace(old, new, 1))
print("    csrc/cpu/utils.hpp: get_available_l2_size() hardcoded (no at::cpu::L2_cache_size) ✅")
PYEOF
    else
        echo "    csrc/cpu/utils.hpp: L2_cache_size call not found (already patched or upstream changed)"
    fi

    # ── UCX + NIXL source ──────────────────────────────────────────────────────
    if [ ! -d "$SRC_DIR/ucx/.git" ]; then
        echo "  Cloning andychensn/ucx@$UCX_COMMIT..."
        git clone --depth=200 --branch "$UCX_BRANCH" \
            https://github.com/andychensn/ucx.git "$SRC_DIR/ucx"
        git -C "$SRC_DIR/ucx" checkout "$UCX_COMMIT"
        echo "  UCX cloned ✅"
    else
        echo "  UCX already present"
    fi

    if [ ! -d "$SRC_DIR/nixl/.git" ]; then
        echo "  Cloning andychensn/nixl@$NIXL_COMMIT..."
        git clone --depth=50 --branch "$NIXL_BRANCH" \
            https://github.com/andychensn/nixl.git "$SRC_DIR/nixl"
        git -C "$SRC_DIR/nixl" checkout "$NIXL_COMMIT"
        echo "  NIXL cloned ✅"
    else
        echo "  NIXL already present"
    fi

    # gguf + regex — needed by vllm.transformers_utils.gguf_utils (loaded at import time).
    # gguf is a pure-Python GGUF model reader; regex is its C-ext string library.
    # Neither is in the SambaNova system Python. Not used at runtime for HF models,
    # but the module-level import in vllm 0.16.0 means both must be present.
    if ! find "$WHEELHOUSE" -name "gguf-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading gguf..."
        python3.12 -m pip download "gguf>=0.17.0" --only-binary=:all: --no-deps --dest "$WHEELHOUSE" 2>&1 | tail -2
        echo "  gguf wheel ✅"
    else
        echo "  gguf wheel already present"
    fi
    if ! find "$WHEELHOUSE" -name "regex-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading regex..."
        python3.12 -m pip download "regex" --only-binary=:all: --no-deps \
            --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEELHOUSE" 2>&1 | tail -2
        echo "  regex wheel ✅"
    else
        echo "  regex wheel already present"
    fi

    # av (PyAV) — needed by fast-coe's rdu_manifest.vlm_pipeline (imported
    # unconditionally at module load, even for text-only models). C-extension
    # package (bundles ffmpeg libs), so must be a prebuilt wheel, not vendored
    # source. Not in the SambaNova system Python.
    if ! find "$WHEELHOUSE" -name "av-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading av..."
        python3.12 -m pip download "av==12.3.0" --only-binary=:all: --no-deps \
            --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEELHOUSE" 2>&1 | tail -2
        echo "  av wheel ✅"
    else
        echo "  av wheel already present"
    fi

    # numpy pinned wheel — system s339 may have newer numpy (2.x) compiled against
    # incompatible ABI; we install 1.26.4 into the venv to override it.
    if ! find "$WHEELHOUSE" -name "numpy-${RDU_NUMPY_VERSION}-cp311-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading numpy==$RDU_NUMPY_VERSION (cp311 wheel for s339)..."
        python3.12 -m pip download "numpy==$RDU_NUMPY_VERSION" --only-binary=:all: --no-deps \
            --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEELHOUSE" 2>&1 | tail -2
        echo "  numpy wheel ✅"
    else
        echo "  numpy wheel already present"
    fi

    # ai-dynamo-runtime (Rust compiled bindings) + ai-dynamo (Python app code)
    # Both downloaded with --no-deps; installed separately on s339 to avoid
    # pulling vllm/torch as pip dependencies (already installed from wheelhouse).
    if ! find "$WHEELHOUSE" -name "ai_dynamo_runtime-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading ai-dynamo-runtime==$DYNAMO_VERSION..."
        python3.12 -m pip download "ai-dynamo-runtime==$DYNAMO_VERSION" --only-binary=:all: --no-deps \
            --dest "$WHEELHOUSE" 2>&1 | tail -2
        echo "  ai-dynamo-runtime wheel ✅"
    else
        echo "  ai-dynamo-runtime wheel already present"
    fi
    if ! find "$WHEELHOUSE" -name "ai_dynamo-*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading ai-dynamo==$DYNAMO_VERSION..."
        python3.12 -m pip download "ai-dynamo==$DYNAMO_VERSION" --only-binary=:all: --no-deps \
            --dest "$WHEELHOUSE" 2>&1 | tail -2
        echo "  ai-dynamo wheel ✅"
    else
        echo "  ai-dynamo wheel already present"
    fi

    # Every other unpinned wheel that docker/rdu/install-deps.sh's
    # install_whl calls expect to already be in wheelhouse/ — vllm's own
    # import-time deps plus the "extra transitive deps" discovered by
    # exercising the real entrypoint import chain (see that script for why
    # each one is needed). None of these have a pinned version;
    # --only-binary=:all: keeps them prebuilt wheels rather than sdists
    # needing a compiler.
    #
    # If you add a new install_whl(...) call to docker/rdu/install-deps.sh,
    # add the matching package name here too — otherwise it only "works" for
    # as long as your own wheelhouse/ happens to already have it cached, and
    # silently breaks on the next genuinely clean rebuild.
    for pkg in \
        pybase64  blake3  depyf  lark  einops  cloudpickle  loguru \
        diskcache  msgspec  ninja  cachetools  anyio  httpcore  httpx \
        openai  compressed_tensors  openai_harmony  mcp  mistral_common \
        docstring_parser  durationpy  email_validator  h11  fastar \
        llguidance  lm_format_enforcer  sniffio  astor  dnspython \
        pydantic_settings  pyjwt  python_multipart  sse_starlette \
        starlette  typing_inspection  uvicorn  pydantic_extra_types \
        tiktoken  ijson  partial_json_parser  watchfiles  anthropic \
        fastapi  outlines_core  prometheus_fastapi_instrumentator \
        python_json_logger  xgrammar  kubernetes \
        model_hosting_container_standards  exceptiongroup  httpx_sse \
        tqdm  pycountry  annotated_doc  interegular \
        jmespath  python_dotenv  requests_oauthlib  websocket_client \
        redis  oauthlib  asgiref  cffi  cryptography  google_auth \
        googleapis_common_protos  grpcio  grpcio_reflection  httptools \
        importlib_metadata  json_logic  opentelemetry_api \
        opentelemetry_exporter_otlp  opentelemetry_sdk \
        opentelemetry_semantic_conventions  protobuf  pyasn1 \
        pyasn1_modules  pyprctl  rich_toolkit  rignore  shellingham \
        typer  websockets  zipp  pydantic  typing_extensions \
        ; do
        if ! find "$WHEELHOUSE" -name "${pkg}-*.whl" 2>/dev/null | grep -q .; then
            echo "  Downloading $pkg..."
            python3.12 -m pip download "$pkg" --only-binary=:all: --no-deps \
                --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEELHOUSE" 2>&1 | tail -2
        fi
    done
    echo "  extra transitive-dep wheels ✅"

    # pydantic_core: NOT in the unpinned loop above on purpose. pydantic
    # checks at import time that its installed pydantic_core is EXACTLY the
    # version it was built against (SystemError if not) — independently
    # fetching "whatever's latest" for both packages will eventually drift
    # out of sync the moment either gets a new PyPI release (e.g. pydantic
    # 2.13.4 requires pydantic_core==2.46.4 exactly, while an independently
    # fetched "latest" pydantic_core could already be 2.47.0). Derive the
    # exact required version from the pydantic wheel's own metadata instead
    # of guessing a version number that will go stale.
    if ! find "$WHEELHOUSE" -name "pydantic_core-*.whl" 2>/dev/null | grep -q .; then
        PYDANTIC_WHL=$(find "$WHEELHOUSE" -name "pydantic-*.whl" 2>/dev/null | head -1)
        if [ -n "$PYDANTIC_WHL" ]; then
            PYDANTIC_CORE_PIN=$(python3.12 -c "
import zipfile, re, sys
with zipfile.ZipFile('$PYDANTIC_WHL') as z:
    meta = [n for n in z.namelist() if n.endswith('METADATA')][0]
    text = z.read(meta).decode()
m = re.search(r'^Requires-Dist: pydantic-core==([0-9.]+)', text, re.MULTILINE)
print(m.group(1) if m else '')
")
            if [ -n "$PYDANTIC_CORE_PIN" ]; then
                echo "  Downloading pydantic_core==$PYDANTIC_CORE_PIN (exact pin required by fetched pydantic)..."
                python3.12 -m pip download "pydantic_core==$PYDANTIC_CORE_PIN" --only-binary=:all: --no-deps \
                    --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEELHOUSE" 2>&1 | tail -2
            else
                echo "  WARNING: could not determine pydantic_core pin from $PYDANTIC_WHL metadata — falling back to latest"
                python3.12 -m pip download "pydantic_core" --only-binary=:all: --no-deps \
                    --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEELHOUSE" 2>&1 | tail -2
            fi
        fi
    fi

    # jiter: needed cp311-specific (docker/rdu/install-deps.sh
    # explicitly avoids the cp312 variant pip would otherwise resolve to on
    # some platform args).
    if ! find "$WHEELHOUSE" -name "jiter-*cp311*.whl" 2>/dev/null | grep -q .; then
        echo "  Downloading jiter (cp311)..."
        python3.12 -m pip download "jiter" --only-binary=:all: --no-deps \
            --python-version 311 --platform manylinux_2_17_x86_64 --dest "$WHEELHOUSE" 2>&1 | tail -2
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

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: build everything on the RDU node (uses NFS clone, no internet needed)
# ═══════════════════════════════════════════════════════════════════════════

build_ucx_nixl() {
    echo "=== Building UCX + NIXL on $(hostname) $(date) ==="
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
    NIXL_WHL=$(find "$WHEELHOUSE" -name "nixl_cu12*cp311*.whl" 2>/dev/null | head -1 || true)
    if [ -n "$NIXL_WHL" ] && [ "${SKIP_NIXL:-0}" = "1" ]; then
        echo "=== NIXL wheel exists (SKIP_NIXL=1): $NIXL_WHL ==="
    else
        echo "=== Building NIXL wheel from $SRC_DIR/nixl ==="
        NIXL_SRC="$BUILD_TMP/nixl"
        cp -r "$SRC_DIR/nixl" "$NIXL_SRC"
        mkdir -p "$WHEELHOUSE"

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
        "$PY" -m pip wheel . --no-deps --no-build-isolation -w "$WHEELHOUSE" \
            --config-settings=setup-args="-Ducx_path=$UCX_INSTALL" \
            --config-settings=setup-args="-Denable_plugins=UCX"
        cd "$REPO_ROOT"

        NIXL_WHL=$(find "$WHEELHOUSE" -name "nixl*.whl" -newer "$BUILD_TMP" | head -1 || true)
        echo "NIXL wheel: $NIXL_WHL"
    fi

    echo ""
    echo "=== UCX + NIXL done $(date) ==="
    echo "    UCX:  $UCX_INSTALL"
    echo "    NIXL: $NIXL_WHL"
}

build_vllm_cpu_wheel() {
    echo "=== Building vllm $VLLM_VERSION+cpu on $(hostname) $(date) ==="
    [ -d "$VLLM_SRC" ] || { echo "ERROR: $VLLM_SRC not found — run --fetch-only first"; exit 1; }

    # Always start cmake configure fresh. vllm's setup.py leaves a persistent
    # in-source-tree build/ dir (with a CMakeCache.txt) that survives across
    # separate --build-only invocations, since $VLLM_SRC lives under the
    # NFS-persisted rdu-build-src/, not a scratch /tmp dir. If an earlier
    # attempt got past cmake configure (which resolves and caches an absolute
    # path to ninja from build_ucx_nixl()'s throwaway $BUILD_TMP) and then
    # failed later during actual compilation, a subsequent run's cmake
    # reuses that now-deleted ninja path from the stale cache and fails with
    # a confusing "No such file or directory" — unrelated to whatever the
    # original failure was. Removing the whole build/ dir first guarantees
    # every invocation reconfigures against the current job's own paths.
    rm -rf "$VLLM_SRC/build"

    # Fix vllm pyproject.toml license field:
    # - "Apache 2.0" is not valid SPDX → setuptools 77+ rejects it
    # - pip 22.3.1 (on s339) requires object format {text=...} not a bare string
    # Result: license = {text = "Apache-2.0"} satisfies both old pip and new setuptools.
    sed -i 's/^license = {text = "Apache 2\.0"}$/license = {text = "Apache-2.0"}/' "$VLLM_SRC/pyproject.toml" 2>/dev/null || true
    sed -i 's/^license = "Apache 2\.0"$/license = {text = "Apache-2.0"}/' "$VLLM_SRC/pyproject.toml" 2>/dev/null || true
    sed -i 's/^license = "Apache-2\.0"$/license = {text = "Apache-2.0"}/' "$VLLM_SRC/pyproject.toml" 2>/dev/null || true
    # license-files is a PEP 639 field; older setuptools rejects it as unknown property
    sed -i '/^license-files = /d' "$VLLM_SRC/pyproject.toml" 2>/dev/null || true

    # HOME may be unset in snrdu jobs; pip needs it for temp dirs
    export HOME="${HOME:-/tmp}"

    # Build with VLLM_TARGET_DEVICE=cpu to compile vllm._C (needed for model inspection).
    # Key cmake flags:
    # - CMAKE_PREFIX_PATH=/opt/sambanova: finds SambaNova's bundled protobuf
    #   (Caffe2Config.cmake requires it; libprotobuf.so is in /opt/sambanova/lib/)
    # - CMAKE_CXX_FLAGS=-DVLLM_NUMA_DISABLED: skips numa.h include (numactl-devel not installed)
    # mla_decode.cpp was removed in --fetch-only phase (requires AVX-512; s339 = AMD EPYC 7742 = AVX2 only)
    export PKG_CONFIG_PATH="/opt/sambanova/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    # ENABLE_NUMA=OFF: tells cmake to skip -lnuma link and add -DVLLM_NUMA_DISABLED compile flag.
    # numactl-devel not installed on s339 (libnuma.so.1 exists but no unversioned .so symlink).
    export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/sambanova -DENABLE_NUMA=OFF"

    echo "  Building CPU wheel with compiled vllm._C (~2-3 min)..."
    cd "$VLLM_SRC"
    VLLM_TARGET_DEVICE=cpu \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}+cpu" \
        "$PY" -m pip wheel . \
        --no-deps \
        --no-build-isolation \
        --no-cache-dir \
        --wheel-dir "$WHEELHOUSE"
    cd "$REPO_ROOT"

    WHL=$(find "$WHEELHOUSE" -name "vllm-${VLLM_VERSION}+cpu*.whl" | head -1 || true)
    if [ -n "$WHL" ]; then
        echo "=== vllm+cpu wheel done: $WHL ==="
    else
        echo "ERROR: wheel not found in $WHEELHOUSE — see build output above"
        ls "$WHEELHOUSE"/vllm-*.whl 2>/dev/null || echo "  (no vllm wheels)"
        exit 1
    fi
}

build_on_rdu_node() {
    [ -x "$PY" ] || { echo "ERROR: $PY not found — must run on RDU node via snrdu"; exit 1; }
    [ -d "$SRC_DIR/ucx/.git" ] || { echo "ERROR: sources not fetched — run with --fetch-only from login node first"; exit 1; }
    [ -d "$FAST_COE_SRC/.git" ] || { echo "ERROR: fast-coe not fetched — run with --fetch-only from login node first"; exit 1; }

    build_ucx_nixl
    echo ""
    build_vllm_cpu_wheel
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
        echo "      --pef ${PEF:-<PEF>} --timeout ${RDU_TIMEOUT:-01:50:00} \\"
        echo "      -o logs/build_rdu_env.log \\"
        echo "      -- bash $REPO_ROOT/build/rdu_env.sh --build-only"
        ;;
    *)
        echo "Usage: $0 [--fetch-only | --build-only | (no arg = fetch + print next-step instructions)]"
        exit 1 ;;
esac
