#!/usr/bin/env bash
# Build the RDU venv for rdu-hdi.
# Must run ON the RDU node via snrdu (needs /opt/sambanova/bin/python3.11).
# RDU node has no internet — all packages must come from local wheelhouse/.
#
# Prerequisites (run from login node):
#   1. bash scripts/build_vllm_cpu_wheel.sh --fetch-only   # clone + patch vllm source
#      source config/cluster.env config/model.env
#      snrdu run ... -- bash scripts/build_vllm_cpu_wheel.sh --build-only  # build +cpu wheel
#   2. bash scripts/build_rdu_ucx_nixl.sh --fetch-only     # fetch UCX/NIXL/deps
#      snrdu run ... -- bash scripts/build_rdu_ucx_nixl.sh --build-only    # build UCX+NIXL
#   3. bash scripts/fetch_fast_coe.sh                      # clone fast-coe, pinned by commit
#      (no build step — pure Python, installed editable below)
#   4. snrdu run ... -- bash scripts/build_rdu_venv.sh     # this script
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

PY=/opt/sambanova/bin/python3.11
VENV=$REPO_ROOT/.venv_rdu
WHEELHOUSE=$REPO_ROOT/wheelhouse

# Prefer the +cpu wheel (torch 2.2.x compat patches baked in) over the standard wheel.
# build_vllm_cpu_wheel.sh produces vllm-<ver>+cpu-cp311-cp311-linux_x86_64.whl.
# Prefer cp311 platform wheel (has compiled vllm._C) over py3-none-any (pure Python)
VLLM_CPU_WHL=$(find "$WHEELHOUSE" -name "vllm-*+cpu-cp311*.whl" 2>/dev/null | head -1 || \
               find "$WHEELHOUSE" -name "vllm-*+cpu*.whl" 2>/dev/null | head -1 || \
               find "$WHEELHOUSE" -name "vllm-*linux_x86_64.whl" 2>/dev/null | head -1 || true)
NIXL_WHL=$(find "$WHEELHOUSE" -name "nixl_cu12*cp311*.whl" 2>/dev/null | head -1 || true)
DYNAMO_RUNTIME_WHL=$(find "$WHEELHOUSE" -name "ai_dynamo_runtime-*.whl" 2>/dev/null | head -1 || true)
DYNAMO_WHL=$(find "$WHEELHOUSE" -name "ai_dynamo-*.whl" 2>/dev/null | head -1 || true)
FAST_COE_SRC=$REPO_ROOT/fast-coe

echo "=== rdu-hdi RDU venv build on $(hostname) $(date) ==="
echo "    Python: $($PY --version)"
echo "    VENV:   $VENV"

[ -x "$PY" ]           || { echo "ERROR: $PY not found — must run on RDU node"; exit 1; }
[ -n "$VLLM_CPU_WHL" ] || { echo "ERROR: no vllm wheel in $WHEELHOUSE — run build_vllm_cpu_wheel.sh first"; exit 1; }
[ -n "$NIXL_WHL" ]     || { echo "ERROR: no nixl wheel — run build_rdu_ucx_nixl.sh first"; exit 1; }
[ -n "$DYNAMO_RUNTIME_WHL" ] || { echo "ERROR: no ai-dynamo-runtime wheel — run build_rdu_ucx_nixl.sh --fetch-only"; exit 1; }
[ -n "$DYNAMO_WHL" ]         || { echo "ERROR: no ai-dynamo wheel — run build_rdu_ucx_nixl.sh --fetch-only"; exit 1; }

echo "    vllm:           $VLLM_CPU_WHL"
echo "    nixl:           $NIXL_WHL"
echo "    dynamo-runtime: $DYNAMO_RUNTIME_WHL"
echo "    dynamo:         $DYNAMO_WHL"

# ── Create venv ───────────────────────────────────────────────────────────────
echo ""
echo "=== Creating venv ==="
"$PY" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

# ── vllm (CPU-only, --no-deps to avoid torch version conflict) ────────────────
echo "=== vllm (CPU wheel) ==="
pip install -q --no-deps "$VLLM_CPU_WHL"

# If using standard upstream wheel (not +cpu), apply torch 2.2.x compat patches.
# The +cpu wheel already has these baked in (env_override.py + torch_utils.py).
# Always apply torch 2.2.x compat patches — the +cpu wheel built with VLLM_TARGET_DEVICE=empty
# may not have these baked in (pip caches/empty-target build can use unpatched system files).
echo "  Applying torch 2.2.x compat patches..."

# env_override.py: comprehensive torch 2.2.x compat shim
COMPAT_SHIM="$REPO_ROOT/patches/vllm_env_override_torch22x.py"
ENV_OVERRIDE=$(find "$VENV" -name "env_override.py" -path "*/vllm/*" 2>/dev/null | head -1)
[ -f "$COMPAT_SHIM" ] && [ -n "$ENV_OVERRIDE" ] && \
    cp "$COMPAT_SHIM" "$ENV_OVERRIDE" && echo "  env_override.py: compat shim applied ✅"

# torch_utils.py: _infer_schema_available guard for direct_register_custom_op
TORCH_UTILS=$(find "$VENV" -name "torch_utils.py" -path "*/vllm/*" 2>/dev/null | head -1)
if grep -q "^from torch.library import Library, infer_schema" "$TORCH_UTILS" 2>/dev/null; then
    sed -i 's/^from torch.library import Library, infer_schema$/try:\n    from torch.library import Library, infer_schema\n    _infer_schema_available = True\nexcept ImportError:\n    from torch.library import Library\n    _infer_schema_available = False\n    def infer_schema(func, *args, **kwargs): return ""/' "$TORCH_UTILS"
    echo "  torch_utils.py: _infer_schema_available flag added ✅"
fi
if grep -q "^def direct_register_custom_op" "$TORCH_UTILS" 2>/dev/null && \
   ! grep -A3 "^def direct_register_custom_op" "$TORCH_UTILS" | grep -q "_infer_schema_available"; then
    sed -i '/^def direct_register_custom_op/,/^    """/{/^    """/i\    if not _infer_schema_available:\n        return
}' "$TORCH_UTILS" 2>/dev/null || true
    echo "  direct_register_custom_op: early-return guard added ✅"
fi

# NOTE: a REGISTER_CONSUMER_MSG patch for chunk-overlap KV transfer
# (VLLM_PD_CHUNK_OVERLAP=1) used to be applied here. Removed 2026-07-03 —
# its patch file was already renamed to patches/vllm_nixl_connector.patch.retired
# (silently making this whole step a no-op with no error, discovered while
# fixing an unrelated patch-path bug below), and launch/rdu_decode.sh +
# launch/gpu_prefill.sh both hardcode VLLM_PD_CHUNK_OVERLAP=0 — the feature
# isn't used. See patches/vllm_nixl_connector.patch.retired if ever revived.

# nixl_connector.py (stock vLLM, not vllm-rdu): _pop_done_transfers treats a
# telemetry-retrieval failure as a transfer failure, even when
# check_xfer_state already confirmed "DONE". capture_telemetry defaults to
# False on the NIXL agent, so get_xfer_telemetry ALWAYS raises
# NIXL_ERR_NO_TELEMETRY unless the agent was explicitly built with
# capture_telemetry=True — meaning every single completed transfer was being
# misclassified as failed, triggering an ~24s KV-load-failure reschedule on
# EVERY request (measured: ~25s/request instead of the expected <1s).
STOCK_NIXL_PY=$(find "$VENV" -name "nixl_connector.py" -path "*/vllm/distributed/*" 2>/dev/null | head -1)
if [ -n "$STOCK_NIXL_PY" ] && grep -q "^                        res = self.nixl_wrapper.get_xfer_telemetry(handle)$" "$STOCK_NIXL_PY" 2>/dev/null; then
    python3 - "$STOCK_NIXL_PY" <<'PYEOF'
import sys
path = sys.argv[1]
old = '''                    if xfer_state == "DONE":
                        # Get telemetry from NIXL
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(res)
                        self.nixl_wrapper.release_xfer_handle(handle)'''
new = '''                    if xfer_state == "DONE":
                        # Telemetry is best-effort (capture_telemetry
                        # defaults to False on the NIXL agent) — a
                        # telemetry-retrieval failure must not invalidate a
                        # transfer check_xfer_state already confirmed DONE.
                        try:
                            res = self.nixl_wrapper.get_xfer_telemetry(handle)
                            self.xfer_stats.record_transfer(res)
                        except Exception:
                            pass
                        self.nixl_wrapper.release_xfer_handle(handle)'''
text = open(path).read()
assert old in text, "nixl_connector.py: expected DONE-branch snippet not found — upstream vllm may have changed"
open(path, "w").write(text.replace(old, new, 1))
PYEOF
    echo "  nixl_connector.py: NIXL_ERR_NO_TELEMETRY false-failure patch applied ✅"
fi

# ── numpy pin ─────────────────────────────────────────────────────────────────
# System s339 may have numpy 2.x in ~/.local (binary-incompatible with torch 2.2.0+sn).
# Install 1.26.4 from wheelhouse into the venv so it takes precedence.
NUMPY_WHL=$(find "$WHEELHOUSE" -name "numpy-${RDU_NUMPY_VERSION}-cp311-*.whl" 2>/dev/null | head -1 || true)
echo "=== numpy==$RDU_NUMPY_VERSION ==="
if [ -n "$NUMPY_WHL" ]; then
    pip install -q --force-reinstall --no-deps "$NUMPY_WHL"
else
    pip install -q --force-reinstall "numpy==$RDU_NUMPY_VERSION"
fi

# ── transformers ──────────────────────────────────────────────────────────────
echo "=== transformers==$RDU_TRANSFORMERS_VERSION ==="
pip install -q "transformers==$RDU_TRANSFORMERS_VERSION"

# ── vllm module-level deps (needed at import time, not in system Python) ──────
echo "=== vllm import-time deps ==="
install_whl() {
    local pattern="$1"
    local whl
    whl=$(find "$WHEELHOUSE" -name "$pattern" 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$whl" ]; then
        pip install -q --no-deps --force-reinstall --no-cache-dir "$whl"
        echo "  installed: $(basename "$whl")"
    else
        echo "  WARNING: no wheel matching $pattern in wheelhouse"
    fi
}

install_whl "gguf-*.whl"              # vllm.transformers_utils.gguf_utils
install_whl "regex-*.whl"             # gguf_utils dependency
install_whl "pybase64-*.whl"          # vllm.multimodal.media.audio
install_whl "blake3-*.whl"            # vllm.utils.hash
install_whl "depyf-*.whl"             # vllm.compilation
install_whl "lark-*.whl"              # vllm.model_executor
install_whl "einops-*.whl"            # model layers
install_whl "cloudpickle-*.whl"       # dynamo/vllm serialization
install_whl "loguru-*.whl"            # vllm logging
install_whl "diskcache-*.whl"         # vllm caching
install_whl "msgspec-*.whl"           # dynamo messaging
install_whl "ninja-*.whl"             # compilation
install_whl "cachetools-*.whl"        # dynamo
install_whl "anyio-*.whl"             # httpx/openai dep
install_whl "httpcore-*.whl"          # httpx dep
install_whl "httpx-*.whl"             # vllm client
install_whl "openai-*.whl"            # vllm benchmarking
install_whl "compressed_tensors-*.whl" # vllm quantization
install_whl "openai_harmony-*.whl"     # vllm.entrypoints.mcp.tool_server
install_whl "mcp-*.whl"               # vllm MCP support
install_whl "mistral_common-*.whl"    # vllm tokenizer support
install_whl "docstring_parser-*.whl"  # vllm dependency
install_whl "durationpy-*.whl"        # dynamo dependency
install_whl "email_validator-*.whl"   # fastapi/openai dependency
install_whl "h11-*.whl"              # httpx dependency
install_whl "fastar-*.whl"           # vllm dependency
install_whl "llguidance-*.whl"       # vllm structured output
install_whl "lm_format_enforcer-*.whl" # vllm structured output

# ── Dynamo runtime ────────────────────────────────────────────────────────────
echo "=== ai-dynamo-runtime + ai-dynamo ==="
pip install -q --no-deps "$DYNAMO_RUNTIME_WHL"
pip install -q --no-deps "$DYNAMO_WHL"

# publisher.py: clamp kv_cache_usage-derived block count to >= 0.
# scheduler_stats.kv_cache_usage can go transiently negative after a
# KV-load-failure reschedule; dynamo's Rust publish() takes an unsigned int
# and raises OverflowError on a negative value, which kills the whole
# engine (EngineDeadError) on the very next metrics tick.
# NOTE: must run after the ai-dynamo install above — dynamo/vllm/publisher.py
# doesn't exist yet before that point, so this silently no-ops if moved earlier.
PUBLISHER_PY=$(find "$VENV" -name "publisher.py" -path "*/dynamo/vllm/*" 2>/dev/null | head -1)
if [ -n "$PUBLISHER_PY" ] && grep -q "^        active_decode_blocks = int(self.num_gpu_block \* scheduler_stats.kv_cache_usage)$" "$PUBLISHER_PY" 2>/dev/null; then
    sed -i 's/^        active_decode_blocks = int(self.num_gpu_block \* scheduler_stats.kv_cache_usage)$/        active_decode_blocks = max(0, int(self.num_gpu_block * scheduler_stats.kv_cache_usage))/' "$PUBLISHER_PY"
    echo "  publisher.py: negative kv_cache_usage clamp applied ✅"
fi

# multimodal_utils/protocol.py: vllm 0.16.0+cpu only exposes MultiModalUUIDDict
# from vllm.multimodal.inputs, not re-exported at vllm.inputs (that re-export
# was added in a later vllm version than what this stack pins). ai-dynamo
# 1.2.1's bare import crashes the whole `python -m dynamo.vllm` entrypoint.
PROTOCOL_PY=$(find "$VENV" -name "protocol.py" -path "*/dynamo/vllm/multimodal_utils/*" 2>/dev/null | head -1)
PROTOCOL_PATCH="$REPO_ROOT/patches/dynamo_multimodal_protocol.patch"
if [ -n "$PROTOCOL_PY" ] && grep -q "^try:$" "$PROTOCOL_PY" 2>/dev/null && grep -q "MultiModalUUIDDict = dict" "$PROTOCOL_PY" 2>/dev/null; then
    echo "  protocol.py: MultiModalUUIDDict fallback already present ✅"
elif [ -f "$PROTOCOL_PATCH" ] && [ -n "$PROTOCOL_PY" ]; then
    SITE_PKG="${PROTOCOL_PY%/site-packages/*}/site-packages"
    cd "$SITE_PKG" && patch -p1 < "$PROTOCOL_PATCH" && echo "  dynamo_multimodal_protocol.patch applied ✅" || \
        echo "WARNING: protocol.py patch failed — dynamo.vllm import will crash on this vllm version"
    cd "$REPO_ROOT"
fi

# ── Dynamo Python deps (attempt from system site-packages or wheelhouse) ──────
echo "=== Dynamo messaging deps ==="
for pkg in aiohttp msgpack pyzmq sortedcontainers uvloop cbor2; do
    pip install -q "$pkg" 2>/dev/null && echo "  $pkg: OK" || echo "  $pkg: unavailable (may be in system)"
done

# ── NIXL ──────────────────────────────────────────────────────────────────────
echo "=== nixl ==="
pip install -q "$NIXL_WHL"

# nixl's meson.build names the installed package after the detected CUDA
# version (nixl_cu12/nixl_cu13), defaulting to nixl_cu12 when no CUDA
# toolkit is present at all (our CPU-only RDU build) — there is no plain
# "nixl" case in its build logic. vllm's nixl_connector.py hardcodes
# `from nixl._api import ...`, so without this the import fails at runtime.
# All submodule imports inside the package are relative (`from . import
# _bindings`), so a directory symlink is a safe, transparent alias.
NIXL_PKG_DIR=$(find "$VENV/lib/python3.11/site-packages" -maxdepth 1 \( -iname "nixl_cu12" -o -iname "nixl_cu13" \) -type d 2>/dev/null | head -1)
if [ -n "$NIXL_PKG_DIR" ] && [ ! -e "$VENV/lib/python3.11/site-packages/nixl" ]; then
    ln -s "$(basename "$NIXL_PKG_DIR")" "$VENV/lib/python3.11/site-packages/nixl"
    echo "  aliased $(basename "$NIXL_PKG_DIR") -> nixl (vllm imports 'nixl', not the CUDA-suffixed name)"
fi

# ── vllm-rdu plugin (fast-coe's, pinned — hdi's exact proven connector/engine) ─
echo "=== vllm-rdu (fast-coe @ $FAST_COE_COMMIT, editable install) ==="
[ -d "$FAST_COE_SRC/server/vllm-rdu" ] || { echo "ERROR: $FAST_COE_SRC/server/vllm-rdu not found — run: bash scripts/fetch_fast_coe.sh"; exit 1; }
pip install -q -e "$FAST_COE_SRC/server/vllm-rdu"

# ── PyAV (fast-coe's rdu_manifest.vlm_pipeline imports it unconditionally,
# even for text-only models — not needed by the old andychensn/vllm-rdu
# package, so not previously in this venv). Prebuilt manylinux wheel fetched
# from PyPI on the login node (which has internet) into wheelhouse/, same
# pattern as every other install_whl dependency above — the RDU node itself
# has no internet, not the login node running this fetch step.
install_whl "av-*.whl"                # rdu_manifest.vlm_pipeline (fast-coe)

# ── Extra transitive deps ──────────────────────────────────────────────────────
# Discovered one at a time by actually exercising `python -m dynamo.vllm`'s
# full entrypoint import chain (async engine, FastAPI, multimodal_utils,
# fast-coe's rdu_hardware worker) on a truly from-scratch venv — none of
# these surface from `import vllm`/`import dynamo.vllm` alone, and every one
# of them, if missing, crashes the whole engine at startup, not just a
# feature. Previously lived in a separate install_extra_deps.sh that was
# easy to forget to run; folded in here so one script produces a working venv.
echo "=== Extra transitive deps ==="
for pkg in \
    openai_harmony  docstring_parser  durationpy  email_validator \
    h11  mcp  mistral_common  fastar  llguidance  pybase64 \
    sniffio  astor  dnspython  pydantic_settings  pyjwt \
    python_multipart  sse_starlette  starlette  typing_inspection \
    uvicorn  pydantic_extra_types  tiktoken  ijson  partial_json_parser \
    watchfiles  anthropic  fastapi  outlines_core \
    prometheus_fastapi_instrumentator  python_json_logger \
    xgrammar  kubernetes  model_hosting_container_standards \
    exceptiongroup  httpx_sse  tqdm  lm_format_enforcer  pydantic_core \
    pycountry  annotated_doc  interegular  jmespath  python_dotenv \
    requests_oauthlib  websocket_client  redis  oauthlib \
    asgiref  cffi  cryptography  google_auth  googleapis_common_protos \
    grpcio  grpcio_reflection  httptools  importlib_metadata  json_logic \
    opentelemetry_api  opentelemetry_exporter_otlp  opentelemetry_sdk \
    opentelemetry_semantic_conventions  protobuf  pyasn1  pyasn1_modules \
    pyprctl  rich_toolkit  rignore  shellingham  typer  websockets  zipp \
    ; do
    install_whl "${pkg}-*.whl"
done

# jiter: wheelhouse has both cp311 and cp312 variants — this venv is cp311.
# install_whl's `sort -V | tail -1` would pick cp312 (name-sorts after
# cp311), which pip then rejects as "not a supported wheel on this platform".
JITER_WHL=$(find "$WHEELHOUSE" -name "jiter-*cp311*.whl" 2>/dev/null | head -1 || true)
if [ -n "$JITER_WHL" ]; then
    pip install -q --no-deps --force-reinstall --no-cache-dir "$JITER_WHL"
    echo "  installed: $(basename "$JITER_WHL")"
else
    echo "  WARNING: no cp311 jiter wheel in wheelhouse"
fi

# pydantic/typing_extensions: upgrade over whatever version transformers or
# another --no-deps install pulled in, to satisfy fastapi/mcp/ai-dynamo's
# stricter version floors.
install_whl "pydantic-*.whl"
install_whl "typing_extensions-*.whl"

echo ""
echo "=== pip check (informational — version-pin mismatches here are expected) ==="
pip check 2>&1 | grep -v "grpcio-reflection\|opencv\|protobuf\|msgpack\|prometheus-client\|aiohttp" | head -20 || true

# ── Validate ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Validating ==="
RDU_UCX_LIB="$REPO_ROOT/rdu-ucx-install/lib"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
LD_LIBRARY_PATH="$RDU_UCX_LIB:${LD_LIBRARY_PATH:-}" python -c "import nixl; print('nixl: OK')" || \
    echo "WARNING: nixl needs UCX libs at runtime (set LD_LIBRARY_PATH=$RDU_UCX_LIB)"
python -c "import rdu_hardware; print('vllm-rdu: OK')" 2>/dev/null || echo "WARNING: rdu_hardware import failed (expected on non-RDU node)"
python -c "import av; print(f'av: {av.__version__}')" 2>/dev/null || echo "WARNING: av import failed — rdu_manifest.vlm_pipeline will fail to load"

# `import vllm`/`import dynamo.vllm` alone do NOT exercise the full entrypoint
# import graph (async engine, FastAPI, multimodal_utils) that `python -m
# dynamo.vllm` (what launch/rdu_decode.sh actually runs) does — a missing
# transitive dep here crashes the real launch ~12 minutes into BAR2 init,
# not at build time, unless checked explicitly like this.
python -c "from dynamo.vllm.main import main; print('dynamo.vllm.main: OK')"

# Same idea for fast-coe's real RDU worker module chain (rdu_hardware.worker
# -> model_runner -> server.rdu_manifest.*) — this is where most of the
# "extra transitive deps" above were actually discovered missing.
PYTHONPATH="$FAST_COE_SRC:$FAST_COE_SRC/server/inference-router/client-py:$FAST_COE_SRC/server/block_hash:${PYTHONPATH:-}" \
    python -c "from rdu_hardware.worker import *; print('rdu_hardware.worker: OK')" 2>&1 | tail -5

echo ""
echo "=== RDU venv build COMPLETE $(date) ==="
