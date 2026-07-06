#!/usr/bin/env bash
# Install the RDU decode Python stack into the base image's own
# /opt/sambanova python3.11 (no separate venv — the container itself is
# the isolation boundary: torch/mpich come from the base image, everything
# else installs alongside them).
#   - UCX/NIXL/vllm+cpu are not built here — this Dockerfile COPYs the
#     already-built artifacts (rdu-ucx-install/, wheelhouse/*.whl) instead,
#     since they're pure binary artifacts already validated on this node
#     family (see build/rdu_env.sh).
#   - coe_api/rdu_engine are self-built and installed as wheels directly
#     (see the "coe_api/rdu_engine" section below and build/bar2.sh);
#     the BAR2 runtime connector libs are COPYed into the image separately
#     by docker/rdu/Dockerfile (rdu-runtime-install/{lib,preload}).
#
# Run inside the docker/rdu/Dockerfile build (working dir: repo root, with
# wheelhouse/, rdu-ucx-install/, patches/rdu/, fast-coe/ all present).
set -euo pipefail

export PYTHONNOUSERSITE=1
PY=/opt/sambanova/bin/python3.11
PIP="$PY -m pip"
WHEELHOUSE="/build/wheelhouse"
FAST_COE_SRC="/build/fast-coe"
SITE_PACKAGES=$("$PY" -c "import site; print(site.getsitepackages()[0])")

install_whl() {
    local pattern="$1"
    local whl
    whl=$(find "$WHEELHOUSE" -name "$pattern" 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$whl" ]; then
        $PIP install -q --no-deps --force-reinstall --no-cache-dir "$whl"
        echo "  installed: $(basename "$whl")"
    else
        echo "  WARNING: no wheel matching $pattern in wheelhouse"
    fi
}

$PIP install -q --upgrade pip

echo "=== vllm+cpu (pre-built) ==="
VLLM_CPU_WHL=$(find "$WHEELHOUSE" -name "vllm-*+cpu-cp311*.whl" 2>/dev/null | head -1)
[ -n "$VLLM_CPU_WHL" ] || { echo "ERROR: no vllm +cpu wheel in wheelhouse"; exit 1; }
$PIP install -q --no-deps "$VLLM_CPU_WHL"

echo "=== numpy (must precede everything else — binary-incompatible 2.x guard) ==="
install_whl "numpy-*.whl"

echo "=== transformers ==="
$PIP install -q "transformers==${RDU_TRANSFORMERS_VERSION}"

echo "=== vllm import-time deps ==="
install_whl "gguf-*.whl"
install_whl "regex-*.whl"
install_whl "pybase64-*.whl"
install_whl "blake3-*.whl"
install_whl "depyf-*.whl"
install_whl "lark-*.whl"
install_whl "einops-*.whl"
install_whl "cloudpickle-*.whl"
install_whl "loguru-*.whl"
install_whl "diskcache-*.whl"
install_whl "msgspec-*.whl"
install_whl "ninja-*.whl"
install_whl "cachetools-*.whl"
install_whl "anyio-*.whl"
install_whl "httpcore-*.whl"
install_whl "httpx-*.whl"
install_whl "openai-*.whl"
install_whl "compressed_tensors-*.whl"
install_whl "openai_harmony-*.whl"
install_whl "mcp-*.whl"
install_whl "mistral_common-*.whl"
install_whl "docstring_parser-*.whl"
install_whl "durationpy-*.whl"
install_whl "email_validator-*.whl"
install_whl "h11-*.whl"
install_whl "fastar-*.whl"
install_whl "llguidance-*.whl"
install_whl "lm_format_enforcer-*.whl"

echo "=== ai-dynamo-runtime + ai-dynamo ==="
DYNAMO_RUNTIME_WHL=$(find "$WHEELHOUSE" -name "ai_dynamo_runtime-*.whl" 2>/dev/null | head -1)
DYNAMO_WHL=$(find "$WHEELHOUSE" -name "ai_dynamo-*.whl" 2>/dev/null | head -1)
$PIP install -q --no-deps "$DYNAMO_RUNTIME_WHL"
$PIP install -q --no-deps "$DYNAMO_WHL"

# publisher.py: clamp kv_cache_usage-derived block count to >= 0 (a
# transiently-negative value after a KV-load-failure reschedule otherwise
# raises OverflowError in dynamo's Rust publish(), killing the engine).
PUBLISHER_PY=$(find "$SITE_PACKAGES" -name "publisher.py" -path "*/dynamo/vllm/*" 2>/dev/null | head -1)
if [ -n "$PUBLISHER_PY" ] && grep -q "^        active_decode_blocks = int(self.num_gpu_block \* scheduler_stats.kv_cache_usage)$" "$PUBLISHER_PY" 2>/dev/null; then
    sed -i 's/^        active_decode_blocks = int(self.num_gpu_block \* scheduler_stats.kv_cache_usage)$/        active_decode_blocks = max(0, int(self.num_gpu_block * scheduler_stats.kv_cache_usage))/' "$PUBLISHER_PY"
    echo "  publisher.py: negative kv_cache_usage clamp applied"
fi

# multimodal_utils/protocol.py: vllm 0.16.0+cpu only exposes
# MultiModalUUIDDict from vllm.multimodal.inputs, not re-exported at
# vllm.inputs (added in a later vllm version). ai-dynamo 1.2.1's bare
# import crashes the whole `python -m dynamo.vllm` entrypoint without this.
PROTOCOL_PY=$(find "$SITE_PACKAGES" -name "protocol.py" -path "*/dynamo/vllm/multimodal_utils/*" 2>/dev/null | head -1)
PROTOCOL_PATCH="/build/patches/rdu/dynamo_multimodal_protocol.patch"
if [ -n "$PROTOCOL_PY" ] && [ -f "$PROTOCOL_PATCH" ]; then
    (cd "$SITE_PACKAGES" && patch -p1 < "$PROTOCOL_PATCH") && \
        echo "  dynamo_multimodal_protocol.patch applied" || \
        { echo "ERROR: protocol.py patch failed"; exit 1; }
fi

echo "=== Dynamo messaging deps ==="
for pkg in aiohttp msgpack pyzmq sortedcontainers uvloop cbor2; do
    $PIP install -q "$pkg" && echo "  $pkg: OK"
done

echo "=== nixl (pre-built wheel) ==="
NIXL_WHL=$(find "$WHEELHOUSE" -name "nixl_cu12-*.whl" -o -name "nixl_cu13-*.whl" 2>/dev/null | head -1)
[ -n "$NIXL_WHL" ] || { echo "ERROR: no nixl wheel in wheelhouse"; exit 1; }
$PIP install -q "$NIXL_WHL"

# nixl's package is named after the detected CUDA version (nixl_cu12/
# nixl_cu13) even with no CUDA toolkit present (our CPU-only RDU build
# defaults to nixl_cu12) — vllm's nixl_connector.py hardcodes `from
# nixl._api import ...`, so a plain "nixl" name must exist. All submodule
# imports are relative, so a directory symlink is a safe, transparent alias.
NIXL_PKG_DIR=$(find "$SITE_PACKAGES" -maxdepth 1 \( -iname "nixl_cu12" -o -iname "nixl_cu13" \) -type d 2>/dev/null | head -1)
if [ -n "$NIXL_PKG_DIR" ] && [ ! -e "$SITE_PACKAGES/nixl" ]; then
    ln -s "$(basename "$NIXL_PKG_DIR")" "$SITE_PACKAGES/nixl"
    echo "  aliased $(basename "$NIXL_PKG_DIR") -> nixl"
fi

# nixl_connector.py (stock vLLM): a telemetry-retrieval failure was being
# treated as a transfer failure even when check_xfer_state already
# confirmed DONE — capture_telemetry defaults to False on the NIXL agent,
# so get_xfer_telemetry ALWAYS raises NIXL_ERR_NO_TELEMETRY, misclassifying
# every completed transfer as failed (~25s/request reschedule storm).
STOCK_NIXL_PY=$(find "$SITE_PACKAGES" -name "nixl_connector.py" -path "*/vllm/distributed/*" 2>/dev/null | head -1)
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
    echo "  nixl_connector.py: NIXL_ERR_NO_TELEMETRY false-failure patch applied"
fi

echo "=== vllm-rdu (fast-coe, editable install) ==="
[ -d "$FAST_COE_SRC/server/vllm-rdu" ] || { echo "ERROR: $FAST_COE_SRC/server/vllm-rdu not found"; exit 1; }
$PIP install -q -e "$FAST_COE_SRC/server/vllm-rdu"

echo "=== av (fast-coe's rdu_manifest.vlm_pipeline, imported unconditionally) ==="
install_whl "av-*.whl"

echo "=== coe_api/rdu_engine (self-built, from wheelhouse/) ==="
RDU_ENGINE_WHL=$(find "$WHEELHOUSE" -name "sambanova_rdu_engine_api-*.whl" 2>/dev/null | head -1)
COE_API_WHL=$(find "$WHEELHOUSE" -name "sambanova_coe_api-*.whl" 2>/dev/null | head -1)
[ -n "$RDU_ENGINE_WHL" ] || { echo "ERROR: no sambanova_rdu_engine_api wheel in wheelhouse (run build/bar2.sh first)"; exit 1; }
[ -n "$COE_API_WHL" ] || { echo "ERROR: no sambanova_coe_api wheel in wheelhouse (run build/bar2.sh first)"; exit 1; }
$PIP install -q --no-deps --force-reinstall --no-cache-dir "$RDU_ENGINE_WHL"
$PIP install -q --no-deps --force-reinstall --no-cache-dir "$COE_API_WHL"
echo "  installed: $(basename "$RDU_ENGINE_WHL"), $(basename "$COE_API_WHL")"

echo "=== Extra transitive deps (discovered by exercising the full entrypoint import chain) ==="
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

# jiter: wheelhouse has both cp311 and cp312 variants — install_whl's
# `sort -V | tail -1` would pick cp312 (name-sorts after cp311), which pip
# then rejects as unsupported on this cp311 install.
JITER_WHL=$(find "$WHEELHOUSE" -name "jiter-*cp311*.whl" 2>/dev/null | head -1)
[ -n "$JITER_WHL" ] && $PIP install -q --no-deps --force-reinstall --no-cache-dir "$JITER_WHL"

# pydantic/typing_extensions: upgrade over whatever version transformers or
# another --no-deps install pulled in, to satisfy fastapi/mcp/ai-dynamo's
# stricter version floors.
install_whl "pydantic-*.whl"
install_whl "typing_extensions-*.whl"

# py-cpuinfo (imported as `cpuinfo`): a gap between rhel810-dev's base
# /opt/sambanova install (doesn't have it) and a bare-metal RDU node's
# ambient site-packages (does) — vllm.usage.usage_lib imports it
# unconditionally. Not in wheelhouse (never needed fetching on bare metal
# where it's already ambient) — pip install directly from PyPI, same as
# transformers/the Dynamo messaging deps above.
echo "=== py-cpuinfo (gap vs rhel810-dev's base /opt/sambanova install) ==="
$PIP install -q py-cpuinfo

echo ""
echo "=== Validating ==="
RDU_UCX_LIB="/opt/rdu-ucx/lib"
$PY -c "import vllm; print(f'vllm: {vllm.__version__}')"
$PY -c "import numpy; print(f'numpy: {numpy.__version__}')"
LD_LIBRARY_PATH="$RDU_UCX_LIB:${LD_LIBRARY_PATH:-}" $PY -c "import nixl; print('nixl: OK')"
$PY -c "import av; print(f'av: {av.__version__}')"
$PY -c "from dynamo.vllm.main import main; print('dynamo.vllm.main: OK')"

# coe_api/rdu_engine are baked in (self-built, not NFS-mounted) --
# LD_LIBRARY_PATH already includes /opt/bar2-runtime/lib via this
# Dockerfile's own ENV instruction, so a plain import must succeed here,
# not just at container runtime. RDUTensor.dtype confirms the pinned
# software-repo commit (config/versions.env's SOFTWARE_REPO_*) has it.
$PY -c "
import rdu_engine
assert hasattr(rdu_engine, 'Checkpoint'), 'rdu_engine.Checkpoint missing'
assert hasattr(rdu_engine, 'PEF'), 'rdu_engine.PEF missing'
assert hasattr(rdu_engine.RDUTensor, 'dtype'), 'RDUTensor.dtype missing -- check config/versions.env SOFTWARE_REPO_COMMIT'
import coe_api
assert hasattr(coe_api.RDUTensor, 'dtype'), 'coe_api.RDUTensor.dtype missing'
print('rdu_engine/coe_api: OK (RDUTensor.dtype present)')
"
PYTHONPATH="$FAST_COE_SRC:$FAST_COE_SRC/server/inference-router/client-py:$FAST_COE_SRC/server/block_hash:${PYTHONPATH:-}" \
    $PY -c "from rdu_hardware.worker import *; print('rdu_hardware.worker: OK')"

echo ""
echo "=== RDU decode image dependency install COMPLETE $(date) ==="
