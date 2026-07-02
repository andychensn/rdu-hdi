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
#   3. snrdu run ... -- bash scripts/build_rdu_venv.sh     # this script
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
VLLM_RDU_SRC=$REPO_ROOT/vllm-rdu

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

# Apply REGISTER_CONSUMER_MSG patch to nixl_connector.py post-install.
# This patch adds chunk-overlap KV transfer support for P/D disaggregation.
NIXL_PY=$(find "$VENV" -name "nixl_connector.py" -path "*/kv_connector*" 2>/dev/null | head -1)
NIXL_PATCH="$REPO_ROOT/patches/vllm_nixl_connector.patch"
if grep -q "REGISTER_CONSUMER_MSG" "$NIXL_PY" 2>/dev/null; then
    echo "  REGISTER_CONSUMER_MSG already present ✅"
elif [ -f "$NIXL_PATCH" ] && [ -n "$NIXL_PY" ]; then
    # Apply patch relative to the venv site-packages
    SITE_PKG=$(dirname "$(dirname "$(dirname "$NIXL_PY")")")
    cd "$SITE_PKG" && patch -p1 < "$NIXL_PATCH" && echo "  vllm_nixl_connector.patch applied ✅" || \
        echo "WARNING: nixl_connector patch failed — VLLM_PD_CHUNK_OVERLAP=1 may not work"
    cd "$REPO_ROOT"
fi

# publisher.py: clamp kv_cache_usage-derived block count to >= 0.
# scheduler_stats.kv_cache_usage can go transiently negative after a
# KV-load-failure reschedule; dynamo's Rust publish() takes an unsigned int
# and raises OverflowError on a negative value, which kills the whole
# engine (EngineDeadError) on the very next metrics tick.
PUBLISHER_PY=$(find "$VENV" -name "publisher.py" -path "*/dynamo/vllm/*" 2>/dev/null | head -1)
if [ -n "$PUBLISHER_PY" ] && grep -q "^        active_decode_blocks = int(self.num_gpu_block \* scheduler_stats.kv_cache_usage)$" "$PUBLISHER_PY" 2>/dev/null; then
    sed -i 's/^        active_decode_blocks = int(self.num_gpu_block \* scheduler_stats.kv_cache_usage)$/        active_decode_blocks = max(0, int(self.num_gpu_block * scheduler_stats.kv_cache_usage))/' "$PUBLISHER_PY"
    echo "  publisher.py: negative kv_cache_usage clamp applied ✅"
fi

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
    whl=$(find "$WHEELHOUSE" -name "$pattern" 2>/dev/null | head -1 || true)
    if [ -n "$whl" ]; then
        pip install -q --no-deps "$whl"
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

# ── Dynamo Python deps (attempt from system site-packages or wheelhouse) ──────
echo "=== Dynamo messaging deps ==="
for pkg in aiohttp msgpack pyzmq sortedcontainers uvloop cbor2; do
    pip install -q "$pkg" 2>/dev/null && echo "  $pkg: OK" || echo "  $pkg: unavailable (may be in system)"
done

# ── NIXL ──────────────────────────────────────────────────────────────────────
echo "=== nixl ==="
pip install -q "$NIXL_WHL"

# ── vllm-rdu plugin ───────────────────────────────────────────────────────────
echo "=== vllm-rdu (local editable install) ==="
[ -d "$VLLM_RDU_SRC" ] || { echo "ERROR: $VLLM_RDU_SRC not found — run: gh repo clone andychensn/vllm-rdu $VLLM_RDU_SRC"; exit 1; }
pip install -q -e "$VLLM_RDU_SRC"

# ── Validate ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Validating ==="
RDU_UCX_LIB="$REPO_ROOT/rdu-ucx-install/lib"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
LD_LIBRARY_PATH="$RDU_UCX_LIB:${LD_LIBRARY_PATH:-}" python -c "import nixl; print('nixl: OK')" || \
    echo "WARNING: nixl needs UCX libs at runtime (set LD_LIBRARY_PATH=$RDU_UCX_LIB)"
python -c "import rdu_hardware; print('vllm-rdu: OK')" 2>/dev/null || echo "WARNING: rdu_hardware import failed (expected on non-RDU node)"

echo ""
echo "=== RDU venv build COMPLETE $(date) ==="
