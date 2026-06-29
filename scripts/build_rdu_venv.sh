#!/usr/bin/env bash
# Build the RDU venv for rdu-hdi.
# Must run ON s339 via snrdu (needs /opt/sambanova/bin/python3.11).
# s339 has no internet — all packages must come from local paths.
#
# From login node:
#   source config/versions.env
#   snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
#       --reservation no_sf_catchup_demos --pef $PEF --timeout 00:30:00 \
#       -o logs/build_rdu_venv.log -- bash scripts/build_rdu_venv.sh
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

PY=/opt/sambanova/bin/python3.11
VENV=$REPO_ROOT/.venv_rdu
WHEELHOUSE=$REPO_ROOT/wheelhouse

# Paths to local artifacts (NFS-accessible from s339)
# Prefer the generic-platform wheel (linux_x86_64) that works on older glibc (RHEL8/s339)
VLLM_CPU_WHL=$(find "$WHEELHOUSE" -name "vllm-*linux_x86_64.whl" 2>/dev/null | head -1 || \
               find "$WHEELHOUSE" -name "vllm-*.whl" 2>/dev/null | head -1 || true)
# Accept wheels built from source (nixl_cu12-*cp311*.whl) or NFS copy (nixl-*pathb*.whl)
NIXL_PATHB_WHL=$(find "$WHEELHOUSE" \( -name "nixl_cu12*cp311*.whl" -o -name "nixl-*pathb*.whl" \) 2>/dev/null | head -1 || true)
DYNAMO_WHL=$(ls "$WHEELHOUSE"/ai_dynamo_runtime-*.whl 2>/dev/null | head -1)
VLLM_RDU_SRC=$REPO_ROOT/vllm-rdu  # local clone of andychensn/vllm-rdu

echo "=== rdu-hdi RDU venv build on $(hostname) $(date) ==="
echo "    Python: $($PY --version)"
echo "    VENV: $VENV"
[ -n "$VLLM_CPU_WHL" ] && echo "    vllm wheel: $VLLM_CPU_WHL" || echo "    ERROR: no vllm wheel in $WHEELHOUSE"
[ -n "$NIXL_PATHB_WHL" ] && echo "    nixl wheel: $NIXL_PATHB_WHL" || echo "    ERROR: no nixl-pathb wheel in $WHEELHOUSE"

# Validate
[ -x "$PY" ] || { echo "ERROR: $PY not found — must run on RDU node"; exit 1; }
[ -n "$VLLM_CPU_WHL" ] || { echo "ERROR: no vllm wheel in $WHEELHOUSE — run scripts/fetch_rdu_wheels.sh first"; exit 1; }
[ -n "$NIXL_PATHB_WHL" ] || { echo "ERROR: no nixl wheel found in $WHEELHOUSE"; echo "  Run: bash scripts/build_rdu_ucx_nixl.sh --fetch-only  (login node)"; echo "  Then: snrdu run ... -- bash scripts/build_rdu_ucx_nixl.sh --build-only  (s339)"; exit 1; }

# ── Create venv ───────────────────────────────────────────────────────────────
echo "=== Creating venv ==="
"$PY" -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

# ── vllm 0.16.0 CPU-only ─────────────────────────────────────────────────────
echo "=== vllm (CPU-only, --no-deps to skip torch version check) ==="
pip install -q --no-deps "$VLLM_CPU_WHL"

# Patch vllm for torch 2.2.x compatibility (s339 has torch 2.2.0+sn, not 2.9.1)
# The RDU side only needs Python scheduling/NIXL code, not GPU compute patches.
VLLM_SITE=$(find "$VENV" -name "*.dist-info" -path "*vllm*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)

# 1. torch_utils.py: infer_schema added in torch 2.4
TORCH_UTILS=$(find "$VENV" -name "torch_utils.py" -path "*/vllm/*" 2>/dev/null | head -1)
if grep -q "^from torch.library import Library, infer_schema" "$TORCH_UTILS" 2>/dev/null; then
    sed -i 's/^from torch.library import Library, infer_schema$/try:\n    from torch.library import Library, infer_schema\nexcept ImportError:\n    from torch.library import Library\n    def infer_schema(func, *args, **kwargs): return ""/' "$TORCH_UTILS"
    echo "  infer_schema compat patch: OK"
fi

# 2. env_override.py: full of torch 2.9 inductor patches; replace with minimal version safe for 2.2.x
ENV_OVERRIDE=$(find "$VENV" -name "env_override.py" -path "*/vllm/*" 2>/dev/null | head -1)
if [ -n "$ENV_OVERRIDE" ]; then
    cat > "$ENV_OVERRIDE" << 'PYEOF'
# Minimal env_override.py for torch 2.2.x compatibility on RDU.
# Original file has torch 2.9 inductor patches that crash on torch 2.2.0+sn.
# The RDU decode path only needs Python scheduling code, not GPU compute patches.
import os
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
PYEOF
    echo "  env_override.py: replaced with torch 2.2.x-safe version"
fi

# Verify nixl_connector has REGISTER_CONSUMER_MSG (present in official vllm 0.16.0)
NIXL_PY=$(find "$VENV" -name "nixl_connector.py" -path "*/kv_connector*" 2>/dev/null | head -1)
if grep -q "REGISTER_CONSUMER_MSG" "$NIXL_PY" 2>/dev/null; then
    echo "  REGISTER_CONSUMER_MSG present ✅"
else
    echo "WARNING: REGISTER_CONSUMER_MSG not found — vllm version may differ from $VLLM_VERSION"
fi

# ── Critical pins ─────────────────────────────────────────────────────────────
echo "=== numpy==$RDU_NUMPY_VERSION (system numpy 2.x incompatible with torch 2.2.0+sn) ==="
pip install -q "numpy==$RDU_NUMPY_VERSION"

echo "=== transformers==$RDU_TRANSFORMERS_VERSION ==="
pip install -q "transformers==$RDU_TRANSFORMERS_VERSION"

# ── Dynamo runtime ────────────────────────────────────────────────────────────
echo "=== ai-dynamo-runtime ==="
if [ -n "$DYNAMO_WHL" ]; then
    pip install -q "$DYNAMO_WHL"
else
    # Try PyPI (may not be available from s339)
    pip install -q "ai-dynamo-runtime==$DYNAMO_VERSION" 2>/dev/null || \
        echo "WARNING: could not install ai-dynamo-runtime — add wheel to wheelhouse"
fi

# ── Dynamo Python deps ────────────────────────────────────────────────────────
echo "=== Dynamo Python deps ==="
pip install -q aiohttp msgpack msgspec pyzmq sortedcontainers uvloop cbor2 diskcache 2>/dev/null || \
    echo "WARNING: some Dynamo deps may be missing (no internet on s339)"

# ── NIXL (pathb: Broadcom UCX for bnxt_re) ───────────────────────────────────
echo "=== nixl (pathb wheel) ==="
pip install -q "$NIXL_PATHB_WHL"

# ── vllm-rdu plugin ───────────────────────────────────────────────────────────
echo "=== vllm-rdu (local editable install) ==="
if [ ! -d "$VLLM_RDU_SRC" ]; then
    echo "ERROR: vllm-rdu not found at $VLLM_RDU_SRC"
    echo "  Run: gh repo clone andychensn/vllm-rdu $VLLM_RDU_SRC"
    exit 1
fi
pip install -q -e "$VLLM_RDU_SRC"

# ── Validate ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Validating ==="
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import nixl; print('nixl: OK')"
python -c "import rdu_hardware; print('vllm-rdu: OK')" 2>/dev/null || echo "WARNING: rdu_hardware import failed"
python -c "
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import REGISTER_CONSUMER_MSG
print('sn_vllm patch: OK')
" 2>/dev/null || echo "WARNING: sn_vllm patch check failed"

echo ""
echo "=== RDU venv build COMPLETE $(date) ==="
