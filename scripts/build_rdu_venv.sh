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

# All wheels must be pre-fetched by: bash scripts/build_rdu_ucx_nixl.sh --fetch-only
# s339 has no internet — nothing is downloaded here.
VLLM_CPU_WHL=$(find "$WHEELHOUSE" -name "vllm-*linux_x86_64.whl" 2>/dev/null | head -1 || true)
NIXL_WHL=$(find "$WHEELHOUSE" -name "nixl_cu12*cp311*.whl" 2>/dev/null | head -1 || true)
DYNAMO_RUNTIME_WHL=$(find "$WHEELHOUSE" -name "ai_dynamo_runtime-*.whl" 2>/dev/null | head -1 || true)
DYNAMO_WHL=$(find "$WHEELHOUSE" -name "ai_dynamo-*.whl" 2>/dev/null | head -1 || true)
VLLM_RDU_SRC=$REPO_ROOT/vllm-rdu  # local clone of andychensn/vllm-rdu

echo "=== rdu-hdi RDU venv build on $(hostname) $(date) ==="
echo "    Python: $($PY --version)"
echo "    VENV: $VENV"

# Validate — all wheels must exist (fetched by build_rdu_ucx_nixl.sh --fetch-only)
[ -x "$PY" ]           || { echo "ERROR: $PY not found — must run on RDU node"; exit 1; }
[ -n "$VLLM_CPU_WHL" ] || { echo "ERROR: no vllm wheel in $WHEELHOUSE"; echo "  Run from login node: bash scripts/build_rdu_ucx_nixl.sh --fetch-only"; exit 1; }
[ -n "$NIXL_WHL" ]     || { echo "ERROR: no nixl wheel in $WHEELHOUSE"; echo "  Run from login node: bash scripts/build_rdu_ucx_nixl.sh --fetch-only"; echo "  Then on s339: bash scripts/build_rdu_ucx_nixl.sh --build-only"; exit 1; }
[ -n "$DYNAMO_RUNTIME_WHL" ] || { echo "ERROR: no ai-dynamo-runtime wheel in $WHEELHOUSE"; echo "  Run from login node: bash scripts/build_rdu_ucx_nixl.sh --fetch-only"; exit 1; }
[ -n "$DYNAMO_WHL" ]         || { echo "ERROR: no ai-dynamo wheel in $WHEELHOUSE"; echo "  Run from login node: bash scripts/build_rdu_ucx_nixl.sh --fetch-only"; exit 1; }
echo "    vllm:            $VLLM_CPU_WHL"
echo "    nixl:            $NIXL_WHL"
echo "    dynamo-runtime:  $DYNAMO_RUNTIME_WHL"
echo "    dynamo:          $DYNAMO_WHL"

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
echo "=== ai-dynamo-runtime + ai-dynamo ==="
pip install -q --no-deps "$DYNAMO_RUNTIME_WHL"
pip install -q --no-deps "$DYNAMO_WHL"

# ── Dynamo Python deps ────────────────────────────────────────────────────────
echo "=== Dynamo Python deps ==="
pip install -q aiohttp msgpack msgspec pyzmq sortedcontainers uvloop cbor2 diskcache 2>/dev/null || \
    echo "WARNING: some Dynamo deps unavailable (no internet on s339) — may already be in system site-packages"

# ── NIXL (built from source by build_rdu_ucx_nixl.sh --build-only) ───────────
echo "=== nixl ==="
pip install -q "$NIXL_WHL"

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
