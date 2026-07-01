#!/bin/bash
# Install ALL Python deps from wheelhouse into the RDU venv.
# Run on s339 via snrdu. Idempotent.
set -euo pipefail
export PYTHONNOUSERSITE=1
REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
VENV="$REPO_ROOT/.venv_rdu"
WHLHOUSE="$REPO_ROOT/wheelhouse"
source "$VENV/bin/activate"

echo "=== Installing ALL wheels from $WHLHOUSE ==="
# Skip the vllm wheel itself and nixl (already installed) and heavy ML deps
SKIP_PATTERNS="vllm-0.16.0|nixl_cu12|ai_dynamo|vllm_rdu|numpy-|transformers-"
count=0
for whl in "$WHLHOUSE"/*.whl; do
    base=$(basename "$whl")
    if echo "$base" | grep -qE "$SKIP_PATTERNS"; then
        continue
    fi
    pip install -q --no-deps "$whl" 2>/dev/null && echo "  OK: $base" && ((count++)) || echo "  SKIP: $base"
done
echo "Installed $count packages"

echo ""
echo "=== Testing real startup path ==="
python3.11 -c "from dynamo.vllm.main import main; print('dynamo.vllm.main: OK')"
