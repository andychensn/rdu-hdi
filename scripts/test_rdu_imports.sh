#!/usr/bin/env bash
# Smoke-test all key Python imports in the RDU venv.
# Run ON s339 via snrdu after building the RDU venv.
# Usage: bash scripts/test_rdu_imports.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/cluster.env"

RDU_VENV="${RDU_VENV:-$REPO_ROOT/.venv_rdu}"
source "$RDU_VENV/bin/activate"

SOFTWARE_BUILD=${SOFTWARE_BUILD:-/import/snvm-sc-scratch1/guoyaof/software/runtime/build}
RDU_UCX="${RDU_UCX:-$REPO_ROOT/rdu-ucx-install}"
export LD_LIBRARY_PATH="$RDU_UCX/lib:$SOFTWARE_BUILD/ucx-install/lib:$SOFTWARE_BUILD/nixl-install/lib:$SOFTWARE_BUILD/etcd-cpp-api-install/lib:${LD_LIBRARY_PATH:-}"

PASS=0; FAIL=0
check() { if eval "$1" 2>/dev/null; then echo "  ✅ $2"; ((PASS++)); else echo "  ❌ $2"; ((FAIL++)); fi; }

echo "=== RDU venv import test on $(hostname) ==="
check "python -c 'import vllm; assert vllm.__version__ == \"0.16.0\"'" "vllm 0.16.0"
check "python -c 'import nixl'" "nixl (with LD_LIBRARY_PATH)"
check "python -c 'import numpy; assert numpy.__version__.startswith(\"1.\")'" "numpy 1.x"
check "python -c 'import dynamo'" "ai-dynamo-runtime (dynamo)"
check "python -c \"
import pathlib, vllm
nc = pathlib.Path(vllm.__file__).parent / 'distributed/kv_transfer/kv_connector/v1/nixl_connector.py'
assert 'REGISTER_CONSUMER_MSG' in nc.read_text()
\"" "nixl_connector REGISTER_CONSUMER_MSG present"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "ALL IMPORTS OK ✅" || { echo "SOME IMPORTS FAILED ❌"; exit 1; }
