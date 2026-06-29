#!/usr/bin/env bash
# Smoke-test the Dynamo runtime imports in the RDU venv.
# Run ON s339 via snrdu after building the RDU venv.
# Usage: bash scripts/test_dynamo_import.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/cluster.env"

RDU_VENV="${RDU_VENV:-$REPO_ROOT/.venv_rdu}"
source "$RDU_VENV/bin/activate"

SOFTWARE_BUILD=${SOFTWARE_BUILD:-/import/snvm-sc-scratch1/guoyaof/software/runtime/build}
RDU_UCX="${RDU_UCX:-$REPO_ROOT/rdu-ucx-install}"
export LD_LIBRARY_PATH="$RDU_UCX/lib:$SOFTWARE_BUILD/ucx-install/lib:$SOFTWARE_BUILD/nixl-install/lib:$SOFTWARE_BUILD/etcd-cpp-api-install/lib:${LD_LIBRARY_PATH:-}"

echo "=== Dynamo import test on $(hostname) ==="
python3 << 'EOF'
results = []
for mod, label in [("dynamo", "dynamo"), ("dynamo._core", "dynamo._core")]:
    try:
        __import__(mod)
        results.append(f"  ✅ {label}")
    except Exception as e:
        results.append(f"  ❌ {label}: {e}")
for r in results:
    print(r)
fail = sum(1 for r in results if "❌" in r)
print(f"\n{'PASS' if fail == 0 else 'FAIL'} — {len(results)-fail}/{len(results)} imports OK")
import sys; sys.exit(fail)
EOF
