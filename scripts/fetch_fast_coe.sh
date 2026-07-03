#!/usr/bin/env bash
# Vendor fast-coe (hdi's proven vllm-rdu source) into this repo, pinned by commit.
#
# fast-coe/server/vllm-rdu is the RDU decode connector/engine code hdi actually runs
# (rdu_hardware/connector_override.py: 4 independent per-NIC NIXL agents, BAR2/DDR
# slab cache, cache-aware routing) — a different, working lineage from the diverged
# andychensn/vllm-rdu fork this repo used to install. See docs/local/PARITY_PLAN.md.
#
# The RDU node has no internet, so the tree must be present on shared NFS storage
# at build+run time (this is a PYTHONPATH/editable-install source tree, not a wheel).
#
# Usage:
#   bash scripts/fetch_fast_coe.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

FAST_COE_DIR="$REPO_ROOT/fast-coe"

if [ -d "$FAST_COE_DIR/.git" ]; then
    CURRENT=$(git -C "$FAST_COE_DIR" rev-parse HEAD)
    if [ "$CURRENT" = "$FAST_COE_COMMIT" ]; then
        echo "fast-coe already present at $FAST_COE_COMMIT ✅"
        exit 0
    else
        echo "ERROR: $FAST_COE_DIR exists but is at $CURRENT, expected $FAST_COE_COMMIT"
        echo "  Remove it and re-run to re-fetch, or checkout the pin manually."
        exit 1
    fi
fi

echo "Cloning sambanova/fast-coe@$FAST_COE_COMMIT..."
git clone git@github.com:sambanova/fast-coe.git "$FAST_COE_DIR"
git -C "$FAST_COE_DIR" checkout "$FAST_COE_COMMIT"
echo "fast-coe cloned + pinned ✅"

echo ""
echo "=== Validating expected layout ==="
for p in server/vllm-rdu/rdu_hardware/connector_override.py \
         server/rdu_manifest \
         server/inference-router/client-py/inference_router \
         server/block_hash; do
    if [ -e "$FAST_COE_DIR/$p" ]; then
        echo "  OK: $p"
    else
        echo "  WARNING: missing expected path: $p"
    fi
done
