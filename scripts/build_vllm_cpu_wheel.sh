#!/usr/bin/env bash
# Build a vllm-0.16.0+cpu wheel compatible with torch 2.2.x (SambaNova s339 RDU nodes).
#
# Two-phase: run --fetch-only on the login node (needs internet), then
# --build-only on s339 via snrdu (no internet, but has torch 2.2.0+sn).
#
# Phase 1 (login node — needs internet):
#   bash scripts/build_vllm_cpu_wheel.sh --fetch-only
#
# Phase 2 (s339 via snrdu):
#   source config/cluster.env config/model.env
#   snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
#       --allow-local-lib-python --reservation "$RDU_RESERVATION" \
#       --pef "$PEF" --timeout 00:30:00 \
#       -o logs/build_vllm_cpu_wheel.log \
#       -- bash scripts/build_vllm_cpu_wheel.sh --build-only
#
# Output: wheelhouse/vllm-0.16.0+cpu-cp311-cp311-linux_x86_64.whl
#
# Why a custom wheel (not the PyPI vllm==0.16.0)?
#   PyPI vllm 0.16.0 targets torch 2.9.x and uses APIs unavailable in torch 2.2.0+sn:
#   _symmetric_memory, _unregister_process_group, custom_graph_pass, infer_schema, etc.
#   This script applies torch 2.2.x compat patches (from patches/ in this repo) to the
#   vllm 0.16.0 source before building, producing a +cpu wheel that works on s339.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"

VLLM_SRC="$REPO_ROOT/rdu-build-src/vllm-patched"
WHEEL_OUT="$REPO_ROOT/wheelhouse"
PY=/opt/sambanova/bin/python3.11
MODE="${1:---fetch-only}"

# ── Phase 1: fetch + patch (login node) ───────────────────────────────────────
fetch_and_patch() {
    echo "=== Phase 1: clone vllm v$VLLM_VERSION and apply torch 2.2.x patches ==="

    if [ ! -d "$VLLM_SRC/.git" ]; then
        echo "  Cloning vllm v$VLLM_VERSION..."
        mkdir -p "$(dirname "$VLLM_SRC")"
        git clone --depth=1 --branch "v$VLLM_VERSION" \
            https://github.com/vllm-project/vllm.git "$VLLM_SRC"
        echo "  Cloned ✅"
    else
        echo "  Source already present at $VLLM_SRC"
    fi

    echo ""
    echo "=== Applying torch 2.2.x compat patches ==="

    # Patch 1: env_override.py — replace minimal upstream version with comprehensive
    # torch 2.2.x compat shim that pre-registers stubs for _symmetric_memory,
    # _unregister_process_group, custom_graph_pass, and ~10 other torch 2.5+/2.9.x
    # symbols via sys.modules injection before any vllm submodule is imported.
    PATCH1="$REPO_ROOT/patches/vllm_env_override_torch22x.py"
    [ -f "$PATCH1" ] || { echo "ERROR: $PATCH1 not found"; exit 1; }
    cp "$PATCH1" "$VLLM_SRC/vllm/env_override.py"
    echo "  env_override.py: replaced with torch 2.2.x compat shim ✅"

    # Patch 2: torch_utils.py — add _infer_schema_available flag so
    # direct_register_custom_op skips op registration on torch 2.2.x
    # (infer_schema not available → schema_str="" → define() fails with bare op name).
    TORCH_UTILS="$VLLM_SRC/vllm/utils/torch_utils.py"
    if grep -q "^from torch.library import Library, infer_schema" "$TORCH_UTILS" 2>/dev/null; then
        sed -i 's/^from torch.library import Library, infer_schema$/try:\n    from torch.library import Library, infer_schema\n    _infer_schema_available = True\nexcept ImportError:\n    from torch.library import Library\n    _infer_schema_available = False\n    def infer_schema(func, *args, **kwargs): return ""/' "$TORCH_UTILS"
        echo "  torch_utils.py: _infer_schema_available flag added ✅"
    else
        echo "  torch_utils.py: infer_schema import pattern not found (already patched?)"
    fi

    # Add early-return guard to direct_register_custom_op if not already present
    if grep -q "^def direct_register_custom_op" "$TORCH_UTILS" 2>/dev/null && \
       ! grep -q "_infer_schema_available" "$TORCH_UTILS" 2>/dev/null; then
        # Insert the guard line right after the closing ): of the function signature
        sed -i '/^def direct_register_custom_op/,/^    """/{/^    """/{i\    if not _infer_schema_available:\n        return
}}' "$TORCH_UTILS"
        echo "  direct_register_custom_op: early-return guard added ✅"
    elif grep -q "_infer_schema_available" "$TORCH_UTILS" 2>/dev/null; then
        echo "  direct_register_custom_op: guard already present ✅"
    fi

    echo ""
    echo "=== Phase 1 complete. Run Phase 2 on s339. ==="
    echo "  snrdu run ... -- bash scripts/build_vllm_cpu_wheel.sh --build-only"
}

# ── Phase 2: build wheel on s339 ──────────────────────────────────────────────
build_wheel() {
    echo "=== Phase 2: build vllm $VLLM_VERSION+cpu on $(hostname) $(date) ==="
    [ -x "$PY" ] || { echo "ERROR: $PY not found — must run on RDU node"; exit 1; }
    [ -d "$VLLM_SRC" ] || { echo "ERROR: $VLLM_SRC not found — run --fetch-only first"; exit 1; }

    mkdir -p "$WHEEL_OUT"

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

    # Build with --no-build-isolation so vllm's setup.py can import the system torch.
    # The license fix above (Apache-2.0 SPDX) ensures any setuptools version accepts it.
    # VLLM_TARGET_DEVICE=empty: tells setup.py to set ext_modules=[] (no cmake, no C extensions).
    # This produces a pure-Python wheel. The RDU decode only needs vllm's Python scheduling
    # code; it never calls GPU C extensions directly (those are stubbed or handled by vllm-rdu).
    echo "  Building pure-Python wheel (no cmake, ~1 min)..."
    cd "$VLLM_SRC"
    VLLM_TARGET_DEVICE=empty \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}+cpu" \
        "$PY" -m pip wheel . \
        --no-deps \
        --no-build-isolation \
        --no-cache-dir \
        --wheel-dir "$WHEEL_OUT"

    WHL=$(find "$WHEEL_OUT" -name "vllm-${VLLM_VERSION}+cpu*.whl" | head -1 || true)
    if [ -n "$WHL" ]; then
        echo ""
        echo "=== Done: $WHL ==="
        echo "Next: bash scripts/build_rdu_venv.sh"
    else
        echo "ERROR: wheel not found in $WHEEL_OUT — see build output above"
        ls "$WHEEL_OUT"/vllm-*.whl 2>/dev/null || echo "  (no vllm wheels)"
        exit 1
    fi
}

case "$MODE" in
    --fetch-only) fetch_and_patch ;;
    --build-only) build_wheel ;;
    *) echo "Usage: $0 [--fetch-only | --build-only]"; exit 1 ;;
esac
