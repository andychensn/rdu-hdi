#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/import/snvm-sc-scratch1/andyc/rdu-hdi
cd "$REPO_ROOT"

echo "=== pip install self-built coe_api/rdu_engine wheels into .venv_rdu ==="
.venv_rdu/bin/python3.11 -m pip install --no-deps --force-reinstall \
    wheelhouse/sambanova_rdu_engine_api-1.0.0-cp311-cp311-linux_x86_64.whl \
    wheelhouse/sambanova_coe_api-1.0.0-cp311-cp311-linux_x86_64.whl

echo ""
echo "=== verify import (self-built, via .venv_rdu site-packages, NOT PYTHONPATH) ==="
LD_LIBRARY_PATH="$REPO_ROOT/rdu-runtime-install/lib:${LD_LIBRARY_PATH:-}" \
    .venv_rdu/bin/python3.11 -c "
import rdu_engine
print('rdu_engine file:', rdu_engine.__file__)
print('has Checkpoint:', hasattr(rdu_engine, 'Checkpoint'))
print('has PEF:', hasattr(rdu_engine, 'PEF'))
print('RDUTensor has dtype:', hasattr(rdu_engine.RDUTensor, 'dtype'))
import coe_api
print('coe_api file:', coe_api.__file__)
print('coe_api.Checkpoint.__module__:', coe_api.Checkpoint.__module__)
print('coe_api.RDUTensor has dtype:', hasattr(coe_api.RDUTensor, 'dtype'))
print('IMPORT OK')
"
