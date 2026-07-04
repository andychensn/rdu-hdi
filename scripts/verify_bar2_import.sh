#!/usr/bin/env bash
REPO_ROOT=/import/snvm-sc-scratch1/andyc/rdu-hdi
LD_LIBRARY_PATH="$REPO_ROOT/rdu-runtime-install/lib:${LD_LIBRARY_PATH:-}" \
    "$REPO_ROOT/.venv_rdu/bin/python" -c "
import rdu_engine
print('rdu_engine file:', rdu_engine.__file__)
print('has Checkpoint:', hasattr(rdu_engine, 'Checkpoint'))
print('has PEF:', hasattr(rdu_engine, 'PEF'))
import coe_api
print('coe_api file:', coe_api.__file__)
print('coe_api.Checkpoint.__module__:', coe_api.Checkpoint.__module__)
print('IMPORT OK')
"
