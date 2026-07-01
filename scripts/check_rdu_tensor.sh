#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1
source /import/snvm-sc-scratch1/andyc/rdu-hdi/.venv_rdu/bin/activate
python3.11 -c "
import rdu_engine
print('rdu_engine version:', getattr(rdu_engine, '__version__', 'unknown'))
t = rdu_engine.RDUTensor
methods = [m for m in dir(t) if not m.startswith('_')]
print('RDUTensor methods:', methods)
if hasattr(rdu_engine, 'CoETensor'):
    ct = rdu_engine.CoETensor
    print('CoETensor methods:', [m for m in dir(ct) if not m.startswith('_')])
"
