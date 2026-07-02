#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1
source /import/snvm-sc-scratch1/andyc/rdu-hdi/.venv_rdu/bin/activate
python3.11 -c "
import rdu_engine
t = rdu_engine.RDUTensor
print('RDUTensor has __setitem__:', hasattr(t, '__setitem__'))
print('RDUTensor has dtype property:', hasattr(t, 'dtype'))
print('RDUTensor has shape property:', hasattr(t, 'shape'))
c = rdu_engine.Checkpoint if hasattr(rdu_engine, 'Checkpoint') else None
print('Checkpoint class found:', c)
if c is not None:
    print('Checkpoint has get_symbol_properties:', hasattr(c, 'get_symbol_properties'))
tl = rdu_engine.TensorLayout if hasattr(rdu_engine, 'TensorLayout') else None
print('TensorLayout class found:', tl)
if tl is not None:
    print('TensorLayout has dtype:', hasattr(tl, 'dtype'))
    print('TensorLayout has shape:', hasattr(tl, 'shape'))
"
