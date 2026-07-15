#!/usr/bin/env bash
# Stop all three VCC containers. Unlike vnc+idc's SLURM-based teardown
# (scancel by job name), there's no job/allocation to cancel here -- just
# stop the named Podman containers directly over SSH.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/vcc/cluster.env"

echo "Stopping RDU decode on $RDU_HOST..."
ssh -o BatchMode=yes "$RDU_HOST" "podman stop -t 5 vcc-rdu-decode" 2>&1 || echo "  (not running)"

echo "Stopping GPU prefill on $GPU_HOST..."
ssh -o BatchMode=yes "$GPU_HOST" "podman stop -t 5 vcc-gpu-prefill" 2>&1 || echo "  (not running)"

echo "Stopping control plane on $CONTROL_PLANE_HOST..."
ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" "podman stop -t 5 rdu-hdi-control-plane" 2>&1 || echo "  (not running)"

echo "Done."
