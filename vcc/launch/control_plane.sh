#!/usr/bin/env bash
# Launch (or stop) the control-plane container on VCC's CONTROL_PLANE_HOST.
# Same image/entrypoint as launch/control_plane.sh (vnc+idc) -- only the
# orchestration differs: direct SSH + rootless Podman instead of
# sudo -g docker /usr/bin/docker-run-wrapper (that wrapper doesn't exist on
# VCC, and rootless Podman needs no sudo at all for a container this simple:
# no GPU/RDU device passthrough). It does bind-mount GPU_MODEL_PATH though
# (see below) -- SELinux Enforcing on CONTROL_PLANE_HOST denies rootless
# Podman read access to that host path without --security-opt label=disable,
# confirmed live (Permission denied inside the container despite correct
# Unix permission bits -- an SELinux type-enforcement denial, not a DAC one).
#
# Usage:
#   bash vcc/launch/control_plane.sh          # launch detached, returns once started
#   bash vcc/launch/control_plane.sh --stop   # graceful SIGTERM to the running container
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/vcc/cluster.env"
source "$REPO_ROOT/vcc/model.env"

if [[ "${1:-}" == "--stop" ]]; then
    ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" "podman exec rdu-hdi-control-plane sh -c 'kill -TERM 1'"
    exit 0
fi

echo "Starting control plane on $CONTROL_PLANE_HOST (image: $CONTROL_PLANE_IMAGE)..."
# -d (detached): a plain foreground `podman run` over a non-pty ssh session
# dies the moment that ssh connection drops (idle timeout, network blip --
# the same class of drop hit during VCC image transfers) -- confirmed live:
# this exact thing silently killed a control-plane container hours into an
# otherwise-idle run, with --rm erasing all trace of it. Detached, the
# container is a podman-managed background process independent of the ssh
# session that started it, matching "leave the endpoint live" intent.
#
# IMPORTANT PREREQUISITE: even detached, a rootless-Podman container is
# still killed ~7s after the LAUNCHING user's last login session ends,
# unless that user has systemd lingering enabled (confirmed live: this
# silently killed a freshly-started, otherwise-healthy control-plane
# within seconds, repeatedly, until fixed). One-time fix per VCC node:
#   ssh $CONTROL_PLANE_HOST loginctl enable-linger \$(whoami)
# Same applies to RDU_HOST for the RDU decode container.
#
# Model-path mounts: dynamo.frontend's discovery watcher constructs each
# registering worker's Model Deployment Card by resolving that worker's
# announced model path -- if the path doesn't exist locally in THIS
# container, it falls through to a remote fetch (ModelExpress, then public
# HuggingFace), which always fails here (no path to either from VCC) with
# a 404, leaving the model permanently unregistered (empty /v1/models,
# every real request 404s) even though the worker itself registered fine.
# vnc+idc never hits this because docker-run-wrapper auto-mounts
# /import,/scratch into EVERY container including control-plane, so both
# workers' model paths are already visible there. VCC has no such mount:
#   - GPU_MODEL_PATH is local to CONTROL_PLANE_HOST (same node) -- mount directly.
#   - RDU_MODEL_PATH lives on a DIFFERENT physical node (RDU_HOST, no shared
#     filesystem) -- can't be bind-mounted directly. Only its own small
#     metadata files (config.json, tokenizer*, etc -- NOT the multi-GB
#     safetensors weights) are actually needed to satisfy the local-path
#     check, staged once via scp to RDU_MODEL_METADATA_DIR (vcc/model.env)
#     and bind-mounted at RDU_MODEL_PATH's own path so the container sees
#     it at the exact path RDU decode announces.
#
# Tried repointing RDU decode's registration at a real public HF id
# instead, to drop this staging step -- reverted (2026-07-16): the
# discovery watcher's download path doesn't honor ignore_weights, so it
# started pulling a real safetensors weight shard instead of just
# metadata (only stopped by HF's own rate limit). Local staging has no
# such risk -- no network fetch involved at all.
#
# Auto-stage RDU's metadata onto CONTROL_PLANE_HOST if not already there
# (idempotent, same pattern as vcc/launch/gpu_prefill.sh's
# GPU_DRIVER_LIBS_DIR check) -- first run per account/node copies a fixed
# list of small non-weight files from RDU_HOST via scp -3; every run after
# that is a no-op. Re-run this script (or delete RDU_MODEL_METADATA_DIR)
# if RDU_HOST's checkpoint's config/tokenizer ever changes.
RDU_MODEL_METADATA_FILES=(
    config.json tokenizer.json tokenizer_config.json vocab.json merges.txt
    generation_config.json chat_template.jinja configuration_minimax_m2.py
    modeling_minimax_m2.py model.safetensors.index.json LICENSE README.md
)
if ! ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" "[ -e '$RDU_MODEL_METADATA_DIR/config.json' ]"; then
    echo "First run on $CONTROL_PLANE_HOST: staging RDU model metadata into $RDU_MODEL_METADATA_DIR..."
    ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" "mkdir -p '$RDU_MODEL_METADATA_DIR'"
    for f in "${RDU_MODEL_METADATA_FILES[@]}"; do
        scp -3 -q -o BatchMode=yes \
            "$RDU_HOST:$RDU_MODEL_PATH/$f" \
            "$CONTROL_PLANE_HOST:$RDU_MODEL_METADATA_DIR/$f"
    done
fi

exec ssh -o BatchMode=yes "$CONTROL_PLANE_HOST" \
    "podman run -d --replace --net=host --rm \
        --name rdu-hdi-control-plane \
        --security-opt label=disable \
        -v $GPU_MODEL_PATH:$GPU_MODEL_PATH:ro \
        -v $RDU_MODEL_METADATA_DIR:$RDU_MODEL_PATH:ro \
        -e CONTROL_PLANE_IP=$CONTROL_PLANE_IP -e ETCD_PORT=$ETCD_PORT \
        -e NATS_PORT=$NATS_PORT -e VLLM_PORT=$VLLM_PORT \
        -e BLOCK_SIZE=$BLOCK_SIZE \
        $CONTROL_PLANE_IMAGE"
