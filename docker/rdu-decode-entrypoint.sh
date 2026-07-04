#!/usr/bin/env bash
# RDU decode container entrypoint. Adapts launch/rdu_decode.sh's --inner
# block (the part that actually runs `python -m dynamo.vllm`) for a
# container context:
#   - No venv activation — /opt/sambanova/bin/python3.11 is already on
#     PATH, with everything scripts/build_rdu_env.sh would have installed
#     into .venv_rdu baked directly into this image's site-packages
#     (docker/rdu-decode-install-deps.sh, run at image build time).
#   - UCX is at a fixed path baked into the image (/opt/rdu-ucx), not
#     resolved relative to $REPO_ROOT.
#   - fast-coe is at a fixed path baked into the image (/build/fast-coe).
#   - coe_api/rdu_engine and the BAR2 runtime connector libs are NOT baked
#     in — BAR2_INSTALL/BAR2_RUNTIME_LIBS/BAR2_PRELOAD must be passed in as
#     env vars pointing at NFS paths visible inside the container
#     (docker-run-wrapper auto-mounts /import and /scratch at identical
#     host paths on RDU nodes; a k8s pod spec needs an explicit hostPath
#     volume for the same paths — see Phase 5 k8s manifests). Cluster
#     topology (CONTROL_PLANE_IP, GPU_ROCE_IP, ...) and model config
#     (MODEL, PEF, ...) are passed in as env vars too, not baked in, since
#     both vary independently of the image build.
set -euo pipefail
export PYTHONNOUSERSITE=1

: "${CONTROL_PLANE_IP:?CONTROL_PLANE_IP must be set}"
: "${ETCD_PORT:=12379}"
: "${NATS_PORT:=14222}"
: "${GPU_ROCE_IP:?GPU_ROCE_IP must be set (the GPU prefill RoCE IP)}"
: "${MODEL:?MODEL must be set (checkpoint path)}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN must be set}"
: "${PEF:?PEF must be set (compiled PEF path)}"
: "${MODEL_CONFIG:=}"
: "${BAR2_INSTALL:?BAR2_INSTALL must be set (mounted NFS path — see Dockerfile.rdu header)}"
: "${BAR2_RUNTIME_LIBS:?BAR2_RUNTIME_LIBS must be set (mounted NFS path)}"
: "${BAR2_PRELOAD:?BAR2_PRELOAD must be set (mounted NFS path)}"
: "${RDU_CACHE:=/tmp/rdu-cache}"

FAST_COE_SRC=/build/fast-coe
_UCX_LIB=/opt/rdu-ucx/lib
_UCX_MOD=/opt/rdu-ucx/lib/ucx

mkdir -p "$RDU_CACHE"

for p in "$BAR2_INSTALL/python" "$BAR2_RUNTIME_LIBS" "$BAR2_PRELOAD/libc_samba_runtime.so"; do
    [ -e "$p" ] || { echo "ERROR: $p not found — is the NFS path actually mounted into this container?"; exit 1; }
done

# /dev/rdu and /dev/rdu_mem_map are NOT provided by docker-run-wrapper's
# automatic /import,/scratch NFS mounting (that only covers filesystem
# paths, not device nodes) — they must be passed explicitly, e.g.
# `--device /dev/rdu --device /dev/rdu_mem_map` (docker) or via the k8s
# device plugin (sambanova.ai/rdu-tile, see Phase 5 k8s manifests). Both
# are world-writable (crw-rw-rw-) on this cluster, confirmed via direct
# inspection, so no privileged mode is needed — just explicit passthrough.
# Fail with a clear message here rather than the cryptic native
# "Unable to initialize Storage Direct" / NovaRuntime assert this produces
# otherwise.
for d in /dev/rdu /dev/rdu_mem_map; do
    [ -e "$d" ] || { echo "ERROR: $d not found — was it passed through to this container? (--device $d, or the k8s device plugin)"; exit 1; }
done

# NIXL ships as a meson-python wheel that vendors its shared libs into a
# dot-prefixed sibling dir in site-packages (e.g. .nixl_cu12.mesonpy.libs/)
# — libnixl.so etc. live directly in that dir; the UCX backend plugin
# (libplugin_UCX.so) lives one level deeper, in its own plugins/ subdir.
SITE_PACKAGES=$(/opt/sambanova/bin/python3.11 -c "import site; print(site.getsitepackages()[0])")
_NIXL_LIBS_DIR=$(find "$SITE_PACKAGES" -maxdepth 1 -iname "*.nixl*.mesonpy.libs" 2>/dev/null | head -1)
[ -n "$_NIXL_LIBS_DIR" ] || { echo "ERROR: nixl mesonpy libs dir not found under $SITE_PACKAGES"; exit 1; }
_NIXL_LIB="$_NIXL_LIBS_DIR"
_NIXL_PLUGIN_DIR="$_NIXL_LIBS_DIR/plugins"

# Requires --net=host (docker) / hostNetwork: true (k8s) to see the real
# host RoCE interface — same requirement launch/rdu_decode.sh already has
# on bare metal.
RDU_ROCE_IP_LOCAL=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep '^10\.17\.' | head -1 || true)
RDU_ROCE_IP_LOCAL=${RDU_ROCE_IP_LOCAL:-$(hostname -I 2>/dev/null | awk '{print $1}')}

echo "=== RDU decode on $(hostname) $(date) ==="
echo "    etcd:         http://$CONTROL_PLANE_IP:$ETCD_PORT"
echo "    producer:     $GPU_ROCE_IP:5600"
echo "    side-channel: $RDU_ROCE_IP_LOCAL:5600"

RDU_CONFIG="$MODEL_CONFIG"
[ -n "$RDU_CONFIG" ] && [ ! -f "$RDU_CONFIG" ] && {
    echo "WARNING: MODEL_CONFIG=$RDU_CONFIG not found — running without model config"
    RDU_CONFIG=""
}
[ -n "$RDU_CONFIG" ] && echo "  rdu_config: $RDU_CONFIG (fast-coe schema, used as-is)"

exec env \
    PYTHONNOUSERSITE=1 \
    ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
    DYN_REQUEST_PLANE=tcp \
    HF_HUB_OFFLINE=1 \
    VLLM_NIXL_SIDE_CHANNEL_HOST="$RDU_ROCE_IP_LOCAL" \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    RDU_ENABLED=1 \
    VLLM_PD_CHUNK_OVERLAP=0 \
    VLLM_PD_PRODUCER_CHUNK_OVERLAP=0 \
    VLLM_PD_PRODUCER_HOST="$GPU_ROCE_IP" \
    VLLM_PD_PRODUCER_PORT=5600 \
    SN_REMOTE_KV_MEMTYPE=VRAM \
    RAW_FORCE_PER_XFER=1 \
    SN_MULTI_AGENT_NICS="${SN_MULTI_AGENT_NICS:-bnxt_re0,bnxt_re2,bnxt_re4,bnxt_re6}" \
    VLLM_CPU_KVCACHE_SPACE=200 \
    VLLM_CPU_OMP_THREADS_BIND=nobind \
    UCX_TLS=rc \
    UCX_NET_DEVICES="${UCX_NET_DEVICES:-bnxt_re0:1,bnxt_re2:1,bnxt_re4:1,bnxt_re6:1}" \
    UCX_MAX_RNDV_RAILS=4 \
    UCX_MAX_RMA_RAILS=1 \
    UCX_MULTI_LANE_MAX_RATIO=100 \
    UCX_IB_PREFER_NEAREST_DEVICE=n \
    UCX_IB_ROCE_REACHABILITY_MODE=all \
    UCX_RC_VERBS_MAX_RD_ATOMIC=16 \
    UCX_RC_VERBS_TIMEOUT=20000us \
    UCX_RC_VERBS_TX_QUEUE_LEN=256 \
    UCX_RC_VERBS_RX_QUEUE_LEN=4095 \
    UCX_RC_VERBS_RETRY_COUNT=7 \
    UCX_RC_VERBS_RNR_RETRY_COUNT=7 \
    UCX_RCACHE_MAX_UNRELEASED=1024 \
    UCX_LOG_LEVEL=WARN \
    UCX_MODULE_DIR="$_UCX_MOD" \
    NIXL_PLUGIN_DIR="$_NIXL_PLUGIN_DIR" \
    INFERENCE_MODE=local_queue \
    PROG_LOAD=HBM \
    ARG_LOAD=HBM \
    ENABLE_STRICT_CONVERSION=1 \
    SF_RNT_FSM_POLL_BUSY_WAIT=1 \
    SF_RNT_DMA_POLL_BUSY_WAIT=1 \
    SF_RNT_NUMA_BIND="${SF_RNT_NUMA_BIND:-2}" \
    SF_RNT_LOG_LEVEL=ERR \
    AL_EXEC_LOOPING=1 \
    GRAPH_TIMEOUT_USEC=120000000 \
    TRITON_CACHE_DIR="$RDU_CACHE/triton" \
    VLLM_CACHE_ROOT="$RDU_CACHE/vllm" \
    HF_HOME="$RDU_CACHE/huggingface" \
    VLLM_CONFIG_ROOT="$RDU_CACHE/vllm_config" \
    TRANSFORMERS_CACHE="$RDU_CACHE/huggingface" \
    PYTHONPATH="$BAR2_INSTALL/python:$FAST_COE_SRC:$FAST_COE_SRC/server/inference-router/client-py:$FAST_COE_SRC/server/block_hash:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$_UCX_LIB:$_NIXL_LIB:$BAR2_RUNTIME_LIBS:$BAR2_INSTALL/lib:${LD_LIBRARY_PATH:-}" \
    LD_PRELOAD="$BAR2_PRELOAD/libc_samba_runtime.so:$BAR2_PRELOAD/libcpp_samba_runtime.so${LD_PRELOAD:+:$LD_PRELOAD}" \
    /opt/sambanova/bin/python3.11 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --disaggregation-mode decode \
        --load-format dummy \
        --max-num-seqs 2 \
        --tensor-parallel-size 1 \
        --no-enable-prefix-caching \
        --max-model-len "$MAX_MODEL_LEN" \
        --reasoning-parser minimax_m2_append_think \
        --compilation-config '{"mode": 0}' \
        ${RDU_CONFIG:+--additional-config "{\"rdu_config\": \"$RDU_CONFIG\"}"} \
        --trust-remote-code \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"rdu","kv_connector_extra_config":{"rdu_mode":"real","rdu_ddr_cache_budget_gb":30,"backends":["UCX"],"enforce_handshake_compat":false}}'
