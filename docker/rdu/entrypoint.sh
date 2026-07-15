#!/usr/bin/env bash
# RDU decode container entrypoint -- runs `python -m dynamo.vllm` in
# disaggregated decode mode.
#   - /opt/sambanova/bin/python3.11 is used directly; everything
#     build/rdu_env.sh / docker/rdu/install-deps.sh installs
#     lives in its site-packages (no venv).
#   - UCX is at a fixed path baked into the image (/opt/rdu-ucx).
#   - sambanova/vllm-rdu is pip-installed (editable, from the pinned
#     checkout baked into the image at build time) -- registers itself via
#     vLLM's own hardware-plugin entry point (vllm.platform_plugins), no
#     PYTHONPATH wiring needed here.
#   - coe_api/rdu_engine (pip-installed) and the BAR2 runtime connector libs
#     (/opt/bar2-runtime/{lib,preload}) are baked into the image at build
#     time (self-built, see build/bar2.sh). Cluster topology
#     (CONTROL_PLANE_IP, GPU_ROCE_IP, ...) and model config (MODEL, PEF, ...)
#     are passed in as env vars instead, since both vary independently of
#     the image build -- MODEL/PEF point at NFS paths for the
#     checkpoint/compiled-graph data, which this image doesn't embed.
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
: "${RDU_CACHE:=/tmp/rdu-cache}"

_UCX_LIB=/opt/rdu-ucx/lib
_UCX_MOD=/opt/rdu-ucx/lib/ucx
_BAR2_LIB=/opt/bar2-runtime/lib
_BAR2_PRELOAD=/opt/bar2-runtime/preload

mkdir -p "$RDU_CACHE"

for p in "$_BAR2_LIB" "$_BAR2_PRELOAD/libc_samba_runtime.so"; do
    [ -e "$p" ] || { echo "ERROR: $p not found — image build is incomplete (expected to be baked in, see docker/rdu/Dockerfile)"; exit 1; }
done

# /dev/rdu and /dev/rdu_mem_map are NOT provided by docker-run-wrapper's
# automatic /import,/scratch NFS mounting (that only covers filesystem
# paths, not device nodes) — they must be passed explicitly, e.g.
# `--device /dev/rdu --device /dev/rdu_mem_map` (docker) or via a k8s
# device plugin (sambanova.ai/rdu-tile) in a k8s deployment. Both
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
# host RoCE interface.
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
[ -n "$RDU_CONFIG" ] && echo "  rdu_config: $RDU_CONFIG (vllm-rdu's flat RduConfig schema, used as-is)"

# Dynamo KV-router block-size fix: Dynamo's PrefillRouter is built from
# whichever model card it happens to see with
# ModelType.Chat/Completions -- i.e. this decode worker's card -- and uses that same
# block size to validate every KV-cache-block event GPU prefill publishes (see
# convert.rs's equality-drop guard). Unlike fast-coe's RDUPlatform (which
# force-overrode vllm_config.cache_config.block_size regardless of any CLI
# flag), vllm-rdu's RDUPlatform (vllm_rdu/platform.py) only VALIDATES
# --block-size against the PEF's real page size -- it raises ValueError if
# --block-size is missing/invalid rather than silently computing one. The
# explicit --block-size 256 flag below (RDU's real, hardware-mandated 64 KiB
# physical paging chunk -- 256 tokens is what fits in one such chunk for
# this model) is therefore REQUIRED now, not just documentation of an
# already-enforced value. But Dynamo's own dynamo_kv_event_block_size
# additional_config key lets us report a *different* value to Dynamo's
# discovery/routing layer without touching cache_config.block_size at all
# (components/src/dynamo/vllm/cache_info.py's
# get_configured_kv_event_block_size() reads this key before falling back to
# cache_config.block_size). Safe here specifically because this worker runs with
# --no-enable-prefix-caching below, so it never constructs a KV event publisher and has
# no real KV-event stream of its own to mislabel -- the only consumer of this value in
# our topology is the one-time model-card registration that feeds PrefillRouter::new.
ADDITIONAL_CONFIG_JSON="{\"dynamo_kv_event_block_size\": ${GPU_PREFILL_BLOCK_SIZE:?GPU_PREFILL_BLOCK_SIZE must be set (GPU prefill's real --block-size, so decode reports the same value prefill's KV events are actually chunked at)}"
[ -n "$RDU_CONFIG" ] && ADDITIONAL_CONFIG_JSON="${ADDITIONAL_CONFIG_JSON}, \"rdu_config\": \"$RDU_CONFIG\""
ADDITIONAL_CONFIG_JSON="${ADDITIONAL_CONFIG_JSON}}"
echo "  additional-config: $ADDITIONAL_CONFIG_JSON"

# UCX_RCACHE_MAX_UNRELEASED=1024 below looks redundant with vllm's own
# nixl_connector.py, which auto-sets it to "1024" if unset -- but only if
# nixl hasn't been imported yet when that code runs. GPU prefill (which
# does NOT set this var explicitly) hits exactly that failure: its logs
# (logs/*_gpu_prefill.log) show "NIXL was already imported, we can't reset
# UCX_RCACHE_MAX_UNRELEASED. Please set it to '1024' manually." RDU decode
# never logs that warning, precisely because setting it explicitly here
# makes vllm's own `if "UCX_RCACHE_MAX_UNRELEASED" not in os.environ` check
# short-circuit before it ever needs to auto-set anything. Do not remove
# this as "redundant with upstream default" without re-verifying that
# warning doesn't start appearing in RDU decode's own logs too.
exec env \
    PYTHONNOUSERSITE=1 \
    ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
    NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
    HF_HUB_OFFLINE=1 \
    VLLM_NIXL_SIDE_CHANNEL_HOST="$RDU_ROCE_IP_LOCAL" \
    RDU_ENABLED=1 \
    VLLM_RDU_PLUGIN_TIME_PROFILE=1 \
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
    UCX_MULTI_LANE_MAX_RATIO=100 \
    UCX_IB_PREFER_NEAREST_DEVICE=n \
    UCX_IB_ROCE_REACHABILITY_MODE=all \
    UCX_RC_VERBS_MAX_RD_ATOMIC=16 \
    UCX_RC_VERBS_TIMEOUT=20000us \
    UCX_RCACHE_MAX_UNRELEASED=1024 \
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
    LD_LIBRARY_PATH="$_UCX_LIB:$_NIXL_LIB:$_BAR2_LIB:${LD_LIBRARY_PATH:-}" \
    LD_PRELOAD="$_BAR2_PRELOAD/libc_samba_runtime.so:$_BAR2_PRELOAD/libcpp_samba_runtime.so${LD_PRELOAD:+:$LD_PRELOAD}" \
    /opt/sambanova/bin/python3.11 -m dynamo.vllm \
        --model "$MODEL" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --disaggregation-mode decode \
        --load-format dummy \
        --max-num-seqs 2 \
        --tensor-parallel-size 1 \
        --no-enable-prefix-caching \
        --block-size 256 \
        --max-model-len "$MAX_MODEL_LEN" \
        --reasoning-parser minimax_m2_append_think \
        --compilation-config '{"mode": 0}' \
        --additional-config "$ADDITIONAL_CONFIG_JSON" \
        --trust-remote-code \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"rdu","kv_connector_extra_config":{"rdu_mode":"real","rdu_ddr_cache_budget_gb":30,"rdu_num_dp_groups":2,"backends":["UCX"],"enforce_handshake_compat":false}}'
