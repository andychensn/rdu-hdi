#!/usr/bin/env bash
# Inner RDU decode script — runs ON s339 inside the snrdu job.
# Do not call directly; submitted via rdu_decode.sh through snrdu.
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

RDU_VENV="${RDU_VENV:-$REPO_ROOT/.venv_rdu}"
RDU_CACHE=${RDU_CACHE:-$REPO_ROOT/.rdu_cache}
mkdir -p "$RDU_CACHE"

# Hardware paths — BAR2/SambaFlow are SambaNova system deps (like CUDA, set by snrdu env)
SOFTWARE_BUILD=${SOFTWARE_BUILD:-/import/snvm-sc-scratch1/guoyaof/software/runtime/build}
BAR2_INSTALL=${BAR2_INSTALL:-/import/snvm-sc-scratch2/jayr/sambaflow_gTkgyGCEBH/bazel-install}
BAR2_RUNTIME_LIBS=${BAR2_RUNTIME_LIBS:-/import/snvm-sc-scratch1/guoyaof/sw_ddr_rdma/runtime/build/graph/lib}
BAR2_PRELOAD=${BAR2_PRELOAD:-/import/snvm-sc-scratch1/guoyaof/sw_ddr_rdma_install/bar2_preload_libs}

# UCX/NIXL: prefer repo-built (from build_rdu_ucx_nixl.sh) over SOFTWARE_BUILD fallback
RDU_UCX="${RDU_UCX:-$REPO_ROOT/rdu-ucx-install}"
if [ -d "$RDU_UCX/lib" ]; then
    _UCX_LIB="$RDU_UCX/lib"
    _UCX_MOD="$RDU_UCX/lib/ucx"
    _NIXL_LIB="$RDU_UCX/../wheelhouse"  # nixl .so bundled in wheel; no separate install
else
    # Fallback to SOFTWARE_BUILD until build_rdu_ucx_nixl.sh has been run
    _UCX_LIB="$SOFTWARE_BUILD/ucx-install/lib"
    _UCX_MOD="$SOFTWARE_BUILD/ucx-install/lib/ucx"
    _NIXL_LIB="$SOFTWARE_BUILD/nixl-install/lib"
fi

# Detect RDU's own RoCE IP
RDU_ROCE_IP_LOCAL=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep '^10\.17\.' | head -1 || true)
RDU_ROCE_IP_LOCAL=${RDU_ROCE_IP_LOCAL:-$(hostname -I 2>/dev/null | awk '{print $1}')}

echo "=== RDU decode on $(hostname) $(date) ==="
echo "    etcd: http://$CONTROL_PLANE_IP:$ETCD_PORT"
echo "    producer: $GPU_ROCE_IP:5600"
echo "    side-channel: $RDU_ROCE_IP_LOCAL:5600"

source "$RDU_VENV/bin/activate"

# RDU model config — set MODEL_CONFIG in config/model.env for your model
RDU_CONFIG="${MODEL_CONFIG:-}"
[ -n "$RDU_CONFIG" ] && [ ! -f "$RDU_CONFIG" ] && { echo "WARNING: MODEL_CONFIG=$RDU_CONFIG not found"; RDU_CONFIG=""; }

PYTHONNOUSERSITE=1 \
ETCD_ENDPOINTS="http://$CONTROL_PLANE_IP:$ETCD_PORT" \
NATS_SERVER="nats://$CONTROL_PLANE_IP:$NATS_PORT" \
DYN_REQUEST_PLANE=tcp \
HF_HUB_OFFLINE=1 \
VLLM_NIXL_SIDE_CHANNEL_HOST="$RDU_ROCE_IP_LOCAL" \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
VLLM_PD_CHUNK_OVERLAP=1 \
VLLM_PD_PRODUCER_CHUNK_OVERLAP=1 \
VLLM_PD_PRODUCER_HOST="$GPU_ROCE_IP" \
VLLM_PD_PRODUCER_PORT=5600 \
SN_REMOTE_KV_MEMTYPE=VRAM \
RAW_FORCE_PER_XFER=1 \
SN_MULTI_AGENT_NICS=bnxt_re0,bnxt_re2,bnxt_re4,bnxt_re6 \
VLLM_CPU_KVCACHE_SPACE=200 \
VLLM_CPU_OMP_THREADS_BIND=nobind \
UCX_TLS=rc,tcp \
UCX_NET_DEVICES=bnxt_re0:1,bnxt_re2:1,bnxt_re4:1,bnxt_re6:1 \
UCX_MAX_RNDV_RAILS=4 \
UCX_MAX_RMA_RAILS=1 \
UCX_MULTI_LANE_MAX_RATIO=100 \
UCX_IB_PREFER_NEAREST_DEVICE=n \
UCX_IB_ROCE_REACHABILITY_MODE=all \
UCX_RCACHE_MAX_UNRELEASED=1024 \
UCX_LOG_LEVEL=WARN \
UCX_MODULE_DIR="$_UCX_MOD" \
NIXL_PLUGIN_DIR="$_NIXL_LIB" \
INFERENCE_MODE=local_queue \
PROG_LOAD=HBM \
ARG_LOAD=HBM \
ENABLE_STRICT_CONVERSION=1 \
SF_RNT_FSM_POLL_BUSY_WAIT=1 \
SF_RNT_DMA_POLL_BUSY_WAIT=1 \
SF_RNT_NUMA_BIND=2 \
SF_RNT_LOG_LEVEL=ERR \
AL_EXEC_LOOPING=1 \
GRAPH_TIMEOUT_USEC=120000000 \
TRITON_CACHE_DIR="$RDU_CACHE/triton" \
VLLM_CACHE_ROOT="$RDU_CACHE/vllm" \
HF_HOME="$RDU_CACHE/huggingface" \
VLLM_CONFIG_ROOT="$RDU_CACHE/vllm_config" \
TRANSFORMERS_CACHE="$RDU_CACHE/huggingface" \
PYTHONPATH="$BAR2_INSTALL/python:${PYTHONPATH:-}" \
LD_LIBRARY_PATH="$_UCX_LIB:$_NIXL_LIB:$SOFTWARE_BUILD/etcd-cpp-api-install/lib:$SOFTWARE_BUILD/gflags-install/lib:$BAR2_RUNTIME_LIBS:$BAR2_INSTALL/lib:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$BAR2_PRELOAD/libc_samba_runtime.so:$BAR2_PRELOAD/libcpp_samba_runtime.so${LD_PRELOAD:+:$LD_PRELOAD}" \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --disaggregation-mode decode \
    --load-format dummy \
    --max-num-seqs 2 \
    --tensor-parallel-size 1 \
    --no-enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" \
    --compilation-config '{"mode": 0}' \
    ${RDU_CONFIG:+--additional-config "{\"rdu_config\": \"$RDU_CONFIG\"}"} \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"rdu","kv_connector_extra_config":{"rdu_mode":"real","rdu_ddr_cache_budget_gb":30,"backends":["UCX"],"enforce_handshake_compat":false}}'
