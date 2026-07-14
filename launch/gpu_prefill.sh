#!/usr/bin/env bash
# GPU prefill worker(s) — submits one SLURM job per configured worker, waits for
# each to register with Dynamo. Runs via Docker (vllm/vllm-openai base +
# UCX/NIXL/patch baked in).
# Build the image first: bash docker/gpu/build.sh
#
# Usage: bash launch/gpu_prefill.sh
#   Number/placement of workers is controlled by GPU_NODES/GPU_RESERVATIONS in
#   config/cluster.env — one entry per worker, index-paired (same node repeated
#   colocates workers on that node; empty-string reservation omits --reservation
#   for that worker).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
source "$REPO_ROOT/config/versions.env"
source "$REPO_ROOT/config/cluster.env"
source "$REPO_ROOT/config/model.env"

GPU_CACHE_ROOT="$REPO_ROOT/.gpu_cache"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR" "$GPU_CACHE_ROOT"
TS=$(date +%Y%m%d_%H%M%S)

# LMCache (CPU-tier KV-cache offload+reuse, GPU-prefill-only) composed with
# NixlConnector via vLLM's native MultiConnector -- MultiConnector loads from
# the first connector (in list order) that advertises cached tokens and saves
# to all connectors, so LMCache is tried first, falling through to NIXL's
# cross-node producer/consumer path for the actual GPU->RDU KV handoff.
# NixlConnector's own extra_config is unchanged from the single-connector
# setup this replaces.
KV_CONFIG='{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"backends":["UCX"]}}]}}'

# ── Inner: runs ON the GPU node ───────────────────────────────────────────────
if [[ "${1:-}" == "--inner" ]]; then
    IDX="${2:?worker index required}"
    LOCAL_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep '^10\.17\.' | head -1 || true)
    LOCAL_IP=${LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
    # Distinct NIXL side-channel port per worker: vLLM defaults every worker to
    # the same fixed port (GPU_NIXL_BASE_PORT) when data_parallel_index=0, which
    # collides if two workers share a node under --net=host.
    NIXL_PORT=$((GPU_NIXL_BASE_PORT + IDX))
    # Same collision pattern for the KV-events ZMQ publisher (feeds
    # --router-mode kv's cache-aware routing on the frontend): vLLM's
    # KVEventsConfig defaults its publisher to a single hardcoded
    # tcp://*:5557 -- two same-node workers under --net=host both try to
    # bind it and the second one's engine core dies with
    # zmq.error.ZMQError: Address already in use. Give each worker its own
    # port, same fix shape as NIXL_PORT.
    KV_EVENTS_PORT=$((GPU_KV_EVENTS_BASE_PORT + IDX))
    KV_EVENTS_CONFIG="{\"enable_kv_cache_events\": true, \"endpoint\": \"tcp://*:${KV_EVENTS_PORT}\"}"

    # LMCache config, read from the environment by lmcache's own config
    # loader -- chunk_size must be an exact multiple of BLOCK_SIZE (LMCache
    # raises ValueError otherwise), derived here instead of hardcoded so it
    # can't drift out of sync if BLOCK_SIZE ever changes.
    LMCACHE_CHUNK_SIZE=$((BLOCK_SIZE * 4))

    # Optional local-disk tier (see config/model.env's LMCACHE_MAX_LOCAL_DISK_GB
    # comment) -- on by default; set LMCACHE_MAX_LOCAL_DISK_GB=0 to disable.
    # Mounted under GPU_CACHE_ROOT (NFS, already auto-mounted into the
    # container), so it survives a worker restart even though the CPU tier
    # (pure in-process memory) does not.
    # Scoped per worker index (LMCACHE_DISK_PATH, not a shared directory):
    # the on-disk cache key filename encodes only (model, world_size,
    # TP-rank, chunk_hash), which repeats identically across independent
    # worker processes -- a shared path relies on Python's own randomized
    # hash() (LMCache's default "builtin" chunk-hash algorithm) to avoid
    # cross-worker collisions, which is incidental, not structural
    # (PYTHONHASHSEED is unset anywhere in this repo).
    LMCACHE_DISK_FLAGS=()
    if [[ "${LMCACHE_MAX_LOCAL_DISK_GB:-0}" != "0" ]]; then
        LMCACHE_DISK_PATH="$GPU_CACHE_ROOT/lmcache_disk/worker$IDX"
        mkdir -p "$LMCACHE_DISK_PATH"
        LMCACHE_DISK_FLAGS=(
            -e "LMCACHE_LOCAL_DISK=$LMCACHE_DISK_PATH"
            -e "LMCACHE_MAX_LOCAL_DISK_SIZE=$LMCACHE_MAX_LOCAL_DISK_GB"
        )
        echo "    LMCache disk tier ENABLED: $LMCACHE_DISK_PATH, ${LMCACHE_MAX_LOCAL_DISK_GB}GB -- not yet sized against a real workload, see config/model.env"
    fi

    # Optional: shrink vLLM's own native GPU KV cache for this worker.
    # Not used in normal operation (the whole point of a large GPU cache is
    # to serve as much as possible without ever needing LMCache's CPU
    # tier) -- exists so test/e2e_lmcache_correctness.py can force reliable
    # eviction. Without this, that test's filler traffic competes against
    # this deployment's real ~98K-token-per-worker native capacity, and
    # empirically almost never wins -- vLLM's own cache silently keeps
    # serving replays natively instead of the CPU-tier reload the test
    # means to exercise. block_size=64, so e.g. GPU_NUM_BLOCKS_OVERRIDE=400
    # caps this worker at 400*64=25,600 tokens of native GPU KV capacity.
    NUM_GPU_BLOCKS_FLAG=""
    if [[ -n "${GPU_NUM_BLOCKS_OVERRIDE:-}" ]]; then
        echo "    GPU_NUM_BLOCKS_OVERRIDE set: capping native GPU KV cache at $GPU_NUM_BLOCKS_OVERRIDE blocks ($((GPU_NUM_BLOCKS_OVERRIDE * BLOCK_SIZE)) tokens) -- NOT for normal operation."
        NUM_GPU_BLOCKS_FLAG="--num-gpu-blocks-override $GPU_NUM_BLOCKS_OVERRIDE"
    fi

    # Optional: disable vLLM's own native prefix caching entirely for this
    # worker. Not used in normal operation (native prefix caching is a real,
    # wanted feature) -- exists so test/e2e_lmcache_correctness.py can
    # isolate the LMCache CPU-tier path cleanly. Shrinking the GPU cache via
    # GPU_NUM_BLOCKS_OVERRIDE alone was NOT sufficient in testing to force
    # eviction -- a 42x smaller pool produced identical "Inference Engine
    # computed tokens" values to the full-size pool, an unexplained
    # native-cache interaction worth isolating rather than fighting.
    PREFIX_CACHE_FLAG="--enable-prefix-caching"
    if [[ "${GPU_DISABLE_NATIVE_PREFIX_CACHE:-0}" == "1" ]]; then
        echo "    GPU_DISABLE_NATIVE_PREFIX_CACHE=1: vLLM's own native prefix cache is OFF -- NOT for normal operation."
        PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
    fi

    echo "=== GPU prefill worker $IDX (Docker) on $(hostname) ==="
    echo "    image:      $GPU_IMAGE"
    echo "    RoCE IP:    $LOCAL_IP"
    echo "    NIXL port:  $NIXL_PORT"
    echo "    LMCache:    chunk_size=$LMCACHE_CHUNK_SIZE max_local_cpu=${LMCACHE_MAX_LOCAL_CPU_GB}GB"

    # Pre-flight: refuse to start if this job's allocated GPUs already show
    # non-trivial memory in use. The Docker container name below is unique
    # per SLURM job (so a stuck container from a prior failed/torn-down
    # launch can no longer block a retry with a "name already in use"
    # error) -- but that also means a leaked process still holding these
    # same physical GPUs would otherwise go unnoticed until this worker's
    # own engine fails with a confusing CUDA OOM/bind error deep in its own
    # startup log, or silently contends for memory instead of failing at
    # all. This check restores a fast, clear failure with an actionable
    # next step, scoped to nvidia-smi (no docker-wrapper/sudo needed).
    STALE_MEM_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{sum+=$1} END {print sum+0}')
    if [[ "${STALE_MEM_MIB:-0}" -gt 2048 ]]; then
        echo "ERROR: worker $IDX's allocated GPUs already show ${STALE_MEM_MIB} MiB in use before" >&2
        echo "this launch even starts -- likely a leaked process from a prior failed/torn-down" >&2
        echo "job (or another job sharing this reservation). Refusing to start a new engine that" >&2
        echo "would silently contend for the same GPUs. Run on this node:" >&2
        echo "  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv" >&2
        echo "to identify and (if it's your own leaked process) kill it before retrying." >&2
        exit 1
    fi

    # Mount RDMA/IB devices so UCX can register GPU memory for RoCE NIXL transfer
    RDMA_DEVICES=""
    for dev in /dev/infiniband /dev/uverbs* /dev/nvidia-uvm /dev/nvidiactl; do
        [ -e "$dev" ] && RDMA_DEVICES="$RDMA_DEVICES --device $dev"
    done
    # Background GPU telemetry sampler — a plain child process of this SAME
    # SLURM step (not a new srun/step), so it never touches --overlap's
    # broken step-creation path on this cluster (srun --overlap hangs
    # indefinitely on jobs we own, and is denied outright for jobs we don't
    # -- confirmed empirically, not fixable from this repo). exec below only
    # replaces this process's own image; already-forked background children
    # are unaffected and keep running until the job's cgroup is torn down.
    SMI_LOG="$LOG_DIR/gpu_telemetry_${IDX}_job${SLURM_JOB_ID}.csv"
    # GPU_TELEMETRY_FULL=1 captures nvidia-smi's throttle-reason bitmask fields
    # (hw/sw thermal slowdown, power cap, etc.) plus fan/PCIe/memory-temp/clock
    # detail, for directly distinguishing WHY the driver is throttling rather
    # than just observing temp+clock correlate. Default stays the lean
    # 6-field query to avoid bloating routine runs.
    if [[ "${GPU_TELEMETRY_FULL:-0}" == "1" ]]; then
        SMI_FIELDS="timestamp,index,utilization.gpu,utilization.memory,power.draw,power.limit,enforced.power.limit,clocks.current.sm,clocks.current.memory,clocks.max.sm,temperature.gpu,temperature.gpu.tlimit,temperature.memory,fan.speed,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sync_boost,clocks_throttle_reasons.gpu_idle"
    else
        SMI_FIELDS="timestamp,index,utilization.gpu,power.draw,clocks.sm,temperature.gpu"
    fi
    nvidia-smi --query-gpu="$SMI_FIELDS" \
        --format=csv,noheader -lms 100 > "$SMI_LOG" 2>&1 &
    disown $!
    echo "    GPU telemetry: $SMI_LOG (full=${GPU_TELEMETRY_FULL:-0})"

    # Optional explicit physical-GPU-combo override, for testing arbitrary (not
    # just contiguous) GPU sets with a single TP-N worker -- e.g.
    # GPU_EXPLICIT_DEVICES=1,4,5,7 to mix a specific GPU into an otherwise
    # different set (e.g. isolating whether a single misbehaving GPU
    # dominates a group's performance regardless of which others it's
    # grouped with). Requires requesting the WHOLE node (GPU_GRES=gpu:8) so
    # every physical device is in this job's cgroup device whitelist --
    # cuda-docker-run-wrapper does nothing but forward whatever
    # $CUDA_VISIBLE_DEVICES already is as -e NVIDIA_VISIBLE_DEVICES
    # (confirmed by reading /usr/bin/cuda-docker-run-wrapper directly), so
    # overriding the env var here, before exec, is sufficient -- no wrapper
    # changes needed. Only meaningful with exactly one worker per node.
    if [[ -n "${GPU_EXPLICIT_DEVICES:-}" ]]; then
        echo "    Explicit GPU combo override: CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES -> $GPU_EXPLICIT_DEVICES"
        export CUDA_VISIBLE_DEVICES="$GPU_EXPLICIT_DEVICES"
    fi

    # Optional CPU/memory NUMA-affinity override, originally added to test
    # NUMA affinity as a cause of a per-request-compute-time growth issue --
    # SUPERSEDED: that hypothesis was ruled out (root cause turned out to be
    # a single GPU per node thermally throttling under sustained load, not
    # NUMA affinity), kept only because the override mechanism itself
    # (deliberately mismatching CPU-NUMA from GPU-NUMA) is reusable for
    # future causal tests. SLURM assigns physical GPUs per-launch (not fixed
    # per worker index), so the override is computed from the ACTUAL
    # $CUDA_VISIBLE_DEVICES this job got, not from a static worker-index
    # table. sc3-c129 topology: GPUs{0,1}->NUMA0, {2,3}->NUMA1,
    # {4,5}->NUMA2, {6,7}->NUMA3.
    NUMACTL_PREFIX=()
    if [[ "${GPU_NUMA_SWAP_TEST:-0}" == "1" ]]; then
        FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
        case "$FIRST_GPU" in
            0|1|2|3) FORCE_NUMA=3 ;;
            4|5|6|7) FORCE_NUMA=2 ;;
            *) echo "WARNING: unrecognized CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, skipping NUMA override"; FORCE_NUMA="" ;;
        esac
        if [[ -n "$FORCE_NUMA" ]]; then
            echo "    NUMA swap test: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> forcing cpunodebind=membind=$FORCE_NUMA"
            NUMACTL_PREFIX=(numactl "--cpunodebind=$FORCE_NUMA" "--membind=$FORCE_NUMA")
        fi
    fi

    exec "${NUMACTL_PREFIX[@]}" sudo -g docker /usr/bin/cuda-docker-run-wrapper \
        --pull=always \
        --net=host --rm \
        --name "gpu-prefill-${IDX}-${SLURM_JOB_ID}" \
        --entrypoint python3 \
        --ulimit memlock=-1 \
        $RDMA_DEVICES \
        -e "ETCD_ENDPOINTS=http://$CONTROL_PLANE_IP:$ETCD_PORT" \
        -e "NATS_SERVER=nats://$CONTROL_PLANE_IP:$NATS_PORT" \
        -e "VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP" \
        -e "VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT" \
        -e "NCCL_IB_DISABLE=1" \
        -e "NCCL_P2P_LEVEL=NVL" \
        --shm-size=1g \
        -e "UCX_MODULE_DIR=/opt/ucx/lib/ucx" \
        -e "UCX_TLS=rc,cuda_copy,cuda_ipc" \
        -e "UCX_NET_DEVICES=bnxt_re0:1" \
        -e "UCX_IB_ROCE_REACHABILITY_MODE=all" \
        -e "LMCACHE_LOCAL_CPU=True" \
        -e "LMCACHE_MAX_LOCAL_CPU_SIZE=$LMCACHE_MAX_LOCAL_CPU_GB" \
        -e "LMCACHE_CHUNK_SIZE=$LMCACHE_CHUNK_SIZE" \
        -e "LMCACHE_LOG_LEVEL=${LMCACHE_LOG_LEVEL:-INFO}" \
        "${LMCACHE_DISK_FLAGS[@]}" \
        -e "HF_HOME=$GPU_CACHE_ROOT/huggingface" \
        -e "VLLM_CACHE_ROOT=$GPU_CACHE_ROOT/vllm" \
        -e "TRITON_CACHE_DIR=$GPU_CACHE_ROOT/triton" \
        -e "TORCHINDUCTOR_CACHE_DIR=$GPU_CACHE_ROOT/inductor" \
        -e "FLASHINFER_WORKSPACE_BASE=$GPU_CACHE_ROOT/flashinfer" \
        -e "VLLM_CONFIG_ROOT=$GPU_CACHE_ROOT/vllm_config" \
        "$GPU_IMAGE" \
        -m dynamo.vllm \
            --model "$MODEL" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --disaggregation-mode prefill \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --block-size "$BLOCK_SIZE" \
            $PREFIX_CACHE_FLAG \
            --reasoning-parser minimax_m2_append_think \
            --trust-remote-code \
            --kv-transfer-config "$KV_CONFIG" \
            --kv-events-config "$KV_EVENTS_CONFIG" \
            $NUM_GPU_BLOCKS_FLAG
fi

# ── Outer: submit one SLURM job per configured worker, wait for all to register ──
NUM_WORKERS=${#GPU_NODES[@]}
echo "Submitting $NUM_WORKERS GPU prefill worker(s): ${GPU_NODES[*]}"

SRUN_PIDS=()
GPU_LOGS=()
JOB_NAMES=()

for i in "${!GPU_NODES[@]}"; do
    node="${GPU_NODES[$i]}"
    reservation="${GPU_RESERVATIONS[$i]:-}"
    GPU_LOG="$LOG_DIR/${TS}_gpu_prefill_${i}.log"
    GPU_LOGS+=("$GPU_LOG")
    JOB_NAME="gpu-prefill-${i}-${TS}"
    JOB_NAMES+=("$JOB_NAME")
    echo "  worker $i: node=$node reservation=${reservation:-<none>} log=$GPU_LOG job-name=$JOB_NAME"
    srun \
        -p "$GPU_PARTITION" -w "$node" \
        --gres="$GPU_GRES" \
        --cpus-per-task="${GPU_CPUS:-2}" \
        --mem="$GPU_MEM" \
        --job-name="$JOB_NAME" \
        ${reservation:+--reservation "$reservation"} \
        -t "$GPU_TIME" \
        bash "$(readlink -f "${BASH_SOURCE[0]}")" --inner "$i" \
        > "$GPU_LOG" 2>&1 &
    SRUN_PIDS+=("$!")
done
echo "  srun PIDs: ${SRUN_PIDS[*]}"

# Fail-fast teardown. scancel-by-job-name is the primary mechanism (targets
# exactly this launch's jobs, not scancel-everything-for-the-user); killing
# the local srun client PID is a best-effort fallback on top. scancel has
# NOT always reliably stopped a worker's foreground Docker container in
# this repo's history -- this needs empirical confirmation on any given
# cluster (check `docker ps`/`nvidia-smi` on the node after a real
# teardown), and may need a container-level kill mechanism added if it
# doesn't hold up.
teardown_all() {
    echo "  Tearing down all worker(s) from this launch..."
    for job_name in "${JOB_NAMES[@]}"; do
        echo "    scancel --name=$job_name"
        scancel --name="$job_name" 2>/dev/null || true
    done
    for pid in "${SRUN_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
}

REGISTERED=()
for i in "${!GPU_NODES[@]}"; do REGISTERED[$i]=0; done

echo "  Waiting for all $NUM_WORKERS worker(s) to register (up to ${GPU_PREFILL_REGISTER_TIMEOUT}s each)..."
FAILED=0
for t in $(seq 1 "$GPU_PREFILL_REGISTER_TIMEOUT"); do
    all_done=1
    for i in "${!GPU_NODES[@]}"; do
        if [[ "${REGISTERED[$i]}" -eq 1 ]]; then
            continue
        fi
        if grep -q "Registered base model '$SERVED_MODEL_NAME' MDC" "${GPU_LOGS[$i]}" 2>/dev/null; then
            echo "  worker $i registered (${t}s)"
            REGISTERED[$i]=1
            continue
        fi
        if ! kill -0 "${SRUN_PIDS[$i]}" 2>/dev/null; then
            echo "ERROR: worker $i's srun exited early before registering. Log:"
            tail -30 "${GPU_LOGS[$i]}"
            FAILED=1
        fi
        all_done=0
    done
    if [[ "$FAILED" -eq 1 ]]; then
        break
    fi
    if [[ "$all_done" -eq 1 ]]; then
        break
    fi
    if [[ $((t % 30)) -eq 0 ]]; then
        reg_count=0
        for r in "${REGISTERED[@]}"; do
            if [[ "$r" -eq 1 ]]; then
                reg_count=$((reg_count + 1))
            fi
        done
        echo "  ${t}s... ($reg_count of $NUM_WORKERS registered)"
    fi
    sleep 1
done

if [[ "$FAILED" -eq 1 ]]; then
    teardown_all
    exit 1
fi

for i in "${!GPU_NODES[@]}"; do
    if [[ "${REGISTERED[$i]}" -ne 1 ]]; then
        echo "ERROR: worker $i did not register within ${GPU_PREFILL_REGISTER_TIMEOUT}s. Log:"
        tail -20 "${GPU_LOGS[$i]}"
        teardown_all
        exit 1
    fi
done

echo "  All $NUM_WORKERS GPU prefill worker(s) registered:"
for i in "${!GPU_LOGS[@]}"; do
    echo "    worker $i: ${GPU_LOGS[$i]}"
done
