# rdu-hdi — GPU Prefill + RDU Decode via Dynamo

Disaggregated inference on SambaNova hardware: H200 GPU handles prefill,
SN40L RDU handles decode, coordinated by NVIDIA Dynamo.

---

## Architecture

```
vllm/vllm-openai (Docker image)  ─────────  GPU prefill worker
  + andychensn/ucx  (bnxt_re RoCE)          │ NIXL KV cache transfer
  + andychensn/nixl                          │
  + vllm nixl_connector patch                │
  + ai-dynamo[vllm]                          │
                                             │
andychensn/vllm-rdu (NFS venv)  ──────────  RDU decode worker
  + ai-dynamo[vllm]                          │
  + NIXL (pathb/bnxt_re)                     │
                                             │
etcd + NATS + dynamo.frontend   ──────────  Control plane (login node)
```

All external dependencies are pinned to exact commit SHAs in `config/versions.env`.

---

## Configuration

Two config files to edit before use:

**`config/cluster.env`** — cluster topology (nodes, IPs, reservations, Docker image tag)
```bash
GPU_NODE=sc3-c129           # your GPU node
GPU_ROCE_IP=10.17.176.33    # GPU node RoCE IP
RDU_NODE=sc3-s339           # your RDU node
RDU_ROCE_IP=10.17.112.29    # RDU node RoCE IP
...
```

**`config/model.env`** — model/PEF paths and inference settings
```bash
MODEL=/path/to/checkpoints/MyModel
SERVED_MODEL_NAME=MyModel
PEF=/path/to/my-model.pef
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=196608
...
```

---

## Prerequisites

- Login node access (`sc-vnc9` or equivalent) with `snrdu` on PATH
- `gh` CLI authenticated to GitHub (for `gh repo clone`)
- `sudo -g docker` access on GPU nodes (for `cuda-docker-run-wrapper`)
- GPU node reservation + RDU node reservation (see `config/cluster.env`)

## One-time setup

```bash
git clone https://github.com/andychensn/rdu-hdi.git && cd rdu-hdi
REPO=$(pwd)

# Edit config for your cluster + model before continuing
vi config/cluster.env config/model.env

source config/versions.env
source config/cluster.env
source config/model.env

# 1. Clone runtime repos (vllm-rdu plugin, benchmark tooling)
gh repo clone andychensn/vllm-rdu "$REPO/vllm-rdu"
git -C "$REPO/vllm-rdu" checkout "$VLLM_RDU_COMMIT"
git clone https://github.com/SemiAnalysisAI/InferenceX.git "$REPO/InferenceX"
git -C "$REPO/InferenceX" checkout "$INFERENCEX_COMMIT"

# 2. Fetch etcd + nats-server binaries (SHA256-verified)
bash scripts/fetch_vendor.sh

# 3. Build GPU prefill Docker image (~20 min, login node, no GPU required)
bash scripts/build_docker_gpu.sh

# 4. Fetch + build RDU UCX/NIXL (two phases: login node then RDU node)
source config/cluster.env
bash scripts/build_rdu_ucx_nixl.sh --fetch-only
snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
    --allow-local-lib-python --reservation "$RDU_RESERVATION" \
    --pef "$PEF" --timeout 00:30:00 -o logs/build_rdu_ucx_nixl.log \
    -- bash scripts/build_rdu_ucx_nixl.sh --build-only

# 5. Build RDU venv (~10 min on RDU node)
snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
    --allow-local-lib-python --reservation "$RDU_RESERVATION" \
    --pef "$PEF" --timeout 00:30:00 -o logs/build_rdu_venv.log \
    -- bash scripts/build_rdu_venv.sh
```

---

## Per-session launch

```bash
# 1. Control plane (auto-creates .venv_cp on first run)
bash launch/control_plane.sh

# 2. GPU prefill (~10 min: model load + warmup)
bash launch/gpu_prefill.sh

# 3. RDU decode — waits for GPU registration, then ~12 min
source config/cluster.env && bash launch/rdu_decode.sh

# 4. Warmup (first request ~47s, cold NIXL init)
curl -s http://localhost:18000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"hello\",\"max_tokens\":1}"
```

> Do not start RDU decode before GPU prefill — Dynamo builds the wrong pipeline.

## Benchmark

```bash
bash scripts/benchmark.sh --input-len 1000 --output-len 1000 --concurrency 1
```

Results saved to `benchmark_results/` (gitignored).

## Teardown

```bash
bash launch/control_plane.sh --stop
scancel $(squeue -u $USER -h -o '%i')
```

---

## Docker

Two wrappers on the cluster:

| Wrapper | Where | Supports |
|---------|-------|---------|
| `/usr/bin/docker-wrapper` | Login node | build, push, pull, ps, … |
| `/usr/bin/cuda-docker-run-wrapper` | GPU nodes | run (GPU passthrough + `/import` mount) |

Both require `sudo -g docker`. Internal registry: `sc-artifacts2.sambanovasystems.com/sw-docker-scratch/`.
`--net=host` required when running on GPU nodes (for RoCE RDMA).

---

## Known gaps

- **vllm nixl_connector patch**: `REGISTER_CONSUMER_MSG` not in stock vllm 0.16.0 — applied in `Dockerfile.gpu`. Source: `sambanova/sn_vllm`.
- **dynamo.vllm protocol patch**: `ai-dynamo 1.2.1` imports `MultiModalUUIDDict` from vllm (added in 0.20.x). Patched in `Dockerfile.gpu` to be conditional.
- **RDU torch compat**: s339 has `torch 2.2.0+sn`; vllm 0.16.0 uses torch 2.4+ APIs. Two files patched by `build_rdu_venv.sh`. Long-term fix: RDU Docker image with matching torch.

## Component repos

| Repo | Purpose |
|------|---------|
| [`andychensn/ucx`](https://github.com/andychensn/ucx) | UCX 1.22 + SN RDMA patches |
| [`andychensn/nixl`](https://github.com/andychensn/nixl) | NIXL + SN UCX integration |
| [`andychensn/vllm-rdu`](https://github.com/andychensn/vllm-rdu) | vLLM plugin for SambaNova RDU |
| [`sambanova/sn_vllm`](https://github.com/sambanova/sn_vllm) | Source of `patches/vllm_nixl_connector.patch` |

## Docker GPU prefill — notes

Docker works. Key non-obvious fixes required:

1. **`--shm-size=1g`**: Docker's default 64MB `/dev/shm` is exhausted by UCX's IB transport when allocating receive descriptor pools (~4MB × 4 TP workers). Without this, UCX fails with `uct_mem.c:482 Assertion mem.memh != UCT_MEM_HANDLE_NULL`.

2. **Broadcom OOT `libbnxt_re`**: Ubuntu's inbox `libbnxt_re-rdmav34.so` sends wrong UVERBS attributes to the host's Broadcom OOT bnxt_re kernel driver (237.1.137.0), causing `EINVAL`. Fixed by building from source: `/import/it-tools/idc/fw/brcm/237/bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/libbnxt_re-237.1.137.0.tar.gz` (shipped with `rc-compat/v39` for Ubuntu 22.04 compatibility).

3. **`--pull=always`**: Without this, GPU nodes use a stale cached image and don't get Dockerfile updates.
