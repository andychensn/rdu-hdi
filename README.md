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
  + ai-dynamo (base package only)            │
                                             │
sambanova/fast-coe's vllm-rdu (NFS venv) ──  RDU decode worker
  + andychensn/ucx + andychensn/nixl         │   (hdi's proven connector/engine,
    (repo-built, rdu-ucx-install/)           │    not the retired standalone
  + ai-dynamo (base package only)            │    andychensn/vllm-rdu fork)
                                             │
etcd + NATS + dynamo.frontend   ──────────  Control plane (login node)
```

Both sides install plain `ai-dynamo`/`ai-dynamo-runtime`, never the `[vllm]` extra — that extra
pulls in vllm 0.20.x as a dependency, which breaks MiniMax-M2.7 (both sides pin vllm 0.16.0
instead, for different reasons — see `config/versions.env`).

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
- SSH key access to GitHub (for cloning the private `sambanova/fast-coe` repo)
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

# 1. Fetch benchmark tooling
git clone https://github.com/SemiAnalysisAI/InferenceX.git "$REPO/InferenceX"
git -C "$REPO/InferenceX" checkout "$INFERENCEX_COMMIT"

# 2. Fetch etcd + nats-server binaries (SHA256-verified)
bash scripts/fetch_vendor.sh

# 3. Build GPU prefill Docker image (~20 min, login node, no GPU required)
bash scripts/build_docker_gpu.sh

# 4. Build the whole RDU environment: fast-coe source, UCX/NIXL from source,
#    the +cpu vllm wheel, and the final .venv_rdu — all in one script, two
#    phases (login node needs internet; RDU-node build takes ~5 min total).
bash scripts/build_rdu_env.sh --fetch-only
snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
    --allow-local-lib-python --reservation "$RDU_RESERVATION" \
    --pef "$PEF" --timeout "$RDU_TIMEOUT" -o logs/build_rdu_env.log \
    -- bash scripts/build_rdu_env.sh --build-only
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

`patches/` is split by which side each override applies to — `patches/gpu/` (used by `Dockerfile.gpu`) and `patches/rdu/` (used by `scripts/build_rdu_env.sh`). Both a full-file overlay (`.py`, copied wholesale) and a unified diff (`.patch`, applied via `patch -p1`) can appear on either side; the extension tells you which.

GPU side (`Dockerfile.gpu`, `patches/gpu/`):
- **vllm nixl_connector patch**: `REGISTER_CONSUMER_MSG` not in stock vllm 0.16.0 — fixed by overlaying `patches/gpu/nixl_connector.py`, a verbatim copy of hdi's real `gpu_vllm_fork` producer file (not a reconstructed patch). Source: `sambanova/sn_vllm`.
- **dynamo.vllm protocol patch**: `ai-dynamo 1.2.1` imports `MultiModalUUIDDict` from vllm (added in 0.20.x). Patched inline in `Dockerfile.gpu` to be conditional (same fix as the RDU side below, implemented independently rather than sharing a file — keep both in sync by hand if this logic ever changes).

RDU side (`scripts/build_rdu_env.sh`, `patches/rdu/`):
- **dynamo.vllm protocol patch**: same `MultiModalUUIDDict` gap as above, applied here via `patches/rdu/dynamo_multimodal_protocol.patch`. The RDU side does *not* need the `REGISTER_CONSUMER_MSG` patch — `VLLM_PD_CHUNK_OVERLAP` is hardcoded to `0` in `launch/rdu_decode.sh` and `launch/gpu_prefill.sh`, so the feature it enables is never used (the RDU-side attempt to apply this patch was a permanent no-op and was deleted outright 2026-07-03, not kept as a "just in case" reference — it was a never-verified hand-reconstruction, the same category of artifact `patches/gpu/nixl_connector.py` replaced above; if this feature is ever revived, find hdi's actual consumer-side implementation instead).
- **RDU torch compat**: s339 has `torch 2.2.0+sn`; vllm 0.16.0 uses torch 2.4+ APIs. Two files patched by `build_rdu_env.sh`, one of them (`patches/rdu/vllm_env_override_torch22x.py`) a full-file replacement of vllm's own `env_override.py`. Long-term fix: RDU Docker image with matching torch.
- **rdma-core devel headers**: s339 ships `libibverbs`/`librdmacm` runtime `.so.1` but not the `-devel` headers, and there's no package-manager access to install them. `build_rdu_env.sh --fetch-only` downloads a pinned, SHA256-verified header subset instead of searching the filesystem for them (a prior version searched all of `/import`, which took hours on this NFS mount).

## Component repos

| Repo | Purpose |
|------|---------|
| [`andychensn/ucx`](https://github.com/andychensn/ucx) | UCX 1.22 + SN RDMA patches (used by both GPU and RDU sides) |
| [`andychensn/nixl`](https://github.com/andychensn/nixl) | NIXL + SN UCX integration |
| `sambanova/fast-coe` | hdi's proven vllm-rdu connector/engine (`server/vllm-rdu`), pinned by commit in `config/versions.env`. Supersedes the retired standalone `andychensn/vllm-rdu` fork — see `docs/local/PARITY_PLAN.md`. |
| [`sambanova/sn_vllm`](https://github.com/sambanova/sn_vllm) | Source of the GPU-side `REGISTER_CONSUMER_MSG` producer file |

## Docker GPU prefill — notes

Docker works. Key non-obvious fixes required:

1. **`--shm-size=1g`**: Docker's default 64MB `/dev/shm` is exhausted by UCX's IB transport when allocating receive descriptor pools (~4MB × 4 TP workers). Without this, UCX fails with `uct_mem.c:482 Assertion mem.memh != UCT_MEM_HANDLE_NULL`.

2. **Broadcom OOT `libbnxt_re`**: Ubuntu's inbox `libbnxt_re-rdmav34.so` sends wrong UVERBS attributes to the host's Broadcom OOT bnxt_re kernel driver (237.1.137.0), causing `EINVAL`. Fixed by building from source: `/import/it-tools/idc/fw/brcm/237/bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/libbnxt_re-237.1.137.0.tar.gz` (shipped with `rc-compat/v39` for Ubuntu 22.04 compatibility).

3. **`--pull=always`**: Without this, GPU nodes use a stale cached image and don't get Dockerfile updates.
