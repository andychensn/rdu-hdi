# rdu-hdi Quick Start

End-to-end setup for a new team member: GPU prefill on H200 + RDU decode on SN40L via Dynamo.

**Estimated time:** ~40 min one-time setup · ~25 min to launch per session.

---

## Prerequisites

| Requirement | Notes |
|------------|-------|
| sc-vnc9 login node access | Run all scripts from here |
| sc3-c129 GPU reservation | `RDU-prefill-experiment-test` |
| sc3-s339 RDU reservation | `no_sf_catchup_demos` (max 01:50:00) |
| `snrdu` on PATH | SambaNova job scheduler — pre-installed on sc-vnc9 |
| PEF file path | Set `PEF=` in `config/cluster.env` |
| BAR2 runtime on s339 | SambaNova hardware SDK — set by snrdu env. Paths in `rdu_inner.sh` (`BAR2_INSTALL`, `BAR2_PRELOAD`, `SOFTWARE_BUILD`) default to guoyaof/jayr NFS; ask your manager for access. |

---

## One-time setup

### 0. Clone the repo

```bash
git clone https://github.com/andychensn/rdu-hdi.git
cd rdu-hdi
REPO=$(pwd)
```

### 1. Clone runtime dependencies

```bash
source "$REPO/config/versions.env"

# vllm-rdu hardware plugin
gh repo clone andychensn/vllm-rdu "$REPO/vllm-rdu"
git -C "$REPO/vllm-rdu" checkout "$VLLM_RDU_COMMIT"

# InferenceX benchmark tooling
git clone https://github.com/SemiAnalysisAI/InferenceX.git "$REPO/InferenceX"
git -C "$REPO/InferenceX" checkout "$INFERENCEX_COMMIT"
```

### 2. Fetch infrastructure binaries (~1 min)

Downloads etcd and nats-server from GitHub releases and verifies SHA256:

```bash
bash "$REPO/scripts/fetch_vendor.sh"
```

### 3. Build GPU prefill Docker image (~20 min, login node only)

Builds `Dockerfile.gpu` on top of `vllm/vllm-openai:v0.16.0`, adding UCX, NIXL,
the vllm patch, and ai-dynamo. **No GPU node required.**

```bash
bash "$REPO/scripts/build_docker_gpu.sh"
```

Pushes to `sc-artifacts2.sambanovasystems.com/sw-docker-scratch/rdu-hdi-gpu-prefill:$GPU_IMAGE_TAG`.
The tag is set in `config/cluster.env`. Increment `GPU_IMAGE_TAG` when UCX/NIXL commits or the vllm patch changes.

### 4. Build RDU UCX + NIXL + wheels (~15 min)

s339 has no internet — fetch everything from the login node first.

```bash
# Phase 1: fetch sources + wheels (login node, ~3 min)
bash "$REPO/scripts/build_rdu_ucx_nixl.sh" --fetch-only

# Phase 2: compile UCX + NIXL on s339 (~13 min)
source "$REPO/config/cluster.env"
snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
    --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:30:00 \
    -o "$REPO/logs/build_rdu_ucx_nixl.log" \
    -- bash "$REPO/scripts/build_rdu_ucx_nixl.sh" --build-only

tail -f "$REPO/logs/build_rdu_ucx_nixl.log"
```

### 5. Build RDU venv (~10 min on sc3-s339)

```bash
snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
    --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:30:00 \
    -o "$REPO/logs/build_rdu_venv.log" \
    -- bash "$REPO/scripts/build_rdu_venv.sh"

tail -f "$REPO/logs/build_rdu_venv.log"
```

### Validate (optional but recommended)

```bash
# GPU image — smoke test on any GPU node
srun -p gpuonly -w sc3-c128 --gres=gpu:1 -c 1 --mem=4096 -t 00:05:00 \
    bash -c 'sudo -g docker /usr/bin/cuda-docker-run-wrapper --net=host --rm \
        '"$GPU_IMAGE"' python3 -c "
import vllm; print(\"vllm:\", vllm.__version__)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import REGISTER_CONSUMER_MSG
print(\"vllm patch: OK\")
from nixl._api import nixl_agent; print(\"nixl: OK\")
import dynamo.vllm; print(\"dynamo.vllm: OK\")
"'

# RDU venv
snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
    --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:05:00 \
    -o "$REPO/logs/test_rdu_imports.log" \
    -- bash "$REPO/scripts/test_rdu_imports.sh"
```

---

## Per-session launch

```bash
cd "$REPO"

# 1. Control plane — auto-creates .venv_cp on first run
bash launch/control_plane.sh

# 2. GPU prefill — blocks ~10 min (model load + warmup)
bash launch/gpu_prefill.sh

# 3. RDU decode — waits for GPU registration, then blocks ~12 min
source config/cluster.env && bash launch/rdu_decode.sh

# 4. Warmup — first request ~47s (NIXL init), all subsequent are fast
curl -s http://10.10.0.156:18000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiniMax-M2.7","prompt":"hello","max_tokens":1}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('warmup OK:', d['usage'])"
```

> `rdu_decode.sh` gates on GPU prefill registration — do not start RDU first.

## Benchmark

```bash
bash scripts/benchmark.sh --input-len 1000  --output-len 1000 --concurrency 1 --num-prompts 10
bash scripts/benchmark.sh --input-len 10000 --output-len 1000 --concurrency 2 --num-prompts 20
```

Results saved to `benchmark_results/` (gitignored).

## Teardown

```bash
bash launch/control_plane.sh --stop
scancel $(squeue -u $USER -h -o '%i')
```

---

## External dependencies

| Component | Pin | Public? |
|-----------|-----|---------|
| GPU Docker image | `vllm/vllm-openai:0.16.0` base | ✅ |
| UCX 1.22 | `andychensn/ucx@e153f2e4` | ✅ |
| NIXL | `andychensn/nixl@c2abc770` | ✅ |
| vllm-rdu | `andychensn/vllm-rdu@5bc4a563` | ✅ |
| ai-dynamo + runtime | PyPI `1.2.1` | ✅ |
| vllm nixl_connector patch | `patches/vllm_nixl_connector.patch` (baked into Docker image) | ✅ |
| benchmark tooling | `SemiAnalysisAI/InferenceX@37505e11` | ✅ |
| etcd 3.5.15 | GitHub releases (SHA256-verified) | ✅ |
| nats-server 2.10.28 | GitHub releases (SHA256-verified) | ✅ |
| nixl-pathb wheel | Built from `andychensn/nixl@sn/rdu-working` by Step 4 | ✅ |
| BAR2 runtime (s339) | guoyaof/jayr NFS paths | ⚠️ internal |
| Model weights | `/import/ml-sc-scratch6/yund/...` | ⚠️ internal NFS |
| PEF file | `/import/ml-sc-scratch4/jayr/...` | ⚠️ internal NFS |

All version numbers and commit SHAs are in `config/versions.env`.

---

## Docker on the cluster

| Wrapper | Where | Supports |
|---------|-------|---------|
| `/usr/bin/docker-wrapper` | sc-vnc9 (login node) | build, push, pull, ps, images, tag, … |
| `/usr/bin/cuda-docker-run-wrapper` | GPU nodes (c127/c128/c129) | run only (GPU passthrough + /import) |

Both require `sudo -g docker`. Internal registry: `sc-artifacts2.sambanovasystems.com/sw-docker-scratch/`.

```bash
# Build + push (login node)
sudo -g docker /usr/bin/docker-wrapper build -t IMAGE:TAG .
sudo -g docker /usr/bin/docker-wrapper push IMAGE:TAG

# Run on GPU node (inside srun job)
sudo -g docker /usr/bin/cuda-docker-run-wrapper --net=host --rm IMAGE CMD
```

`--net=host` is required for RoCE RDMA. `-v` mounts restricted to `$SLURM_TMPDIR`; use `/import` paths directly instead.

---

## Known gaps

- **vllm nixl_connector patch**: `REGISTER_CONSUMER_MSG` not in stock vllm 0.16.0 — patched in `Dockerfile.gpu`. Source: `sambanova/sn_vllm`.
- **RDU venv torch compat**: s339 has `torch 2.2.0+sn`; vllm 0.16.0 uses torch 2.4+ APIs. Two files patched post-install by `build_rdu_venv.sh` (`torch_utils.py`, `env_override.py`). Long-term: RDU Docker image with correct torch version.
