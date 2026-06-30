# rdu-hdi Quick Start

End-to-end setup for a new team member: GPU prefill on H200 + RDU decode on SN40L via Dynamo.

**Estimated time:** ~45 min one-time setup · ~25 min to launch per session.

---

## Prerequisites

| Requirement | Notes |
|------------|-------|
| sc-vnc9 login node access | Run all scripts from here |
| sc3-c129 GPU reservation | `RDU-prefill-experiment-test` |
| sc3-s339 RDU reservation | `no_sf_catchup_demos` (max 01:50:00) |
| `snrdu` on PATH | SambaNova job scheduler — pre-installed on sc-vnc9 |
| CUDA 13.x on GPU node | `nvcc`, `autoreconf`, `libtoolize`, `rdma-core-devel` |
| PEF file path | Set `PEF=` in `config/cluster.env` before Step 5 |
| BAR2 runtime on s339 | SambaNova hardware SDK — set by snrdu env. Paths in `rdu_inner.sh` (`BAR2_INSTALL`, `BAR2_PRELOAD`, `SOFTWARE_BUILD`) default to guoyaof/jayr NFS; ask your manager for access or the right paths for your setup. This is the RDU equivalent of `/usr/local/cuda`. |

---

## One-time setup

Run these **once** after cloning. Everything is fetched from pinned public sources.
No venv copying from NFS.

### 0. Clone the repo

```bash
git clone https://github.com/andychensn/rdu-hdi.git
cd rdu-hdi
REPO=$(pwd)   # save — used throughout these steps
```

### 1. Clone runtime dependencies (gitignored, required at runtime)

```bash
source "$REPO/config/versions.env"

# vllm-rdu hardware plugin
gh repo clone andychensn/vllm-rdu "$REPO/vllm-rdu"
git -C "$REPO/vllm-rdu" checkout "$VLLM_RDU_COMMIT"

# InferenceX benchmark tooling (SemiAnalysisAI/InferenceX)
git clone https://github.com/SemiAnalysisAI/InferenceX.git "$REPO/InferenceX"
git -C "$REPO/InferenceX" checkout "$INFERENCEX_COMMIT"
```

### 2. Fetch infrastructure binaries (~1 min, needs internet)

Downloads etcd and nats-server from official GitHub releases and verifies SHA256:

```bash
bash "$REPO/scripts/fetch_vendor.sh"
```

### 3. Build GPU venv (~30 min on H200, needs CUDA 13.x + autotools)

```bash
srun -p gpuonly -w sc3-c127 --gres=gpu:4 -c 16 --mem=65536 -t 01:30:00 \
    bash "$REPO/scripts/build_gpu_venv.sh"
```

> Uses sc3-c127 (not c129) because `sudo docker` works there if needed for Docker builds.
> Clones UCX@`$UCX_COMMIT` and NIXL@`$NIXL_COMMIT` — both public on github.com.

### 4. Build RDU UCX + NIXL + wheels (~15 min)

s339 has no internet, so everything must be fetched first from the login node.
`--fetch-only` clones UCX/NIXL sources **and** downloads the vllm/dynamo-runtime wheels to `wheelhouse/`.

```bash
# Phase 1: fetch all sources and wheels (login node, needs internet, ~3 min)
bash "$REPO/scripts/build_rdu_ucx_nixl.sh" --fetch-only

# Phase 2: compile UCX + NIXL on s339 (no internet needed, ~13 min)
source "$REPO/config/cluster.env"
snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
    --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:30:00 \
    -o "$REPO/logs/build_rdu_ucx_nixl.log" \
    -- bash "$REPO/scripts/build_rdu_ucx_nixl.sh" --build-only

tail -f "$REPO/logs/build_rdu_ucx_nixl.log"
```

Outputs: `rdu-ucx-install/`, `wheelhouse/nixl_cu12-*cp311*.whl`, `wheelhouse/vllm-*.whl`, `wheelhouse/ai_dynamo_runtime-*.whl`.

### 5. Build RDU venv (~10 min on sc3-s339)

```bash
# $PEF already loaded from cluster.env above

snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
    --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:30:00 \
    -o "$REPO/logs/build_rdu_venv.log" \
    -- bash "$REPO/scripts/build_rdu_venv.sh"

tail -f "$REPO/logs/build_rdu_venv.log"
```

### Validate venvs (optional but recommended)

```bash
# Validate GPU venv (run on GPU node)
srun -p gpuonly -w sc3-c127 --gres=gpu:1 -c 2 --mem=8192 -t 00:05:00 \
    bash "$REPO/scripts/test_rdu_imports.sh"  # reuses the import checks

# Validate RDU venv (run on s339)
snrdu run -sp zd3 --qos 5 --nodelist sc3-s339 --allow-local-lib-python \
    --reservation no_sf_catchup_demos --pef "$PEF" --timeout 00:05:00 \
    -o "$REPO/logs/test_rdu_imports.log" \
    -- bash "$REPO/scripts/test_rdu_imports.sh"
```

---

## Per-session launch

After one-time setup, 3 commands + a warmup:

```bash
cd "$REPO"   # or wherever you cloned rdu-hdi

# 1. Start control plane (etcd + NATS + Dynamo frontend)
bash launch/control_plane.sh

# 2. Start GPU prefill — blocks ~10 min (model load + DeepGEMM warmup)
bash launch/gpu_prefill.sh

# 3. Start RDU decode — auto-waits for GPU registration, then blocks ~12 min
source config/cluster.env && bash launch/rdu_decode.sh

# 4. Warmup — ALWAYS do this before benchmarking.
#    First request takes ~47s (cold NIXL init). All subsequent are fast.
curl -s http://10.10.0.156:18000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiniMax-M2.7","prompt":"hello","max_tokens":1}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('warmup OK:', d['usage'])"
```

> `rdu_decode.sh` gates on GPU prefill registration before submitting snrdu.
> Do not start RDU before GPU or Dynamo will build the wrong pipeline.

## Benchmark

Uses [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) (public,
pinned at `$INFERENCEX_COMMIT` in `versions.env`). Cloned in Step 1.

```bash
# ISL=1000, OSL=1000, concurrency=1
bash scripts/benchmark.sh --input-len 1000 --output-len 1000 --concurrency 1 --num-prompts 10

# ISL=10000, OSL=1000
bash scripts/benchmark.sh --input-len 10000 --output-len 1000

# Custom endpoint / model
bash scripts/benchmark.sh --endpoint http://10.10.0.156:18000 --model MiniMax-M2.7
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
| UCX 1.22 | `andychensn/ucx@e153f2e4` | ✅ |
| NIXL | `andychensn/nixl@c2abc770` | ✅ |
| vllm-rdu | `andychensn/vllm-rdu@5bc4a563` | ✅ |
| vllm 0.16.0 | PyPI | ✅ |
| torch 2.9.1+cu130 | PyTorch wheel index | ✅ |
| deep-gemm | `deepseek-ai/DeepGEMM@477618cd` | ✅ |
| ai-dynamo-runtime 1.2.1 | PyPI | ✅ |
| ai-dynamo + ai-dynamo-runtime | PyPI `1.2.1` (installed by build scripts) | ✅ |
| benchmark tooling | `SemiAnalysisAI/InferenceX@37505e11` | ✅ |
| etcd 3.5.15 | GitHub releases (SHA256-verified) | ✅ |
| nats-server 2.10.28 | GitHub releases (SHA256-verified) | ✅ |
| nixl-pathb wheel | Built from `andychensn/nixl@sn/rdu-working` by Step 4 | ✅ |
| vllm nixl_connector patch | `patches/vllm_nixl_connector.patch` (applied by `build_gpu_venv.sh`) | ✅ |
| BAR2 runtime (s339) | guoyaof/jayr NFS paths | ⚠️ internal |
| Model weights | `/import/ml-sc-scratch6/yund/...` | ⚠️ internal NFS |
| PEF file | `/import/ml-sc-scratch4/jayr/...` | ⚠️ internal NFS |

All version numbers and commit SHAs are in `config/versions.env`.

---

## Known gaps

- **GPU venv vllm patch** (`patches/vllm_nixl_connector.patch`): vllm 0.16.0 is missing `REGISTER_CONSUMER_MSG` support in `nixl_connector.py` — without it the handshake listener crashes with "unhashable type: dict" when the RDU consumer registers. The fix is tracked in `sambanova/sn_vllm` but we use the stock precompiled wheel (avoids ~45 min CUDA recompile) and apply only this 50-line patch. Applied automatically by `build_gpu_venv.sh`.
- **RDU venv torch compat**: s339 has `torch 2.2.0+sn` (SambaNova RDU build); vllm 0.16.0 was written against torch 2.4+. Two files crash at import (`torch_utils.py`: missing `infer_schema`; `env_override.py`: torch 2.9 inductor paths). `build_rdu_venv.sh` patches both after install. Long-term fix: build vllm CPU wheel against torch 2.2.x.
