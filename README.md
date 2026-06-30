# rdu-hdi — GPU Prefill + RDU Decode via Dynamo

Disaggregated inference on SambaNova hardware: H200×4 GPU handles prefill,
SN40L RDU handles decode, coordinated by NVIDIA Dynamo.

**Model:** MiniMax-M2.x FP8 &nbsp;|&nbsp; **First run:** 2026-06-29 — TTFT 230ms, TPOT 2.23ms

---

## Stack

```
vllm/vllm-openai:v0.16.0 (Docker)          GPU prefill (dynamo.vllm)
  + andychensn/ucx   sn/v1.22    ──────►   │  NIXL KV cache transfer (RoCE RDMA)
  + andychensn/nixl  sn/rdu-working         │
  + vllm nixl_connector patch               │
  + ai-dynamo[vllm] 1.2.1                   │
                                            │
andychensn/vllm-rdu (NFS venv) ──────────► RDU decode (dynamo.vllm)
  + ai-dynamo[vllm] 1.2.1                   │
  + NIXL (pathb/bnxt_re build)              │
                                            │
etcd + NATS + dynamo.frontend ──────────── Control plane (login node)
  (vendor/bin + .venv_cp)
```

All source dependencies are pinned to exact commit SHAs in `config/versions.env`.
GPU worker runs as a Docker container; no venv build required on GPU nodes.

---

## Setup & Launch

See **[docs/quickstart.md](docs/quickstart.md)** — one-time setup + per-session launch.

```bash
# Per-session (after one-time setup):
bash launch/control_plane.sh                                 # etcd + NATS + Dynamo frontend
bash launch/gpu_prefill.sh                                   # GPU Docker worker (blocks ~10 min)
source config/cluster.env && bash launch/rdu_decode.sh       # RDU worker (blocks ~12 min)
bash launch/control_plane.sh --stop && scancel $(squeue -u $USER -h -o '%i')
```

---

## Repo contents

```
config/
  versions.env    — all commit SHAs and version pins
  cluster.env     — node names, IPs, reservations, Docker image tag
Dockerfile.gpu    — GPU prefill image (vllm base + UCX + NIXL + patch + ai-dynamo)
launch/
  control_plane.sh / gpu_prefill.sh / rdu_decode.sh / rdu_inner.sh
scripts/
  build_docker_gpu.sh    — build + push GPU Docker image (login node, ~20 min)
  build_rdu_ucx_nixl.sh  — fetch + compile UCX/NIXL for RDU venv (two-phase)
  build_rdu_venv.sh      — build RDU Python venv on s339
  fetch_vendor.sh        — download etcd + nats-server (SHA256-verified)
  benchmark.sh           — wrapper for InferenceX/benchmark_serving.py
  test_*.sh              — import validation scripts
patches/
  vllm_nixl_connector.patch  — adds REGISTER_CONSUMER_MSG to vllm 0.16.0
docs/
  quickstart.md   — new-member setup guide
```

Runtime-only (gitignored, set up by quickstart):
`.venv_cp/`, `.venv_rdu/`, `.gpu_cache/`, `vllm-rdu/`, `InferenceX/`, `vendor/bin/`

---

## Version pins

All pins are in `config/versions.env`. Key ones:

| Component | Pin |
|-----------|-----|
| vllm Docker base | `vllm/vllm-openai:0.16.0` |
| GPU image tag | `v0.16.0-rdu-hdi.1` (in `cluster.env`) |
| UCX | `andychensn/ucx@e153f2e4` (sn/v1.22) |
| NIXL | `andychensn/nixl@c2abc770` (sn/rdu-working) |
| vllm-rdu | `andychensn/vllm-rdu@5bc4a563` |
| ai-dynamo + runtime | 1.2.1 |
| etcd | 3.5.15 (SHA256 in versions.env) |
| nats-server | 2.10.28 (SHA256 in versions.env) |

---

## Component repos

| Repo | Purpose |
|------|---------|
| [`andychensn/ucx`](https://github.com/andychensn/ucx) | UCX 1.22 + SN RDMA patches for bnxt_re |
| [`andychensn/nixl`](https://github.com/andychensn/nixl) | NIXL + SN UCX integration |
| [`andychensn/vllm-rdu`](https://github.com/andychensn/vllm-rdu) | vLLM hardware plugin for SambaNova RDU |
| [`sambanova/sn_vllm`](https://github.com/sambanova/sn_vllm) | vLLM fork — source of `patches/vllm_nixl_connector.patch` |
