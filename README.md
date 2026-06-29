# rdu-hdi — GPU Prefill + RDU Decode via Dynamo

Disaggregated inference on SambaNova hardware: H200×4 GPU handles prefill,
SN40L RDU handles decode, coordinated by NVIDIA Dynamo.

**Model:** MiniMax-M2.x FP8 &nbsp;|&nbsp; **First run:** 2026-06-29 — TTFT 230ms, TPOT 2.23ms

---

## Stack

```
andychensn/ucx        ── sn/v1.22 ──► libucx.so  (GPU RoCE RDMA over bnxt_re)
andychensn/nixl       ── sn/rdu-working ──► nixl wheel
                                                  │
vllm 0.16.0 (PyPI)    ──────────────────► GPU prefill (dynamo.vllm)
                                                  │  NIXL KV transfer
andychensn/vllm-rdu   ──────────────────► RDU decode (dynamo.vllm)
                                                  │
ai-dynamo/dynamo v1.2.1 ────────────────► Dynamo control plane + dynamo.vllm
```

All source dependencies are pinned to exact commit SHAs in `config/versions.env`.
No NFS venv copies. Everything is fetched or built from pinned public sources.

---

## Setup & Launch

See **[docs/quickstart.md](docs/quickstart.md)** — one-time setup (~45 min) + per-session launch.

```bash
# Per-session (after one-time setup):
bash launch/control_plane.sh          # etcd + NATS + Dynamo frontend
bash launch/gpu_prefill.sh            # GPU worker (blocks ~10 min)
source config/cluster.env && bash launch/rdu_decode.sh  # RDU worker (blocks ~12 min)
bash launch/control_plane.sh --stop && scancel $(squeue -u $USER -h -o '%i')
```

---

## Repo contents (16 files)

```
config/
  versions.env    — all commit SHAs and version pins
  cluster.env     — node names, IPs, reservations, FAST_COE_HOME
launch/
  control_plane.sh / gpu_prefill.sh / rdu_decode.sh / rdu_inner.sh
scripts/
  build_gpu_venv.sh  — UCX + NIXL + .venv_gpu  (~30 min on H200)
  build_rdu_venv.sh  — .venv_rdu on s339 via snrdu  (~10 min)
  fetch_vendor.sh    — etcd + nats-server (SHA256-verified from GitHub releases)
  fetch_rdu_wheels.sh — vllm CPU wheel + nixl-pathb + dynamo-runtime
  benchmark.sh       — wrapper for InferenceX/benchmark_serving.py
  build_vllm_cpu_wheel.sh / test_*.sh
docs/
  quickstart.md   — new-member setup guide
```

Runtime-only (gitignored, set up by quickstart Step 1):
`dynamo_src/`, `dynamo_src_rdu/`, `vllm-rdu/`, `InferenceX/`, `vendor/bin/`, `.venv_gpu/`, `.venv_rdu/`

---

## Version pins

All pins are in `config/versions.env`. Key ones:

| Component | Pin |
|-----------|-----|
| vllm | 0.16.0 (official PyPI) |
| torch | 2.9.1+cu130 |
| UCX | `andychensn/ucx@e153f2e4` |
| NIXL | `andychensn/nixl@c2abc770` |
| vllm-rdu | `andychensn/vllm-rdu@5bc4a563` |
| ai-dynamo-runtime | 1.2.1 |
| dynamo Python source | `ai-dynamo/dynamo@919682da` (v1.2.1) |
| deep-gemm | `deepseek-ai/DeepGEMM@477618cd` |
| etcd | 3.5.15 (SHA256 in versions.env) |
| nats-server | 2.10.28 (SHA256 in versions.env) |

---

## Component repos

| Repo | Purpose |
|------|---------|
| [`andychensn/ucx`](https://github.com/andychensn/ucx) | UCX 1.22 + SN RDMA patches for bnxt_re |
| [`andychensn/nixl`](https://github.com/andychensn/nixl) | NIXL + SN UCX integration |
| [`andychensn/vllm-rdu`](https://github.com/andychensn/vllm-rdu) | vLLM hardware plugin for SambaNova RDU |
