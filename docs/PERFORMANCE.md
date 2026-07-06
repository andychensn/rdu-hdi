# Performance

Latest full benchmark sweep, measured 2026-07-05 against commit `72dfcc71` (Part A repo
restructure) — reflects the current repo layout (`build/`, `docker/{gpu,rdu,control-plane}/`,
`launch/`, `bench/`, `test/`) and the `sambanova-deps-brcm-roce-userland` RPM-based bnxt_re fix
from `cc050eec`. `launch/rdu_decode.sh`'s self-dispatch alignment (`dbe3fdf6`) landed after this
sweep but changes only job submission/orchestration, not the served container's runtime behavior,
so these numbers still reflect current `main`.

## Setup

| Component | Details |
|-----------|---------|
| GPU prefill | sc3-c129, H200×4 TP4, `docker/gpu/Dockerfile` |
| RDU decode | sc3-s339, SN40-16, `docker/rdu/Dockerfile` (self-built `coe_api`/`rdu_engine` + BAR2 runtime baked in) |
| Control plane | etcd 3.5.15 + NATS 2.14.3 + ai-dynamo-runtime 1.2.1, `docker/control-plane/Dockerfile` |
| Model | MiniMax-M2.7 FP8 (TP4 prefill, 16-chip RDU decode) |
| Benchmark tool | SemiAnalysisAI/InferenceX @ 37505e11, via `bench/sweep.sh` (unique `--seed` per invocation) |

## Results

| ISL | OSL | Conc | Prompts | Succeeded | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | E2EL mean | req/s | out tok/s |
|-----|-----|------|---------|-----------|-----------|----------|----------|-----------|-----------|-------|-----------|
| 1000 | 1000 | 1 | 10 | 10 | 254 ms | 301 ms | 351 ms | 2.23 ms | 2484 ms | 0.402 | 402.4 |
| 1000 | 1000 | 2 | 20 | 20 | 331 ms | 364 ms | 420 ms | 2.24 ms | 2566 ms | 0.779 | 778.7 |
| 1000 | 1000 | 4 | 40 | 40 | 2655 ms | 2860 ms | 3091 ms | 2.28 ms | 4935 ms | 0.791 | 790.7 |
| 10000 | 1000 | 1 | 10 | 10 | 894 ms | 973 ms | 1035 ms | 2.38 ms | 3274 ms | 0.305 | 305.3 |
| 10000 | 1000 | 2 | 20 | 20 | 977 ms | 1123 ms | 1223 ms | 2.48 ms | 3458 ms | 0.576 | 575.8 |
| 10000 | 1000 | 4 | 40 | 40 | 3325 ms | 3515 ms | 4375 ms | 2.45 ms | 5777 ms | 0.675 | 675.1 |
| 100000 | 1000 | 1 | 10 | 10 | 22308 ms | 23025 ms | 23244 ms | 4.31 ms | 26618 ms | 0.038 | 37.6 |
| 100000 | 1000 | 2 | 10 | 10 | 39120 ms | 41905 ms | 42017 ms | 4.31 ms | 43429 ms | 0.044 | 43.7 |
| 100000 | 1000 | 4 | 10 | 10 | 73152 ms | 87103 ms | 88320 ms | 4.31 ms | 77460 ms | 0.044 | 43.9 |

**Zero failed requests** across all 9 configs.

## Reproducing

```bash
bash bench/sweep.sh --label my_run
```

Full raw result JSONs and the underlying `RESULTS.md` for this sweep are in
`benchmark_results/part_a_verify/` (gitignored, local only).
