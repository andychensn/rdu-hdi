# Performance

Latest full benchmark sweep, measured 2026-07-06 after a complete wipe-and-rebuild reproducibility
test: every local build artifact and locally-cached Docker image tag was deleted, then the entire
stack was rebuilt from nothing but `README.md` + this repo's own scripts (fresh UCX/NIXL/vllm+cpu
build, fresh `coe_api`/BAR2 self-build, three `--no-cache` Docker builds). Confirms current `main`
is genuinely reproducible from a clean state, not just functionally correct — see
`benchmark_results/repro_test_20260706/RESULTS.md` and
`benchmark_results/REPRO_TEST_20260706_VS_HDI.md` (gitignored, local only) for the full
reproducibility narrative and cross-codebase comparison.

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
| 1000 | 1000 | 1 | 10 | 10 | 233 ms | 265 ms | 306 ms | 2.22 ms | 2454 ms | 0.407 | 407.3 |
| 1000 | 1000 | 2 | 20 | 20 | 309 ms | 326 ms | 344 ms | 2.23 ms | 2540 ms | 0.786 | 786.4 |
| 1000 | 1000 | 4 | 40 | 40 | 2598 ms | 2744 ms | 2916 ms | 2.32 ms | 4921 ms | 0.792 | 792.3 |
| 10000 | 1000 | 1 | 10 | 10 | 909 ms | 1057 ms | 1127 ms | 2.38 ms | 3283 ms | 0.305 | 304.5 |
| 10000 | 1000 | 2 | 20 | 20 | 971 ms | 1140 ms | 1257 ms | 2.47 ms | 3442 ms | 0.578 | 578.5 |
| 10000 | 1000 | 4 | 40 | 40 | 3303 ms | 3532 ms | 4081 ms | 2.49 ms | 5794 ms | 0.674 | 673.5 |
| 100000 | 1000 | 1 | 10 | 10 | 22133 ms | 22706 ms | 23007 ms | 4.32 ms | 26446 ms | 0.038 | 37.8 |
| 100000 | 1000 | 2 | 10 | 10 | 38702 ms | 41709 ms | 41742 ms | 4.32 ms | 43014 ms | 0.044 | 44.2 |
| 100000 | 1000 | 4 | 10 | 10 | 73244 ms | 86955 ms | 87348 ms | 4.32 ms | 77557 ms | 0.044 | 43.9 |

**Zero failed requests** across all 9 configs.

## Reproducing

```bash
bash bench/sweep.sh --label my_run
```

Full raw result JSONs and the underlying `RESULTS.md` for this sweep are in
`benchmark_results/repro_test_20260706/` (gitignored, local only).
