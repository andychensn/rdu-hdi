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
sambanova/fast-coe's vllm-rdu             ──  RDU decode worker
  (Docker image, Dockerfile.rdu)            │   (hdi's proven connector/engine,
  + andychensn/ucx + andychensn/nixl         │    not the retired standalone
    (repo-built, rdu-ucx-install/)          │    andychensn/vllm-rdu fork)
  + coe_api/rdu_engine + BAR2 runtime        │
    connector libs (self-built, baked in)   │
  + ai-dynamo (base package only)            │
                                             │
etcd + NATS + dynamo.frontend             ──  Control plane
  (Docker image, Dockerfile.control-plane)   │   (login node)
```

`coe_api`/BAR2 runtime connector libs are **self-built and baked directly into the RDU Docker
image** — built from a single pinned commit of `josephp/nova/ddr_alloc_mem2mem_AND_bar2_mappings`
(see `config/versions.env`'s `SOFTWARE_REPO_*` comment and `scripts/build_bar2.sh`), no NFS mount
needed at container runtime. (An earlier self-build attempt against the wrong branch/commit hit a
hard ABI/hardware blocker — resolved 2026-07-05 once the actual working commit was identified
directly from the engineer whose NFS tree this replaces.)

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

# 5. Self-build coe_api/rdu_engine + the BAR2 runtime connector libs (required
#    by build_docker_rdu.sh below) — same two-phase pattern.
bash scripts/build_bar2.sh --fetch-only
snrdu run -sp "$RDU_PARTITION" --qos "$RDU_QOS" --nodelist "$RDU_NODE" \
    --allow-local-lib-python --reservation "$RDU_RESERVATION" \
    --pef "$PEF" --timeout "$RDU_TIMEOUT" -o logs/build_bar2.log \
    -- bash scripts/build_bar2.sh --build-only
```

---

## Per-session launch

**Docker is the supported path for all three components** (control plane, GPU prefill, RDU decode)
as of 2026-07-05. Build the images once:

```bash
bash scripts/build_docker_control_plane.sh   # -> $CONTROL_PLANE_IMAGE (config/cluster.env)
bash scripts/build_docker_gpu.sh             # -> $GPU_IMAGE (config/cluster.env)
bash scripts/build_docker_rdu.sh             # -> $RDU_IMAGE (config/cluster.env) — bakes in
                                              #    self-built coe_api/rdu_engine + BAR2 runtime
                                              #    connector libs (scripts/build_bar2.sh), no NFS
                                              #    mount needed for any of it
```

Then launch, in order:

```bash
# 1. Control plane — etcd + NATS + dynamo.frontend in one container, --net=host on the login node
#    NOTE: use -e VAR="$VAR" (explicit value), not bare -e VAR — sudo strips the calling
#    shell's environment by default, so bare -e VAR forwards an EMPTY value and the
#    entrypoint fails with "CONTROL_PLANE_IP must be set".
sudo -g docker /usr/bin/docker-run-wrapper --pull=always --net=host --rm \
    --name rdu-hdi-control-plane \
    -e CONTROL_PLANE_IP="$CONTROL_PLANE_IP" -e ETCD_PORT="$ETCD_PORT" \
    -e NATS_PORT="$NATS_PORT" -e VLLM_PORT="$VLLM_PORT" \
    "$CONTROL_PLANE_IMAGE" &   # (source config/cluster.env first; run in the background, this is persistent)

# 2. GPU prefill (~10 min: model load + warmup)
bash launch/gpu_prefill.sh

# 3. RDU decode — waits for GPU registration, then ~12-14 min BAR2/PEF init
#    via snrdu on the RDU node, see scripts/_run_docker_rdu_decode.sh for the
#    full docker-run-wrapper invocation (--net=host, --device /dev/rdu
#    --device /dev/rdu_mem_map --device /dev/infiniband, --ulimit memlock=-1,
#    --cap-add IPC_LOCK, and MODEL/SERVED_MODEL_NAME/MAX_MODEL_LEN/PEF/
#    MODEL_CONFIG/CONTROL_PLANE_IP/... env vars — no BAR2_INSTALL/
#    BAR2_RUNTIME_LIBS/BAR2_PRELOAD needed, they're baked into the image)

# 4. Warmup (first request ~47s, cold NIXL init)
source config/model.env
curl -s http://localhost:18000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"hello\",\"max_tokens\":1}"
```

> Do not start RDU decode before GPU prefill — Dynamo builds the wrong pipeline.

### Bare-metal launch (deprecated)

`launch/control_plane.sh` (bare etcd/NATS/dynamo processes) and `launch/rdu_decode.sh` (bare
`.venv_rdu`) still exist for reference/local debugging, but are no longer the supported path and are
not actively maintained — `launch/rdu_decode.sh` in particular will fail as-is, since
`config/cluster.env` no longer defines the `BAR2_RUNTIME_LIBS`/`BAR2_PRELOAD` vars it references
(see that script's own header for how to restore them if you need this path). GPU prefill has always
been Docker-only (`launch/gpu_prefill.sh`), unaffected by this.

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

Three wrappers on the cluster:

| Wrapper | Where | Supports |
|---------|-------|---------|
| `/usr/bin/docker-wrapper` | Login node | build, push, pull, ps, … (not `run`) |
| `/usr/bin/cuda-docker-run-wrapper` | GPU nodes | run only (GPU passthrough + `/import`,`/scratch` auto-mount) |
| `/usr/bin/docker-run-wrapper` | RDU nodes | run only (`/import`,`/scratch` auto-mount; device passthrough and RDMA still need explicit flags — see `scripts/test_docker_rdu_e2e.sh`) |

All three require `sudo -g docker` (via `sudo -g docker /usr/bin/<wrapper>`). Internal registry:
`sc-artifacts2.sambanovasystems.com/sw-docker-scratch/`. `--net=host` required on both GPU and RDU
nodes (for RoCE RDMA). Containers run as non-root — default any writable-path config to `/tmp/...`
unless proven otherwise.

---

## NFS dependencies

Confirmed by a full wipe-and-rebuild reproducibility test (2026-07-05): the only genuinely
external assets this stack reads from a network filesystem at runtime are:

1. **The model checkpoint** — split across two paths for the two sides: `MODEL` (`config/model.env`,
   `/import/...`) is read by GPU prefill and by RDU decode's tokenizer/config path (RDU loads
   weights with `--load-format dummy`, it doesn't read tensor data from here); `MINI_CKPT_FP8`
   (`config/minimax_m2.yaml`, `/scratch/...`) is the actual weight source the RDU engine loads.
2. **The PEF** — `PEF` (`config/model.env`, `/import/...`) and the identical path embedded as
   `MINI_PEF_FP8` in `config/minimax_m2.yaml`.
3. **`BRCM_ROCELIB`** (`/import/it-tools/idc/fw/brcm/237/bcm_237.1.148.0a/drivers_linux/bnxt_rocelib`)
   — build-time only, staged into the build context and `COPY`'d into `Dockerfile.gpu`; never
   referenced at container runtime.

Two more `/import` reads exist but are repo-owned, not external dependencies: `MODEL_CONFIG`
(`config/minimax_m2.yaml`, tracked in this repo) and GPU prefill's cache dirs (`.gpu_cache/`,
gitignored scratch space). Both are reachable only because `cuda-docker-run-wrapper`/
`docker-run-wrapper` auto-mount `/import`/`/scratch` with no explicit `-v` flags anywhere in
`launch/gpu_prefill.sh` or `scripts/_run_docker_rdu_decode.sh` — worth re-checking if this stack is
ever deployed somewhere without that auto-mount convention (e.g. a future k8s deploy).

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

## Docker RDU decode — notes

Docker works (`Dockerfile.rdu`, `docker/rdu-decode-entrypoint.sh`). Key non-obvious fix required:

1. **SambaNova's `bnxt_re` RDMA provider replacement**: RDU nodes (confirmed on `sc3-s339`)
   deliberately disable the stock rdma-core `bnxt_re` userspace provider (renamed `.orig`) in favor
   of a SambaNova-supplied one at `/opt/sambanova/lib/libbnxt_re-rdmav34.so` — a different bug than
   the GPU side's Broadcom OOT issue above, but the same symptom class (`UCX ERROR no usable
   transports/devices`). `rhel810-dev` (this image's base) doesn't ship the replacement, so it's
   vendored directly into the repo (`vendor/bnxt_re/`) and swapped in during the Dockerfile build —
   see `Dockerfile.rdu`'s comments for the exact mechanism. Package-version matching alone
   (`rdma-core-48.0` present in both bare metal and container) does not catch this; it's a
   file-level swap, not a package-level one.
2. **`coe_api`/BAR2 runtime connector libs are baked in** (self-built, see `scripts/build_bar2.sh`
   and the architecture note above) — no `BAR2_INSTALL`/`BAR2_RUNTIME_LIBS`/`BAR2_PRELOAD` env vars
   or NFS mounts needed at container runtime. `rdu_engine`'s compiled extension also transitively
   needs `libmpi.so.12` and a few abseil/circllhist libs — these ship inside `rhel810-dev` under
   `/opt/sambanova/lib` but aren't on the default `LD_LIBRARY_PATH`; `Dockerfile.rdu` adds that
   directory explicitly (a bare-metal launch masks this, since the node's own ambient
   `LD_LIBRARY_PATH` already includes it — a container starts clean).
