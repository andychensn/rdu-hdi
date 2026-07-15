# Deploying rdu-hdi on VCC

VCC ("vc2-la-dc-*") is a pair of standalone bare-metal demo nodes (GPU +
RDU), reached directly over SSH -- not the SLURM-scheduled, shared-NFS,
dockerd cluster the rest of this repo (`config/`, `launch/`, `docker/*/build.sh`)
targets. This directory adapts the *deployment* (not the build) to that
different environment.

**Build stays on vnc+idc, exactly as documented in the top-level README.**
Nothing here builds an image -- `docker/gpu/build.sh`, `docker/control-plane/build.sh`,
and `docker/rdu/build.sh` still run on the login node, unchanged. This
directory only covers getting an already-built image onto VCC and running it
there.

## Why this needs its own path (the investigation, briefly)

Confirmed live (2026-07-15), so the design below isn't guesswork:

- **No SLURM at all** on either VCC node (no `sinfo`/`squeue`/`srun`/`snrdu`).
  The vnc+idc launch scripts' whole orchestration model (submit a job,
  poll `squeue`, `scancel` to tear down) has nothing to submit to here --
  replaced with direct SSH execution below.
- **No real `dockerd`.** Both nodes run rootless Podman; `docker` is a CLI
  shim for it (confirmed identical build output between `docker build` and
  `podman build`). No `docker` group, no `sudo -g docker`, no
  `docker-wrapper`/`cuda-docker-run-wrapper`/`docker-run-wrapper` -- those
  are vnc+idc-specific. Just run `podman` directly as yourself.
- **No shared filesystem** between the login node and either VCC node (no
  `/import`, no `/scratch` NFS mount, confirmed via `df`/`mount`). Every
  script here ships itself + a resolved env file to the target node via
  `scp` before running it there, instead of relying on the same repo
  checkout already being visible on both ends the way `srun`/`snrdu` do.
- **`sc-artifacts2.sambanovasystems.com` and `github.sambanovasystems.com`
  are permanently unreachable from VCC** -- confirmed both fail DNS
  resolution on both nodes; infra team confirmed this is structural ("both
  require you to be on the corporate network," not a ticket-fixable gap).
  This is why `vcc/deploy_images.sh` moves images by `podman save` + `rsync`
  + `podman load` instead of registry push/pull, and why `bench/run.sh`'s
  live `git clone` of InferenceX isn't used here. Neither host is reached at
  **runtime** by anything in `docker/rdu/entrypoint.sh` or
  `docker/control-plane/entrypoint.sh` (grepped both directly) -- the
  private-repo clones in `build/rdu_env.sh`/`build/bar2.sh` are build-time
  only and already baked into the image by the time it's built on vnc+idc.
  `github.com` (public) resolves fine from both nodes, for what it's worth.
- **No `nvidia-container-toolkit`/CDI on `GPU_HOST`, and it's not available
  from any configured repo** -- but this is *not* a blocker. Raw
  `--device /dev/nvidia*` passthrough plus `--security-opt label=disable`
  (SELinux is Enforcing and denies device access without it -- this was the
  actual root cause, not a missing toolkit) plus copying three host driver
  libs (`libcuda.so*`, `libnvidia-ml.so*`, `libnvidia-ptxjitcompiler.so*`)
  into a bind-mounted directory with `LD_LIBRARY_PATH` set **at container
  launch** gets real GPU compute -- confirmed with an actual 4096x4096 matmul
  on the B200 inside rootless Podman, using vLLM's own torch build. See
  `vcc/launch/gpu_prefill.sh`'s driver-lib-injection block.
- **`RDU_HOST` needs the same `--security-opt label=disable` treatment** for
  `/dev/rdu`/`/dev/rdu_mem_map` (SELinux is Enforcing there too, same root
  cause class). `docker/rdu/entrypoint.sh`'s RoCE-IP autodetection also
  hardcodes the vnc+idc cluster's `10.17.0.0/16` convention -- patched to
  accept an `RDU_ROCE_IP_OVERRIDE` env var (backward compatible, vnc+idc's
  launch scripts don't set it so their behavior is unchanged) -- **this one
  change needs the RDU image rebuilt once** before it takes effect; every
  other gap here needed zero image changes.
- **Cross-node RoCE reachability between `GPU_HOST` and `RDU_HOST` is
  confirmed working**, unlike the vnc+idc cluster where this is NOT
  guaranteed by node proximity alone (see that repo's own
  `cluster-infra-lessons` history). Both expose interfaces on the same
  `172.16.0.0/24` L2 fabric; pinged 4 different `GPU_HOST` addresses from
  `RDU_HOST` at 0.4-0.5ms RTT.
- **`GPU_HOST`'s RoCE NICs are Mellanox `mlx5`, not Broadcom `bnxt_re`**
  (confirmed via `ibv_devinfo`) -- a different vendor than the GPU side of
  the vnc+idc cluster, so `UCX_NET_DEVICES` names a real local `mlx5`
  interface instead of copying the `bnxt_re0/2/4/6` list.
  `RDU_HOST`'s NICs ARE Broadcom `bnxt_re` (same vendor as vnc+idc's RDU
  node), so `docker/rdu/entrypoint.sh`'s existing default NIC list is
  expected to apply unchanged -- not yet live-verified end to end (blocked
  on the PEF gap below).

## What's simplified vs. vnc+idc (not infra constraints, just scope)

- **1P1D only.** No multi-worker GPU prefill, no LMCache CPU-tier offload.
  Both are straightforward to add back (`vcc/launch/gpu_prefill.sh` already
  parallels `launch/gpu_prefill.sh`'s structure) but there's no evidence yet
  VCC's demo workload needs either.
- **No benchmark tooling ported.** Per direct instruction, benchmarking
  doesn't need to run from VCC -- run `bench/run.sh` from vnc+idc against
  the VCC-hosted endpoint instead (VCC's own DNS gap doesn't matter for a
  `curl`-based benchmark client, only for git/registry access).

## Before you can actually launch: the PEF isn't staged yet

Everything above is confirmed working. This one is not: `RDU_HOST`'s
`/scratch` has a `MiniMax-M2.7-FP8-RDU-packed` checkpoint (matches
vnc+idc's `config/minimax_m2.yaml` exactly) but only a `BS8` PEF variant, not
the `BS2` one `config/model.env`/`config/minimax_m2.yaml` actually use in
production. The exact `BS2` PEF **does** exist on VCC already, just on the
wrong node -- `GPU_HOST` at:

```
/home/jayr/move_to_rdu/minimax-m2__full_layers_TP16_ssSS_CG_SS_MAX_SS_TG_parallel_sdk_fp8_per_tensor_CoE_ckpt_sharing_BS2_SSSS_CG_max196608_SS_MAX_max196608_SS_TG_max196608/minimax_m2_minimal_pef_dir/minimax-m2__full_layers_TP16_ssSS_CG_SS_MAX_SS_TG_parallel_sdk_fp8_per_tensor_CoE_ckpt_sharing_BS2_SSSS_CG_max196608_SS_MAX_max196608_SS_TG_max196608.pef
```

Stage it onto `RDU_HOST` at the path `vcc/model.env`'s `RDU_PEF_PATH` points
to (or edit that variable to wherever you put it):

```bash
ssh vc2-la-dc-sn40-r1h1 mkdir -p /home/andyc/minimax-m2-BS2-pef  # on RDU_HOST, not GPU_HOST
scp -3 \
  'vc2-la-dc-b200-h1:/home/jayr/move_to_rdu/minimax-m2__full_layers_TP16_ssSS_CG_SS_MAX_SS_TG_parallel_sdk_fp8_per_tensor_CoE_ckpt_sharing_BS2_SSSS_CG_max196608_SS_MAX_max196608_SS_TG_max196608/minimax_m2_minimal_pef_dir/minimax-m2__full_layers_TP16_ssSS_CG_SS_MAX_SS_TG_parallel_sdk_fp8_per_tensor_CoE_ckpt_sharing_BS2_SSSS_CG_max196608_SS_MAX_max196608_SS_TG_max196608.pef' \
  'vc2-la-dc-sn40-r1h1:/home/andyc/minimax-m2-BS2-pef/'
```
(`scp -3` routes GPU_HOST -> login node -> RDU_HOST in one hop; swap for two
plain `scp`s through a local temp file if `-3` isn't available. Note:
`RDU_HOST`'s `/scratch` (-> `/var/scratch`) is owned by a corporate-LDAP
UID/group the local VCC account isn't in -- confirmed via a real permission
denial -- hence `/home/andyc` instead of `/scratch` here.)

## Usage

```bash
# One-time (or whenever images are rebuilt): ship images to both nodes
bash vcc/deploy_images.sh              # all three images
bash vcc/deploy_images.sh gpu          # just one, e.g. after a GPU-side rebuild

# Launch, in order (same ordering requirement as vnc+idc: GPU before RDU,
# or Dynamo builds the wrong pipeline)
bash vcc/launch/control_plane.sh &
bash vcc/launch/gpu_prefill.sh
bash vcc/launch/rdu_decode.sh

# Smoke test -- confirmed the login node CANNOT reach either VCC node's
# service ports directly (consistent with the DNS/network isolation above,
# just the reverse direction). Run this from inside VCC instead:
ssh vc2-la-dc-sn40-r1h1 "curl -s http://172.16.0.104:18000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"MiniMax-M2.7\",\"prompt\":\"hello\",\"max_tokens\":1}'"
# (or ssh into GPU_HOST and curl localhost:18000 directly)

# Teardown
bash vcc/teardown.sh
```

## Known-good so far / what's still unverified

Verified live: `vcc/deploy_images.sh` end-to-end (saved the real
control-plane image, rsynced it to `GPU_HOST`, loaded it there), then
`vcc/launch/control_plane.sh` actually launched it via the new SSH/Podman
path and it served real requests -- confirmed both from `GPU_HOST` itself
and, importantly, from `RDU_HOST` reaching across the `172.16.0.0/24` fabric
to `GPU_HOST`'s control plane (the cross-node path the real stack depends
on). Also verified separately: GPU device passthrough + real B200 compute
inside rootless Podman, cross-node RoCE ping reachability, `podman
build`/`run` parity with `docker`.

One thing this test surfaced: the login node (vnc+idc) cannot reach either
VCC node's service ports directly (confirmed -- `curl` to both
`GPU_HOST`'s fabric IP and its other routable IP timed out from the login
node), consistent with the same network isolation behind the DNS gap above,
just the reverse direction. Not a blocker -- it just means smoke-testing/
benchmarking against the endpoint has to happen from inside VCC (SSH in, or
tunnel) rather than directly from the login node's own shell.

Not yet verified end-to-end (blocked on the PEF staging step above, not on
anything infra-related): the full GPU-prefill + RDU-decode stack actually
registering with Dynamo and serving a real completion on VCC. Once the PEF
is staged and the RDU image is rebuilt with the `RDU_ROCE_IP_OVERRIDE` fix,
this should be the next thing to confirm.
