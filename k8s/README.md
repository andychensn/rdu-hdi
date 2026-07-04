# rdu-hdi Kubernetes deployment (Phase 5 design)

Design produced per `docs/local/DOCKERIZE_BAR2_PLAN.md` §4.6-§4.10 and Phase 5. **Not yet validated
against a real k8s cluster with RDU nodes** — none was available during this project (see Phase 6 /
`docs/local/DOCKERIZE_BAR2_PLAN.md` §8 "residual open items"). Treat this as a reviewed design, not
a proven deployment.

## Deploy order

1. **`00-rdu-device-plugin.yaml`** — RDU-node prerequisite. This is a *reference stub*, not the real
   manifest: the actual device plugin lives in SambaNova's software repo at
   `k8s/k8s-device-plugin/rdu-plugin-ds.yaml` (per the plan's §4.6b finding) and must be sourced from
   there — it is not part of this repo and was not fabricated here. The stub documents the contract
   this deployment depends on (resource name `sambanova.ai/rdu-tile`, the `/sys/class/rdu_class`
   readiness gate) so the dependency is explicit rather than assumed.
2. **`01-configmap.yaml`** — non-secret cluster/model config, mirrors `config/cluster.env` +
   `config/model.env`. Edit this (not the Deployments) when switching models/clusters.
2. **`02-control-plane.yaml`** — etcd + NATS + Dynamo frontend, `Deployment` + `Service`. No RDU/GPU
   dependency, runs on any node (§5 "Phase 3" note).
3. **`03-gpu-prefill.yaml`** — GPU prefill `Deployment`. Requires a GPU-labeled node.
4. **`04-rdu-decode.yaml`** — RDU decode `Deployment`. Requires an RDU-labeled node with the device
   plugin (step 1) running and healthy.

```bash
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/02-control-plane.yaml
kubectl apply -f k8s/03-gpu-prefill.yaml
kubectl apply -f k8s/04-rdu-decode.yaml
```

(`00-rdu-device-plugin.yaml` is documentation only — apply the real DaemonSet from the software repo
instead, see the file's header comment.)

## Design decisions, and why (see plan §4.6-§4.8 for the full evidence trail)

- **Device plugin over privileged pods** (§4.6): `fast-coe`'s own production operator uses
  privileged pods + hostPath for RDU access, with no k8s resource accounting for RDUs. A real
  kubelet device plugin (`sambanova.ai/rdu-tile`) already exists in the software repo, is
  non-privileged, and its own startup check (`/sys/class/rdu_class`) doubles as the "is `snd`
  healthy" precondition. We use the device plugin, not privileged pods.
- **Device plugin alone is not sufficient for RDMA** (§4.8, a real gap the plan's initial draft
  missed): `Allocate()` only wires `/dev/rdu`/`/dev/rdu_mem_map`. RDU decode also needs RDMA over
  RoCE (same as GPU prefill's existing `--net=host --ulimit memlock=-1 --device /dev/infiniband`
  Docker invocation). Resolved with three more additions, all non-privileged:
  - `hostPath` volume mount for `/dev/infiniband` (world-writable device nodes, confirmed via direct
    inspection — no `privileged: true` needed).
  - `hostNetwork: true` (RoCE IP visibility — UCX/NIXL need to see the real host `bnxt_re*` IP).
  - `securityContext.capabilities.add: ["IPC_LOCK"]` (k8s-native equivalent of `--ulimit
    memlock=-1`, needed for RDMA memory registration).
  This combination applies to **both** the GPU-prefill and RDU-decode Deployments — the underlying
  NIXL/UCX-over-RoCE requirement is identical on both sides of the disaggregated pipeline (§5).
- **BAR2/NFS paths stay hostPath-mounted, not baked into the image** (per the explicit
  2026-07-04 project decision to defer full `coe_api` self-build — see `config/versions.env`'s
  `SOFTWARE_REPO_*` comment): `BAR2_INSTALL`, `BAR2_RUNTIME_LIBS`, `BAR2_PRELOAD`, the model
  checkpoint, and the PEF are all under `/import` on these clusters — mounted as a single `hostPath`
  volume at `/import` (matching what `docker-run-wrapper` already does automatically on bare
  SLURM/RDU nodes) rather than mounting each sub-path individually, so path changes in
  `config/cluster.env`/`config/model.env` don't require editing the pod spec's volume list.
- **`CONTROL_PLANE_IP` is the Service DNS name, not a literal IP**: on bare metal this env var is a
  real IP (the login node's), but `docker/control-plane-entrypoint.sh` and
  `docker/rdu-decode-entrypoint.sh` only ever use it to build `http://`/`nats://` URLs — a
  ClusterIP Service's DNS name (`rdu-hdi-control-plane.default.svc.cluster.local`) works
  identically there. GPU-prefill and RDU-decode pods run with `hostNetwork: true` (for RoCE) but can
  still resolve/reach a ClusterIP Service through kube-proxy in the normal way — no `hostNetwork`
  needed on the control-plane pod itself.
- **RDU-decode's own bnxt_re provider fix stays inside the image** (Phase 4, `Dockerfile.rdu` +
  `vendor/bnxt_re/`) — no additional k8s-side accommodation is needed for it; it's a Docker image
  build-time concern, orthogonal to the pod spec.

## What this design does NOT cover (see plan §8)

- **Not validated against a live k8s cluster with RDU nodes** — no such cluster was available during
  this project. Everything above is a reviewed paper design, following the same non-privileged
  device-plugin + hostPath + hostNetwork + IPC_LOCK pattern already confirmed sufficient for Docker
  on bare SLURM nodes (Phase 3/4), but the k8s-specific mechanics (Service DNS resolution under
  hostNetwork, device plugin `Allocate()` behavior under a real kubelet, NUMA/CPU-manager
  interaction with `SF_RNT_NUMA_BIND`) have not been exercised.
- **`nodeSelector` labels are placeholders** (`sambanova.ai/rdu=true`, `nvidia.com/gpu=true`) —
  adjust to match your cluster's actual node labeling convention (the plan notes `fast-coe` uses an
  `snRduArch=sn40-16`-style convention).
- **No Helm chart / no CRD / no autoscaling** — deliberately kept as plain Deployments + a Service
  per the plan's Phase 5 guidance ("a plain Helm chart or a pair of Deployments... is simpler and
  doesn't require running a custom operator"). Revisit if multi-model/multi-cluster reuse becomes a
  real need.
