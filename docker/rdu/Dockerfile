# RDU decode image for rdu-hdi disaggregated inference.
#
# Bakes in: /opt/sambanova's Python 3.11 + torch==2.2.0+sn (from the base
# image), vllm+cpu + ai-dynamo + sambanova/vllm-rdu + UCX/NIXL (COPIED as
# pre-built binary artifacts from wheelhouse/ and rdu-ucx-install/), AND
# coe_api/rdu_engine + the BAR2 runtime connector libs (self-built, see
# build/bar2.sh) from wheelhouse/ and rdu-runtime-install/{lib,preload}.
# The model checkpoint and PEF are read from NFS at container runtime
# (large data files this image doesn't embed) — everything else is
# self-contained.
#
# RUN this image with --net=host (RoCE IP visibility, matching
# gpu_prefill.sh's own --net=host requirement) and explicit device
# passthrough for /dev/rdu, /dev/rdu_mem_map, AND /dev/infiniband (three
# separate requirements — docker-run-wrapper's automatic /import,/scratch
# NFS mounting covers none of them, and --net=host alone gives network-
# NAMESPACE visibility but not RDMA hardware access). All are world-
# accessible on this cluster (confirmed via direct inspection), so
# `--device /dev/rdu --device /dev/rdu_mem_map --device /dev/infiniband`
# needs no privileged mode — same RDMA-device pattern gpu_prefill.sh
# already uses. A k8s deployment would use the sambanova.ai/rdu-tile
# device plugin for /dev/rdu*, plus a hostPath volume for /dev/infiniband.
#
# Build (from the login node — the FROM base image brings /opt/sambanova
# along, same principle as docker/gpu/Dockerfile bringing CUDA along via FROM
# vllm/vllm-openai; no snrdu/RDU node needed to BUILD this image, only to
# RUN it):
#   bash docker/rdu/build.sh
ARG RHEL810_DEV_IMAGE

FROM ${RHEL810_DEV_IMAGE} AS builder

ARG BRCM_ROCE_USERLAND_VERSION

RUN dnf install -y git patch rdma-core libibverbs libibverbs-utils && dnf clean all

# SambaNova's RDMA userland replaces the stock rdma-core bnxt_re provider,
# which on this cluster's hardware sends usable-transport-breaking ioctls.
# Without it, UCX reports "no usable transports/devices" for bnxt_re0/2/4/6.
# sambanova-deps-brcm-roce-userland is the real, versioned RPM that
# provisions this on bare metal too (config/versions.env's
# BRCM_ROCE_USERLAND_VERSION), from SambaNova's internal Artifactory --
# rhel810-dev's base doesn't have it installed, but does have the repo
# config to install it directly.
#
# NOTE: the RPM's own postinstall script
# (/opt/sambanova/share/sambanova-deps-brcm-roce-userland/libbnxt_re.sh)
# has a real gap -- it renames the stock driver file AND the stock .so to
# .orig, but only copies the new *driver* file back into place, never the
# .so (verified directly: after a plain `dnf install`,
# /usr/lib64/libibverbs/libbnxt_re-rdmav34.so doesn't exist at all). The
# explicit `cp` below finishes what that script leaves incomplete.
RUN dnf install -y sambanova-deps-brcm-roce-userland-${BRCM_ROCE_USERLAND_VERSION} && \
    cp /opt/sambanova/lib/libbnxt_re-rdmav34.so /usr/lib64/libibverbs/libbnxt_re-rdmav34.so && \
    ldconfig && \
    dnf clean all

ENV PATH=/opt/sambanova/bin:$PATH
ENV PYTHONNOUSERSITE=1

# ── UCX (pre-built) ────────────────────────────────────────────────────────
COPY rdu-ucx-install/ /opt/rdu-ucx/
ENV LD_LIBRARY_PATH=/opt/rdu-ucx/lib:${LD_LIBRARY_PATH:-}
ENV UCX_MODULE_DIR=/opt/rdu-ucx/lib/ucx

# ── BAR2 runtime connector libs (self-built, see build/bar2.sh) ────
# coe_api/rdu_engine's compiled extension dynamically links against
# libc_samba_runtime.so/libcpp_samba_runtime.so at IMPORT time — not just at
# actual RDU session creation — so these must be on LD_LIBRARY_PATH before
# install-deps.sh's own import validation below runs. Baked in
# directly (27 .so files from the runtime 'graph' CMake target group).
COPY rdu-runtime-install/lib/ /opt/bar2-runtime/lib/
COPY rdu-runtime-install/preload/ /opt/bar2-runtime/preload/
# memcopy/scatter-gather utility PEFs (see build/bar2.sh's stage_utility_pefs
# -- staged here, not compiled from this build, see that function's own
# comment for why). coe_api/rdu_engine's C++ default lookup for these is
# ${INSTALL_ROOT:-/opt/sambaflow}/bazel-install/pefs/{name}.pef (confirmed
# directly against coe_api's source, MemCopyApp.cpp/ScatterGatherApp.cpp) --
# rdu_engine.always_load_memcopy_pef(True) (vllm-rdu's own hardcoded,
# unconditional default) and coe_api's own unconditional scatter-gather-app
# establishment both resolve here with zero vllm-rdu-side configuration
# needed, once these files exist at this path.
COPY rdu-runtime-install/pefs/ /opt/bar2-runtime/bazel-install/pefs/
ENV INSTALL_ROOT=/opt/bar2-runtime
# rdu_engine also transitively needs libmpi.so.12 (mpich) and a few abseil/
# circllhist libs -- confirmed via `ldd` on the built .so that these are
# genuinely required, not optional. They ship inside the rhel810-dev base
# image itself under /opt/sambanova/lib (confirmed by direct inspection),
# just not on LD_LIBRARY_PATH by default. On bare metal this is masked by
# whatever ambient LD_LIBRARY_PATH the node's own environment happens to
# already carry -- a container starts clean, so it must be added explicitly.
ENV LD_LIBRARY_PATH=/opt/bar2-runtime/lib:/opt/sambanova/lib:${LD_LIBRARY_PATH:-}

# ── Python deps (vllm+cpu, ai-dynamo, sambanova/vllm-rdu, NIXL, patches) ──
COPY wheelhouse/ /build/wheelhouse/
COPY patches/rdu/ /build/patches/rdu/
# .dockerignore excludes vllm-rdu/.git/ from the build context entirely
# (keeps the context small) — docker/rdu/build.sh verifies the
# checkout is at the pinned VLLM_RDU_COMMIT on the login node, before this
# COPY, since the commit can't be checked here without the .git directory.
COPY vllm-rdu/ /build/vllm-rdu/

ARG RDU_TRANSFORMERS_VERSION
COPY docker/rdu/install-deps.sh /build/install-deps.sh
RUN chmod +x /build/install-deps.sh && \
    RDU_TRANSFORMERS_VERSION="${RDU_TRANSFORMERS_VERSION}" /build/install-deps.sh

# See config/versions.env's RDU_ENGINE_TARGET_RUNTIME_VERSION_OVERRIDE
# comment -- only affects this deployment's rdu_engine.initialize() call
# (rdu_executor/runtime.py), not other vllm-rdu consumers.
ARG RDU_ENGINE_TARGET_RUNTIME_VERSION_OVERRIDE
ENV RDU_ENGINE_TARGET_RUNTIME_VERSION_OVERRIDE="${RDU_ENGINE_TARGET_RUNTIME_VERSION_OVERRIDE}"

COPY docker/rdu/entrypoint.sh /opt/rdu-decode-entrypoint.sh
RUN chmod +x /opt/rdu-decode-entrypoint.sh

RUN rm -rf /build/wheelhouse /build/patches

# rhel810-dev (this image's base) carries at least one file (/etc/munge,
# presumably from an mpich/munge package installed in a corporate-LDAP-
# connected build environment) owned by a UID/GID far outside the 16-bit
# range any rootless-Podman user namespace can represent -- confirmed
# directly: `podman load` on a standalone (non-SLURM, rootless-Podman) VCC
# node failed with "potentially insufficient UIDs or GIDs available in user
# namespace (requested 1447850569:988 for /etc/munge)". munge itself is
# unused here (this deployment never runs coe_api's multi-process/MPI
# "coe-follower" mode -- confirmed earlier this only matters when
# INSTALL_ROOT triggers that codepath, which it doesn't in our
# single-process setup), so normalizing ownership is safe. Broad find+chown
# (not just /etc/munge) in case another exotic-UID file exists elsewhere in
# the base image.
#
# This alone does NOT fix the podman-load failure, though: chown here only
# changes what the FINAL merged filesystem looks like -- it doesn't rewrite
# the earlier layer (baked into rhel810-dev itself, confirmed: the failing
# blob's hash was identical before and after adding this RUN line) that
# actually carries the bad ownership. podman/OCI tooling extracts each
# layer in sequence to build up the merged rootfs; a later layer's chown
# can't prevent an earlier layer's own tar entries from failing to extract
# in the first place under a rootless user namespace. Real fix is the
# `final` flattening stage below -- kept anyway as defense in depth (and
# because it's the only thing that fixes ownership if this image is ever
# used unflattened, e.g. as a builder stage in something else).
RUN find / -xdev -not -path '/proc/*' -not -path '/sys/*' -not -path '/dev/*' \( -uid +65535 -o -gid +65535 \) -exec chown 0:0 {} + 2>/dev/null || true

# Flatten into a single layer. This is what actually fixes the podman-load
# failure above: `COPY --from=builder / /` reads the FULLY MERGED
# filesystem view of the builder stage (whatever chown just normalized it
# to) and writes it as ONE NEW layer with no history of `builder`'s own
# layer boundaries -- so the original bad-UID layer baked into rhel810-dev
# is never part of this image's own layer history at all, and rootless
# Podman never has to extract it. `FROM scratch` (not
# `FROM ${RHEL810_DEV_IMAGE}` again) is required for this to work: staying
# on the same base would just re-inherit its layers underneath the COPY,
# reintroducing the exact problem. All runtime ENV/ARG/ENTRYPOINT below is
# metadata, not filesystem content, so COPY doesn't carry it over --
# re-declared explicitly, matching every ENV set anywhere above (must stay
# in sync with this Dockerfile's builder stage if that ever changes).
FROM scratch AS final
COPY --from=builder / /
ENV PATH=/opt/sambanova/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONNOUSERSITE=1
ENV UCX_MODULE_DIR=/opt/rdu-ucx/lib/ucx
ENV INSTALL_ROOT=/opt/bar2-runtime
ENV LD_LIBRARY_PATH=/opt/bar2-runtime/lib:/opt/sambanova/lib:/opt/rdu-ucx/lib
ARG RDU_ENGINE_TARGET_RUNTIME_VERSION_OVERRIDE
ENV RDU_ENGINE_TARGET_RUNTIME_VERSION_OVERRIDE="${RDU_ENGINE_TARGET_RUNTIME_VERSION_OVERRIDE}"
ENTRYPOINT ["/opt/rdu-decode-entrypoint.sh"]
