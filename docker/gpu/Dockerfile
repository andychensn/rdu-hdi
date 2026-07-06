# GPU prefill image for rdu-hdi disaggregated inference.
#
# Layers on top of vllm/vllm-openai:${VLLM_VERSION} (stock image with torch,
# vllm, flashinfer, deepgemm already installed):
#   - UCX (andychensn/ucx) — bnxt_re RoCE RDMA support; branch/commit pinned
#     in config/versions.env, NOT hardcoded here (see ARGs below)
#   - NIXL (andychensn/nixl) — KV cache transfer library; same pinning rule
#   - patches/gpu/nixl_connector.py overlay — replaces vllm's stock
#     nixl_connector.py wholesale, adding REGISTER_CONSUMER_MSG handshake
#     support so the GPU prefill producer can coordinate KV transfer with
#     the RDU decode consumer
#   - ai-dynamo — dynamo.vllm + dynamo.frontend Python modules (no [vllm] extra: that upgrades vllm to 0.20.x)
#
# Build (from login node sc-vnc9):
#   bash docker/gpu/build.sh
#
# Run (inside SLURM job on GPU node):
#   sudo -g docker /usr/bin/cuda-docker-run-wrapper --net=host --rm IMAGE vllm serve ...
#
# All ARGs below are intentionally declared with NO default value.
# config/versions.env is the single source of truth for every pin here;
# docker/gpu/build.sh sources it and passes every value via
# --build-arg. A default here would silently drift from versions.env the
# next time one of them changes and the other doesn't — better to fail
# loudly (empty ref/tag) than build successfully against a stale pin.
# Do not `docker build -f docker/gpu/Dockerfile .` directly; always go through
# docker/gpu/build.sh.
ARG VLLM_VERSION

# ── Stage 1: builder — compiles UCX and NIXL wheel ───────────────────────────
# Uses the same base so CUDA toolkit version matches.
FROM vllm/vllm-openai:${VLLM_VERSION} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make autoconf automake libtool pkg-config git \
        libibverbs-dev librdmacm-dev rdma-core \
    && CUDA_MAJ=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') \
    && apt-get install -y --no-install-recommends cuda-nvml-dev-${CUDA_MAJ} 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

# Build UCX 1.22 with CUDA and bnxt_re verbs support
ARG UCX_COMMIT
ARG UCX_BRANCH
RUN git clone --depth=200 --branch ${UCX_BRANCH} \
        https://github.com/andychensn/ucx.git /tmp/ucx-src && \
    git -C /tmp/ucx-src checkout ${UCX_COMMIT} && \
    # Strip GDAKI (DOCA/mlx5 GPU-Direct-Async) — not needed, avoids build failures
    sed -i 's/^SUBDIRS = \. gdaki/SUBDIRS = ./' \
        /tmp/ucx-src/src/uct/ib/mlx5/Makefile.am 2>/dev/null || true && \
    sed -i '\#m4_include(\[src/uct/ib/mlx5/gdaki/configure.m4\])#d' \
        /tmp/ucx-src/src/uct/ib/mlx5/configure.m4 2>/dev/null || true && \
    rm -rf /tmp/ucx-src/src/uct/ib/mlx5/gdaki && \
    # Strip RDU module — not needed on GPU side
    sed -i '/^SUBDIRS = / s/ rdu / /' /tmp/ucx-src/src/uct/Makefile.am 2>/dev/null || true && \
    sed -i '\#m4_include(\[src/uct/rdu/configure.m4\])#d' \
        /tmp/ucx-src/src/uct/configure.m4 2>/dev/null || true && \
    cd /tmp/ucx-src && \
    autoreconf -fiv && \
    ./configure \
        --prefix=/opt/ucx \
        --enable-shared --disable-static --enable-mt \
        --with-cuda=/usr/local/cuda \
        --without-java --without-go --without-rocm \
        --without-gdrcopy --without-valgrind \
        --without-knem --without-efa --without-mpi \
        --disable-doxygen-doc --enable-optimizations \
        MPICC= && \
    make -j$(nproc) install && \
    ls /opt/ucx/include/ucp/api/ucp.h && echo "UCX headers installed ✅" && \
    rm -rf /tmp/ucx-src && \
    echo "UCX IB transports: $(ls /opt/ucx/lib/ucx/libuct_ib*.so 2>/dev/null | wc -l)"

# Build NIXL wheel with UCX and CUDA backends
ARG NIXL_COMMIT
ARG NIXL_BRANCH
RUN pip install meson-python pybind11 patchelf pyyaml setuptools build wheel && \
    git clone --depth=50 --branch ${NIXL_BRANCH} \
        https://github.com/andychensn/nixl.git /tmp/nixl-src && \
    git -C /tmp/nixl-src checkout ${NIXL_COMMIT} && \
    mkdir -p /tmp/nixl-wheels && \
    LIBRARY_PATH=/opt/ucx/lib \
    LD_LIBRARY_PATH=/opt/ucx/lib \
    PKG_CONFIG_PATH=/opt/ucx/lib/pkgconfig \
    pip wheel /tmp/nixl-src --no-deps -w /tmp/nixl-wheels \
        --config-settings=setup-args="-Ducx_path=/opt/ucx" \
        --config-settings=setup-args="-Denable_plugins=UCX" \
        --config-settings=setup-args="-Dcuda_path=/usr/local/cuda" && \
    rm -rf /tmp/nixl-src

# ── Stage 2: final image — stock vllm + UCX libs + NIXL + patch ──────────────
FROM vllm/vllm-openai:${VLLM_VERSION} AS final

# Copy UCX install (shared libs needed at runtime by NIXL)
COPY --from=builder /opt/ucx /opt/ucx
# Copy NIXL wheel
COPY --from=builder /tmp/nixl-wheels /tmp/nixl-wheels

ENV LD_LIBRARY_PATH=/opt/ucx/lib:${LD_LIBRARY_PATH:-}
ENV UCX_MODULE_DIR=/opt/ucx/lib/ucx

# Install NIXL wheel (explicit variable avoids silent multi-match on glob)
RUN NIXL_WHL=$(ls /tmp/nixl-wheels/nixl_cu12-*.whl | head -1) && \
    pip install "$NIXL_WHL" && \
    rm -rf /tmp/nixl-wheels

# Install Dynamo Python modules (dynamo.vllm, dynamo.frontend)
# Use base package WITHOUT [vllm] extra — the [vllm] extra upgrades vllm to 0.20.1
# which breaks MiniMax-M2.7. The dynamo.vllm module is in the base ai-dynamo package.
ARG DYNAMO_VERSION
RUN pip install "ai-dynamo==${DYNAMO_VERSION}" "ai-dynamo-runtime==${DYNAMO_VERSION}"

# Overlay nixl_connector.py with REGISTER_CONSUMER_MSG handshake support
# (md5 5cd6b474372f1d8490e380a82d1fd226).
COPY patches/gpu/nixl_connector.py /tmp/nixl_connector.py
RUN VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))") && \
    NIXL_CONN=$(find "$VLLM_DIR" -path "*kv_connector/v1/nixl_connector.py" 2>/dev/null | head -1) && \
    echo "nixl_connector.py: $NIXL_CONN" && \
    [ -n "$NIXL_CONN" ] || { echo "ERROR: nixl_connector.py not found in vllm package"; exit 1; } && \
    cp /tmp/nixl_connector.py "$NIXL_CONN" && \
    grep -q "REGISTER_CONSUMER_MSG" "$NIXL_CONN" && echo "REGISTER_CONSUMER_MSG present ✅" && \
    python3 -c "import vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector" && \
    echo "producer nixl_connector.py overlaid + imports OK" && \
    rm /tmp/nixl_connector.py

# Build Broadcom's OOT libbnxt_re-237.1.137.0 userspace library.
# This matches the OOT bnxt_re kernel driver's UVERBS API exactly — upstream rdma-core's
# libbnxt_re sends wrong ioctl attributes, causing EINVAL. The Broadcom source includes
# rc-compat/v39 so it builds against Ubuntu 22.04's rdma-core 39 (libibverbs rdmav34 ABI).
# Source: /import/it-tools/idc/fw/brcm/237/bcm_237.1.148.0a/drivers_linux/bnxt_rocelib/
COPY .rdma-libs/libbnxt_re-237.1.137.0.tar.gz /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake libtool make \
        libnl-3-200 libnl-3-dev libnl-route-3-200 libnl-route-3-dev \
    && rm -rf /var/lib/apt/lists/* && \
    cd /tmp && tar xzf libbnxt_re-237.1.137.0.tar.gz && \
    cd libbnxt_re-237.1.137.0 && \
    bash autogen.sh && \
    ./configure --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --with-rdma-core-version=39 && \
    make -j$(nproc) && \
    cp src/.libs/libbnxt_re-rdmav34.so \
       /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so && \
    ls -la /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so && \
    rm -rf /tmp/libbnxt_re-237.1.137.0 /tmp/libbnxt_re-237.1.137.0.tar.gz && \
    echo "Broadcom OOT libbnxt_re built for Ubuntu rdmav34 ✅"

# Patch dynamo.vllm.multimodal_utils.protocol for vllm 0.16.0 compatibility.
# ai-dynamo 1.2.1 imports MultiModalUUIDDict which was added in vllm 0.20.x.
RUN python3 -c "import pathlib; p=pathlib.Path('/usr/local/lib/python3.12/dist-packages/dynamo/vllm/multimodal_utils/protocol.py'); t=p.read_text() if p.exists() else ''; p.write_text(t.replace('from vllm.inputs import MultiModalUUIDDict, TokensPrompt  # noqa: F401', 'try:\n    from vllm.inputs import MultiModalUUIDDict, TokensPrompt\nexcept ImportError:\n    MultiModalUUIDDict=dict\n    TokensPrompt=dict')) if p.exists() else None; print('protocol.py patched' if p.exists() else 'not found')"

# Validate the image at build time
RUN python3 -c "import vllm; print('vllm:', vllm.__version__)" && \
    python3 -c "from nixl._api import nixl_agent; print('nixl: OK')" && \
    python3 -c "import dynamo.vllm; print('dynamo.vllm: OK')" && \
    python3 -c "import dynamo.frontend; print('dynamo.frontend: OK')"
