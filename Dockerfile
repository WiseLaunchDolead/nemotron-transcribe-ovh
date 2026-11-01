# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS builder
ARG NEMO_REVISION=20999cc921d2b14aa58708f61756a8b2e787ba20
ARG BUILD_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake ninja-build build-essential pkg-config libsentencepiece-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init . && git remote add origin https://github.com/NVIDIA/NeMo-Speech.cpp.git \
    && git fetch --depth 1 origin "${NEMO_REVISION}" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${NEMO_REVISION}" \
    && git submodule update --init --depth 1 llama.cpp third_party/cpp-httplib
# L40S: compute capability 8.9. No GPU is needed to compile on GitHub Actions.
RUN scripts/configure.sh cuda-server \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF \
    -DNEMO_SPEECH_BUILD_TTS=OFF -DNEMO_SPEECH_BUILD_NMT=OFF \
    -DNEMO_SPEECH_BUILD_DIAR=OFF -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/nemo -DCMAKE_INSTALL_LIBDIR=lib \
    && cmake --build --preset cuda-server --parallel "${BUILD_JOBS}" \
    && cmake --install build/cuda-server

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867 AS service
LABEL org.opencontainers.image.source="https://github.com/WiseLaunchDolead/nemotron-transcribe-ovh" \
    org.opencontainers.image.description="Nemotron Transcribe IQ4_NL prototype for OVHcloud L40S"
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl python3 libsentencepiece0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 42420 ovh && useradd --uid 42420 --gid 42420 --home-dir /workspace ovh \
    && mkdir -p /workspace/models /workspace/.cache && chown -R 42420:42420 /workspace
COPY --from=builder /opt/nemo /opt/nemo
COPY --chmod=755 start.sh /opt/app/start.sh
COPY scripts/download_models.py models.lock.json config.yaml /opt/app/
ENV PATH="/opt/nemo/bin:${PATH}" LD_LIBRARY_PATH="/opt/nemo/lib:${LD_LIBRARY_PATH}" \
    XDG_CACHE_HOME=/workspace/.cache MODEL_DIR=/workspace/models \
    NEMO_SPEECH_HTTP_HOST=0.0.0.0 NEMO_SPEECH_HTTP_PORT=8080
WORKDIR /workspace
USER 42420:42420
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD curl --fail --silent --max-time 4 http://127.0.0.1:8080/ready || exit 1
ENTRYPOINT ["/opt/app/start.sh"]

# CI-only target: the real CUDA driver is injected by OVH on a GPU host.
# A toolkit stub lets a CPU runner exercise CLI parsing without pretending to infer.
# Never publish this target: production must resolve the real driver.
FROM service AS verify
COPY --from=builder /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1
ENV LD_LIBRARY_PATH="/opt/cuda-stubs:${LD_LIBRARY_PATH}"

# Keep the safe, real-driver image as the default Docker build target.
FROM service AS runtime
