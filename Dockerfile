# Dockerfile for strandsdecider (https://github.com/strands-labs/strands-decider)
#
# Build:
#   docker build -t strandsdecider .
#   docker build --build-arg STRANDS_DECIDER_EXTRAS="[vision]" -t strandsdecider:vision .
#   docker build --build-arg STRANDS_DECIDER_GIT=1 -t strandsdecider:git .
#   docker build --build-arg INSTALL_BITSANDBYTES=1 -t strandsdecider:q8 .
#   docker build --build-arg INSTALL_IPEX=1 -t strandsdecider:ipex .
#   docker build --build-arg INSTALL_IPEX=1 --build-arg INSTALL_IPEX_QUANT=1 -t strandsdecider:ipex-quant .
#
# Run (server):
#   docker run -p 8000:8000 strandsdecider
#   docker run -p 8000:8000 -e MODEL_ID=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 strandsdecider
#
# Run (one-off ask):
#   docker run --rm strandsdecider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
#     --state "..." --choice "Which team?=billing,sales,retail"

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# System deps are intentionally minimal: torch/transformers publish manylinux
# wheels, so no CUDA toolkit or compiler toolchain is needed for the default
# (CPU) image. Add a Dockerfile.cuda variant if you need flash-linear-attention.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Select optional pip extras (e.g. "[vision]") and/or install from git main
# (needed for Gemma 4, which is not yet on PyPI).
ARG STRANDS_DECIDER_EXTRAS=""
ARG STRANDS_DECIDER_GIT=0

# Install bitsandbytes so strands-decider can load models in 8-bit or 4-bit
# (pass --load-in-8bit / --load-in-4bit via STRANDS_ARGS at runtime). Note:
# bitsandbytes quantization is primarily designed for CUDA GPUs; CPU-only
# quantized inference is limited/slower.
ARG INSTALL_BITSANDBYTES=0

# Install Intel Extension for PyTorch (IPEX) for CPU optimizations
# (AVX2/VNNI, BF16, graph optimization). Provides ~1.5-2x speedup on
# Intel CPUs without requiring a discrete GPU.
ARG INSTALL_IPEX=0

# Install Intel Neural Compressor (INC) for CPU quantization with IPEX.
# Enables INT8 static/dynamic quantization on CPU. Requires INSTALL_IPEX=1.
# Note: quantization is applied at model load time via STRANDS_ARGS
# (--ipex-quantize=int8_static|int8_dynamic) if strands-decider supports it.
ARG INSTALL_IPEX_QUANT=0

# Install CPU-only torch first (PyTorch's CPU index) so strands-decider's
# `torch>=2.7` dependency resolves to a small CPU wheel instead of the ~3 GB
# CUDA toolkit bundle that the default torch wheel pulls in on Linux.
# strands-decider runs inference entirely on CPU by default.
# When INSTALL_IPEX=1, pin torch to 2.7.0 (minimum required) for IPEX compatibility.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.7.0" \
    && if [ "$INSTALL_IPEX" = "1" ]; then \
        pip install --no-cache-dir --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/ \
            intel-extension-for-pytorch==2.7.0; \
    fi \
    && if [ "$INSTALL_IPEX_QUANT" = "1" ]; then \
        pip install --no-cache-dir neural-compressor py-cpuinfo prettytable; \
    fi \
    && if [ "$INSTALL_BITSANDBYTES" = "1" ]; then \
        pip install --no-cache-dir bitsandbytes; \
    fi \
    && if [ "$STRANDS_DECIDER_GIT" = "1" ]; then \
        pip install --no-cache-dir "strands-decider @ git+https://github.com/strands-labs/strands-decider${STRANDS_DECIDER_EXTRAS}"; \
    else \
        pip install --no-cache-dir "strands-decider${STRANDS_DECIDER_EXTRAS}"; \
    fi

WORKDIR /app

# Copy custom IPEX entrypoint
COPY serve_ipex.py /app/serve_ipex.py
RUN chmod +x /app/serve_ipex.py

# Default: serve the default model on 0.0.0.0:8000. Override MODEL_ID and
# STRANDS_ARGS at runtime (see docker-compose.yml). The entrypoint is the
# strands-decider CLI so `ask` one-offs work via `docker run --rm ... ask ...`.
ENTRYPOINT ["strands-decider"]
CMD ["serve", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", "--port", "8000", "--host", "0.0.0.0"]

EXPOSE 8000
