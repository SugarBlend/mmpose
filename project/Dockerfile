FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    pkg-config \
    libcairo2-dev \
    gcc \
    g++ \
    nano \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON=python3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda

COPY . ./

RUN --mount=type=cache,target=/root/.cache/uv \
    bash project/create_env.sh

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/.venv/bin:${PATH}"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libcairo2 nano \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/.venv /workspace/.venv
COPY --from=builder /workspace/project /workspace/project
COPY --from=builder /workspace/tools /workspace/tools
COPY --from=builder /workspace/configs /workspace/configs
COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python

ENTRYPOINT ["bash"]
