FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    pkg-config \
    libcairo2-dev \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/.cache/uv \
    UV_PYTHON=python3.10 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda

COPY . ./

RUN --mount=type=cache,target=/root/.cache/uv \
    bash project/create_env.sh

RUN export PATH=$PATH:./venv/bin

ENTRYPOINT bash
