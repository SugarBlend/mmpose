FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        python3.10 \
        python3.10-venv \
        python3-pip \
        build-essential \
        g++ \
        freeglut3-dev \
        libx11-dev \
        libxmu-dev \
        libxi-dev \
        libglu1-mesa \
        libglu1-mesa-dev \
        libfreeimage-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        libffi-dev \
        python3-dev \
        gcc; \
    apt-get autoremove -y

# update pip
RUN python3 -m pip install --upgrade pip setuptools wheel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    CUDA_HOME=/usr/local/cuda

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install model requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

# MMCV building
RUN git clone --depth=1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    pip3 install chumpy --no-build-isolation && \
    MMCV_WITH_OPS=1 python3 setup.py bdist_wheel && \
    pip3 install mmdet mmpose mmengine openmim dist/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl && \
    cd .. && rm -rf mmcv

COPY . ./

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
