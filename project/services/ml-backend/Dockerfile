FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        python3.10 python3.10-venv python3-pip \
        build-essential g++ \
        freeglut3-dev libx11-dev libxmu-dev libxi-dev \
        libglu1-mesa libglu1-mesa-dev libfreeimage-dev \
        ffmpeg libsm6 libxext6 \
        libffi-dev python3-dev gcc; \
    apt-get autoremove -y

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/.cache/uv \
    UV_PYTHON=python3.10 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda

COPY requirements-base.txt .
RUN --mount=type=cache,target=/.cache/uv,sharing=locked \
    uv pip install --system -r requirements-base.txt --index-strategy unsafe-best-match

# stage 1: building mmcv wheel
FROM base AS mmcv-builder

RUN --mount=type=cache,target=/.cache/uv,sharing=locked \
    uv pip install --system setuptools wheel chumpy --no-build-isolation

RUN --mount=type=cache,target=/root/.cache/git,sharing=locked \
    git clone --depth=1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git

RUN cd mmcv && \
    MMCV_WITH_OPS=1 python3 setup.py bdist_wheel

# stage 2: install other requirements
FROM base AS final

WORKDIR /app

COPY --from=mmcv-builder /app/mmcv/dist /tmp/mmcv

COPY requirements.txt .
RUN --mount=type=cache,target=/.cache/uv,sharing=locked \
    uv pip install --system -r requirements.txt

COPY requirements-test.txt .
RUN --mount=type=cache,target=/.cache/uv,sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      uv pip install --system -r requirements-test.txt; \
    fi

RUN --mount=type=cache,target=/.cache/uv,sharing=locked \
    uv pip install --system mmdet mmengine openmim \
    mmpose mmpretrain \
    /tmp/mmcv/*.whl
#    sapiens-pretrain@git+https://github.com/facebookresearch/sapiens.git#subdirectory=pretrain \
#    sapiens-pose@git+https://github.com/facebookresearch/sapiens.git#subdirectory=pose \

ENV PORT=${PORT:-9090}

COPY . ./

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
