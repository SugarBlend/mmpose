#!/bin/bash

export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv

uv venv --python 3.10 --clear
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
else
    echo "Virtualenv activation script not found"
    exit 1
fi

uv pip install poetry wheel

# DeployAndServe consider export for Sapiens and completed tensorrt runner
if [ ! -d ./-DeployAndServe ]; then
  git clone https://github.com/SugarBlend/-DeployAndServe.git
fi

file=$(ls ./-DeployAndServe/dist/*.whl 2>/dev/null | head -n 1)
if [ -z "$file" ]; then
  poetry config virtualenvs.create false --local
  poetry build -f wheel -P ./-DeployAndServe
fi

uv sync --project ./project --index-strategy unsafe-best-match --all-groups
uv pip install -r ./requirements.txt --no-build-isolation
uv pip install . --no-build-isolation
uv pip install ./-DeployAndServe/dist/*.whl --no-deps

uv pip install git+https://github.com/HumanSignal/label-studio-ml-backend.git#egg=label-studio-ml-backend
uv pip install "numpy<2"
