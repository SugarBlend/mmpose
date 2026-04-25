#!/bin/bash

export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv

uv venv --python 3.10 --clear
source .venv/bin/activate || source .venv/Scripts/activate

uv pip install poetry wheel

# DeployAndServe consider export for Sapiens and completed tensorrt runner
rm -rf ./-DeployAndServe
git clone https://github.com/SugarBlend/-DeployAndServe.git
poetry config virtualenvs.create false --local
poetry build -f wheel -P ./-DeployAndServe

uv sync --project ./project --index-strategy unsafe-best-match --all-groups
uv pip install -r ./requirements.txt --no-build-isolation
uv pip install . --no-build-isolation
uv pip install ./-DeployAndServe/dist/*.whl --no-deps
