#!/bin/bash

uv venv --python 3.10
source .venv/bin/activate

rm -rf ./-DeployAndServe
git clone https://github.com/SugarBlend/-DeployAndServe.git

uv pip install poetry wheel

cd ./-DeployAndServe

poetry config virtualenvs.create false --local
poetry build -f wheel

uv pip install dist/*.whl

cd ../

uv pip install -r ./project/pyproject.toml --index-strategy unsafe-best-match
uv pip install -r ./requirements.txt --no-build-isolation
uv pip install . --no-build-isolation
