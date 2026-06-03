#!/bin/bash

export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv

uv venv --python 3.12 --clear
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
else
    echo "Virtualenv activation script not found"
    exit 1
fi

uv pip install poetry wheel setuptools==79.0.1

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

if uv run python -c "import mmcv; assert mmcv.__version__ == '2.1.0'" 2>/dev/null; then
    echo "mmcv 2.1.0 already installed, skipping build"
else
  export MMCV_WITH_OPS=1
  git clone https://github.com/open-mmlab/mmcv.git
  cd mmcv
  git checkout v2.1.0
  uv run python setup.py bdist_wheel
  uv pip install ./dist/mmcv-2.1.0*.whl
  cd ..
fi

uv pip install git+https://github.com/HumanSignal/label-studio-ml-backend.git#egg=label-studio-ml-backend
uv pip install "numpy<2"
uv pip install transformers==4.50.0 huggingface-hub<1
