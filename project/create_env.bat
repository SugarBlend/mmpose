uv venv --python 3.10
git clone https://github.com/SugarBlend/-DeployAndServe.git
uv pip install poetry wheel
cd "./-DeployAndServe"
poetry config virtualenvs.create false
poetry build -f wheel
uv pip install dist/deploy2serve-0.3.0-py3-none-any.whl
cd ../
uv pip install -r ./project/pyproject.toml --index-strategy unsafe-best-match
uv pip install -r ./requirements.txt --no-build-isolation
uv pip install . --no-build-isolation
