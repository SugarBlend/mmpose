git clone https://github.com/SugarBlend/-DeployAndServe.git
pip install poetry
cd "-DeployAndServe"
poetry build -f wheel
pip install dist/deploy2serve-0.3.0-py3-none-any.whl
cd ../
pip install -r ./requirements.txt --no-build-isolation
pip install -r ./project/requirements.txt
