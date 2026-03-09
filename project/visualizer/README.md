### Create portable application
```shell
nuitka --mode=onefile .\project\visualizer --output-dir=.\project\visualizer\build --python-flag=-O --python-flag=-S --no-pyi-file --enable-plugin=tk-inter
```
