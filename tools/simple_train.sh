#!/usr/bin/env bash

set -a
source $(pwd)/tools/.env
set +a

python tools/train.py
