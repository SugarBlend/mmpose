#!/usr/bin/env bash

set -a
. $(pwd)/tools/.env
set +a

python tools/train.py
