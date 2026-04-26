#!/usr/bin/env bash

set -a
source $(pwd)/tools/.env
set +a

mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root s3://mlflow-artifacts \
--host 0.0.0.0 \
--port 8082
