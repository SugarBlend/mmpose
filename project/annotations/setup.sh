set -a
source $(pwd)/project/services/.env
set +a

dvc remote add -d -f minio s3://dvc-cache
dvc remote modify --local minio endpointurl http://$MINIO_GLOBAL_HOST:9000
dvc remote modify --local minio access_key_id $MINIO_ROOT_USER
dvc remote modify --local minio secret_access_key $MINIO_ROOT_PASSWORD
