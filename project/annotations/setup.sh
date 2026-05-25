set -a
source ../services/.env
set +a

dvc remote add -d minio s3://dvc-cache
dvc remote modify minio endpointurl http://$MINIO_GLOBAL_HOST:9000
dvc remote modify minio access_key_id $MINIO_ROOT_USER
dvc remote modify minio secret_access_key $MINIO_ROOT_PASSWORD
