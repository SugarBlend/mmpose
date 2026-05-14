import argparse
import boto3
import os
from pathlib import Path
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def _download_file(client, bucket_name: str, key: str, local_path: str, pbar: tqdm) -> str:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    if os.path.exists(local_path):
        return f"Skip: {local_path}"

    client.download_file(bucket_name, key, local_path,
                         Callback=lambda b: pbar.update(b))
    return f"Done: {local_path}"


def download_annotations(
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    prefix: str = "annotations/",
    local_dir: str = "/tmp/annotations",
    workers: int = 8,
):
    os.makedirs(local_dir, exist_ok=True)

    def make_client():
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )

    client = make_client()
    paginator = client.get_paginator("list_objects_v2")
    pages = list(paginator.paginate(Bucket=bucket_name, Prefix=prefix))
    objects = [
        (obj["Key"], obj["Size"])
        for page in pages
        for obj in page.get("Contents", [])
    ]

    if not objects:
        print("No objects found")
        return

    total_size = sum(size for _, size in objects)

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading", ncols=80) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_download_file, make_client(), bucket_name, key,
                                os.path.join(local_dir, Path(key).name), pbar): key
                for key, _ in objects
            }
            for future in as_completed(futures):
                print(future.result())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data fetcher from any s3 object storage service")
    parser.add_argument("--access_key", "-c", type=str, required=True, help="Access key for S3 object storage.")
    parser.add_argument("--secret_key", "-o", type=str, required=True, help="Secret key for S3 object storage.")
    parser.add_argument("--bucket_name", type=str, required=True, help="Bucket name for S3 object storage.")
    parser.add_argument("--endpoint_url", type=str, required=True, help="Service address: <host>:<port>")
    parser.add_argument("--prefix", type=str, required=True, help="Path relative to bucket_name for data")
    parser.add_argument("--local_dir", type=str, default="./", help="Path to local directory for storing "
                                                                    "downloaded data")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_annotations(
        endpoint_url=args.endpoint_url,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket_name=args.bucket_name,
        prefix=args.prefix,
        local_dir=args.local_dir
    )