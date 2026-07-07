import cv2
import os
import numpy as np
import boto3
import botocore.client


class S3Reader(object):
    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=botocore.client.Config(signature_version="s3v4"),
        )

    def fetch(self, path: str, bucket_name: str) -> np.ndarray:
        data = self.client.get_object(Bucket=bucket_name, Key=path)["Body"].read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
