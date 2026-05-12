import os
from pathlib import Path
import boto3
import botocore.client
from botocore.client import Config
from mmengine.fileio.backends import BaseStorageBackend
from mmengine.fileio.backends.registry_utils import register_backend
from typing import Union, Tuple, Optional, Iterator


@register_backend("minio", prefixes=["minio"])
class MinIOBackend(BaseStorageBackend):
    def __init__(
        self,
        endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None,
        bucket_name: str = None,
    ):
        self.bucket_name = bucket_name or os.environ.get("AWS_DEFAULT_BUCKET")
        self._endpoint_url = endpoint_url or os.environ.get("AWS_ENDPOINT_URL")
        self._access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
        self._secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")

        self._client: botocore.client.BaseClient | None = None

    def _get_client(self) -> botocore.client.BaseClient:
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    @staticmethod
    def _clean(filepath: str) -> str:
        return Path(filepath.removeprefix("minio://")).as_posix().lstrip("/")

    def get(self, filepath: str) -> bytes:
        response = self._get_client().get_object(Bucket=self.bucket_name, Key=self._clean(filepath))
        return response["Body"].read()

    def remove(self, filepath: str) -> None:
        self._get_client().delete_object(Bucket=self.bucket_name, Key=self._clean(filepath))

    def get_text(self, filepath: str, encoding="utf-8") -> str:
        return self.get(filepath).decode(encoding)

    def put(self, obj: bytes, filepath: str) -> None:
        self._get_client().put_object(Bucket=self.bucket_name, Key=self._clean(filepath), Body=obj)

    def put_text(self, obj: str, filepath: str, encoding="utf-8") -> None:
        self.put(obj.encode(encoding), filepath)

    def exists(self, filepath: str) -> bool:
        pass

    def isfile(self, filepath: str) -> bool:
        pass

    def isdir(self, filepath: str) -> bool:
        pass

    def join_path(self, filepath: str, *filepaths: str) -> str:
        prefix = "minio://" if filepath.startswith("minio://") else ""
        parts = [self._clean(filepath).rstrip("/")] + [self._clean(f) for f in filepaths if f]
        return prefix + "/".join(parts)

    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False
    ) -> Iterator[str]:
        pass
