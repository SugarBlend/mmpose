import os
import argparse
import torch
import torch.nn.parallel.distributed as _ddp_mod
import torch.distributed.distributed_c10d as _c10d
import logging
from pathlib import Path
import io
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from urllib.parse import urlparse


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
_c10d._get_default_group = lambda: None


def _ddp_patched_setstate(self, state):
    state.pop("process_group", None)
    self.__dict__.update(state)


_ddp_mod.DistributedDataParallel.__setstate__ = _ddp_patched_setstate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", default="minio://mlflow-artifacts/6/fce433ded8ff4012bc96e5f6a7a36214/artifacts/checkpoints/epoch_031/data/model.pth", help="Path to the weights that were obtained during distributed learning.")
    parser.add_argument("-d", "--destination", default="./", help="Save path on local disk.")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.joinpath("../../tools/.env"))
    args = parse_args()
    if any(args.source.startswith(scheme) for scheme in ["s3://", "minio://"]):
        parsed = urlparse(args.source)
        scheme = parsed.scheme
        bucket = parsed.netloc
        path = parsed.path.lstrip("/")

        client = boto3.client(
            "s3",
            endpoint_url=os.environ["AWS_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4"),
        )
        logger.info(f"Try to fetch file from S3 object storage, endpoint url: {os.environ['AWS_ENDPOINT_URL']}")
        response = client.get_object(Bucket=bucket, Key=path)
        data = response["Body"].read()
        f = io.BytesIO(data)
    elif os.path.exists(args.source):
        f = args.source
    else:
        raise RuntimeError("If the file is from the local file system, check the path; it's likely incorrect. "
                           "If you're connected to object storage, check the connection details—login, password, and URL.")

    ckpt = torch.load(f, map_location="cpu", weights_only=False)

    if isinstance(ckpt, _ddp_mod.DistributedDataParallel):
        ckpt = {"state_dict": ckpt.module.state_dict()}
    elif isinstance(ckpt, dict):
        sd = ckpt.get("state_dict", ckpt)
        if any(k.startswith("module.") for k in sd):
            ckpt["state_dict"] = {k.removeprefix("module."): v for k, v in sd.items()}

    torch.save(ckpt, Path(args.destination).joinpath(Path(args.source).name))
    logger.info(f"Saved clean checkpoint to {args.destination}")
