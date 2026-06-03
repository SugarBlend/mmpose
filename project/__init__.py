import torch

_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from .hooks.minio_backend import MinIOBackend
from .hooks.unfreeze_backbone_hook import StageUnfreezeHook
from .hooks.model_registry import MLflowModelRegistryHook
from .hooks.safe_mlflow import SafeMLflowVisBackend
