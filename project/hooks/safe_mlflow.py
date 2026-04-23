import re
from mmengine.visualization import MLflowVisBackend
from mmengine.registry import VISBACKENDS
from typing import Optional


@VISBACKENDS.register_module()
class SafeMLflowVisBackend(MLflowVisBackend):

    @staticmethod
    def _sanitize_key(key: str) -> str:
        # 'coco/AP (M)' → 'coco/AP_M'
        # 'coco/AR .5' → 'coco/AR_50'
        key = key.replace(' (M)', '_M')
        key = key.replace(' (L)', '_L')
        key = re.sub(r'[^\w\s\-./]', '_', key)
        return key.strip('_')

    def add_scalars(self, scalar_dict: dict, step: int = 0, file_path: Optional[str] = None, **kwargs) -> None:
        safe_dict = {self._sanitize_key(k): v for k, v in scalar_dict.items()}
        super().add_scalars(safe_dict, step, file_path, **kwargs)
