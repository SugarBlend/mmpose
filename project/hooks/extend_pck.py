import numpy as np
from mmpose.registry import METRICS
from mmpose.evaluation.metrics.keypoint_2d_metrics import (PCKAccuracy, Sequence, Optional, Dict, Union,
                                                           keypoint_pck_accuracy)


@METRICS.register_module()
class GroupedPCKAccuracy(PCKAccuracy):
    def __init__(
        self,
        thr: float = 0.05,
        norm_item: Union[str, Sequence[str]] = "bbox",
        keypoint_groups: Optional[Dict[str, Sequence[int]]] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None
    ) -> None:
        super().__init__(thr=thr, norm_item=norm_item,
                         collect_device=collect_device, prefix=prefix)
        self.keypoint_groups = keypoint_groups or {}
        self.count = 0

    def compute_metrics(self, results: list) -> Dict[str, float]:
        pred_coords = np.concatenate([result["pred_coords"] for result in results])
        gt_coords = np.concatenate([result["gt_coords"] for result in results])
        mask = np.concatenate([result["mask"] for result in results])

        metrics = dict()

        def _eval(norm_key: str, norm_name: str, mask: np.ndarray, suffix: str = "") -> None:
            norm_size = np.concatenate([result[norm_key] for result in results])
            _, pck, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask, self.thr, norm_size)
            metrics[f"{norm_name}{suffix}"] = pck

        if "bbox" in self.norm_item:
            _eval("bbox_size", "PCK", mask)

        for group_name, indices in self.keypoint_groups.items():
            group_mask = mask.copy()
            keep = np.zeros(mask.shape[1], dtype=bool)
            keep[indices] = True
            group_mask = group_mask & keep[None, :]
            _eval("bbox_size", "PCK", group_mask, group_name)
        return metrics
