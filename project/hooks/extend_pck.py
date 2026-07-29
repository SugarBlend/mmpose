import numpy as np
from mmpose.registry import METRICS, TRANSFORMS
from mmpose.evaluation.metrics.keypoint_2d_metrics import (PCKAccuracy, Sequence, Optional, Dict, Union,
                                                           keypoint_pck_accuracy)


@METRICS.register_module()
class GroupedPCKAccuracy(PCKAccuracy):
    def __init__(
        self,
        thr: float = 0.05,
        norm_item: Union[str, Sequence[str]] = "bbox",
        keypoint_groups: Optional[Dict[str, Sequence[int]]] = None,
        gt_converter: Optional[dict] = None,
        pred_converter: Optional[dict] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None
    ) -> None:
        super().__init__(thr=thr, norm_item=norm_item,
                         collect_device=collect_device, prefix=prefix)
        self.keypoint_groups = keypoint_groups or {}
        self.count = 0
        self.gt_converter = TRANSFORMS.build(gt_converter) if gt_converter is not None else None
        self.pred_converter = TRANSFORMS.build(pred_converter) if pred_converter is not None else None

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']

            pred_coords = np.array(pred['keypoints'])
            gt_coords = np.array(gt['keypoints'])
            gt_vis = np.array(gt['keypoints_visible'])

            pred_vis = np.array(
                pred.get('keypoints_visible', np.ones(pred_coords.shape[:2]))
            )

            if self.pred_converter is not None:
                converted = self.pred_converter(
                    dict(keypoints=pred_coords, keypoints_visible=pred_vis)
                )
                pred_coords = converted['keypoints']
                pred_vis = converted['keypoints_visible']

            if self.gt_converter is not None:
                converted = self.gt_converter(
                    dict(keypoints=gt_coords, keypoints_visible=gt_vis)
                )
                gt_coords = converted['keypoints']
                gt_vis = converted['keypoints_visible']

            mask = gt_vis.astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            if 'bbox' in self.norm_item:
                assert 'bboxes' in gt, (
                    'The ground truth data info does not have the expected '
                    'normalized_item ``"bbox"``.'
                )
                bbox_size_ = np.max(gt['bboxes'][0][2:] - gt['bboxes'][0][:2])
                result['bbox_size'] = np.array(
                    [bbox_size_, bbox_size_]
                ).reshape(-1, 2)

            self.results.append(result)

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
