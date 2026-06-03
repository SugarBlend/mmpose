from __future__ import annotations

import numpy as np
from xtcocotools.cocoeval import COCOeval
from mmpose.evaluation.metrics.coco_wholebody_metric import CocoWholeBodyMetric as CocoWholeBodyMetricBase
from mmpose.evaluation.metrics import CocoMetric as CocoMetricBase
from typing import Any


class CocoWholeBodyMetric(CocoWholeBodyMetricBase):
    def __init__(self, *args, iou_thrs: list[float] | np.ndarray | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.iou_thrs: list[float] | np.ndarray = iou_thrs or np.linspace(0.5, 0.95, 10)

        self.cache: dict[str, Any] = {}
        self.stats_names = [
            "AP", "AP .5", "AP .75", "AP (M)", "AP (L)",
            "AR", "AR .5", "AR .75", "AR (M)", "AR (L)",
        ]

    def _run_coco_eval(self, coco_det, iou_type, sigmas) -> None:
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            iou_type,
            sigmas,
            use_area=self.use_area,
        )
        coco_eval.params.useSegm = None
        coco_eval.params.iouThrs = np.array(self.iou_thrs)  # <-- патч здесь
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.cache[coco_eval.params.iouType] = dict(zip(self.stats_names, coco_eval.stats))

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> dict[str, Any]:
        res_file = f"{outfile_prefix}.keypoints.json"
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta["sigmas"]

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num,
            self.left_hand_num, self.right_hand_num
        ])

        parts = [
            ("keypoints_body", sigmas[cuts[0]:cuts[1]]),
            ("keypoints_foot", sigmas[cuts[1]:cuts[2]]),
            ("keypoints_face", sigmas[cuts[2]:cuts[3]]),
            ("keypoints_lefthand", sigmas[cuts[3]:cuts[4]]),
            ("keypoints_righthand", sigmas[cuts[4]:cuts[5]]),
            ("keypoints_wholebody", sigmas),
        ]

        for iou_type, part_sigmas in parts:
            self._run_coco_eval(coco_det, iou_type, part_sigmas)

        return self.cache


class CocoMetric(CocoMetricBase):
    def __init__(self, *args, iou_thrs: list[float] | np.ndarray | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_thrs = np.array(iou_thrs) if iou_thrs is not None else np.linspace(0.5, 0.95, 10)

        if self.iou_type == "keypoints_crowd":
            self.stats_names = ["AP", "AP .5", "AP .75", "AR", "AR .5", "AR .75", "AP(E)", "AP(M)", "AP(H)"]
        else:
            self.stats_names = ["AP", "AP .5", "AP .75", "AP (M)", "AP (L)", "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"]


    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        res_file = f"{outfile_prefix}.keypoints.json"
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta["sigmas"]

        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas, self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.params.iouThrs = self.iou_thrs

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return list(zip(self.stats_names, coco_eval.stats))
