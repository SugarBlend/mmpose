from xtcocotools.cocoeval import COCOeval
from mmpose.evaluation.metrics import CocoWholeBodyMetric
import numpy as np
from typing import Any


def _do_python_keypoint_eval(self, outfile_prefix: str) -> dict[str, Any]:
    stats_names = [
        "AP", "AP .5", "AP .75", "AP (M)", "AP (L)",
        "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"
    ]

    res_file = f"{outfile_prefix}.keypoints.json"
    coco_det = self.coco.loadRes(res_file)
    sigmas = self.dataset_meta["sigmas"]

    cuts = np.cumsum([
        0, self.body_num, self.foot_num, self.face_num,
        self.left_hand_num, self.right_hand_num
    ])
    cache = {}

    parts = [
        ("keypoints_body", cuts[0], cuts[1]),
        ("keypoints_foot", cuts[1], cuts[2]),
        ("keypoints_face", cuts[2], cuts[3]),
        ("keypoints_lefthand", cuts[3], cuts[4]),
        ("keypoints_righthand", cuts[4], cuts[5]),
        ("keypoints_wholebody", 0, len(sigmas)),
    ]

    for iou_type, lo, hi in parts:
        sig = sigmas[lo:hi] if iou_type != "keypoints_wholebody" else sigmas
        coco_eval = COCOeval(self.coco, coco_det, iou_type, sig, use_area=self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        cache[coco_eval.params.iouType] = dict(zip(stats_names, coco_eval.stats))

    return cache

CocoWholeBodyMetric._do_python_keypoint_eval = _do_python_keypoint_eval
