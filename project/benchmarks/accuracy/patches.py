import numpy as np
from xtcocotools.cocoeval import COCOeval
from mmpose.evaluation.metrics.coco_wholebody_metric import CocoWholeBodyMetric as CocoWholeBodyMetricBase
from mmpose.evaluation.metrics import CocoMetric as CocoMetricBase
from typing import Any


def _summarize_stat(
    coco_eval: COCOeval,
    ap: int,
    iou_thr: float | None = None,
    area_rng: str = "all",
    max_dets: int = 20,
) -> float:
    p = coco_eval.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]

    if ap == 1:
        s = coco_eval.eval["precision"]  # dims: [T, R, K, A, M]
        if iou_thr is not None:
            t = np.where(np.isclose(p.iouThrs, iou_thr))[0]
            if len(t) == 0:
                return -1.0
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        s = coco_eval.eval["recall"]  # dims: [T, K, A, M]
        if iou_thr is not None:
            t = np.where(np.isclose(p.iouThrs, iou_thr))[0]
            if len(t) == 0:
                return -1.0
            s = s[t]
        s = s[:, :, aind, mind]

    valid = s[s > -1]
    return float(np.mean(valid)) if len(valid) else -1.0


def summarize_dynamic(
    coco_eval: COCOeval,
    iou_thrs: list[float] | np.ndarray,
    max_dets: int = 20,
) -> tuple[list[str], np.ndarray]:
    iou_thrs = list(iou_thrs)

    names: list[str] = []
    stats: list[float] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        stats.append(value)

    # overall mean AP/AR across all thresholds (same semantics as before)
    add("AP", _summarize_stat(coco_eval, 1, max_dets=max_dets))
    for thr in iou_thrs:
        add(f"AP {thr:.3g}", _summarize_stat(coco_eval, 1, iou_thr=thr, max_dets=max_dets))
    add("AP (M)", _summarize_stat(coco_eval, 1, area_rng="medium", max_dets=max_dets))
    add("AP (L)", _summarize_stat(coco_eval, 1, area_rng="large", max_dets=max_dets))

    add("AR", _summarize_stat(coco_eval, 0, max_dets=max_dets))
    for thr in iou_thrs:
        add(f"AR {thr:.3g}", _summarize_stat(coco_eval, 0, iou_thr=thr, max_dets=max_dets))
    add("AR (M)", _summarize_stat(coco_eval, 0, area_rng="medium", max_dets=max_dets))
    add("AR (L)", _summarize_stat(coco_eval, 0, area_rng="large", max_dets=max_dets))

    stats_arr = np.array(stats)
    coco_eval.stats = stats_arr
    return names, stats_arr


class CocoWholeBodyMetric(CocoWholeBodyMetricBase):
    def __init__(self, *args, iou_thrs: list[float] | np.ndarray | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.iou_thrs: list[float] | np.ndarray = iou_thrs or np.linspace(0.5, 0.95, 10)

        self.cache: dict[str, Any] = {}
        # kept for backward-compat / debugging only — actual names now come
        # from summarize_dynamic() and reflect the real thresholds used.
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
        coco_eval.params.iouThrs = np.array(self.iou_thrs)
        coco_eval.evaluate()
        coco_eval.accumulate()

        if iou_type == "keypoints_crowd":
            # CrowdPose Easy/Medium/Hard split isn't threshold-based —
            # fall back to the original summarize().
            coco_eval.summarize()
            self.cache[coco_eval.params.iouType] = dict(zip(self.stats_names, coco_eval.stats))
        else:
            stat_names, stats = summarize_dynamic(coco_eval, self.iou_thrs)
            self.cache[coco_eval.params.iouType] = dict(zip(stat_names, stats))

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

        # kept for backward-compat / the keypoints_crowd fallback only
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

        if self.iou_type == "keypoints_crowd":
            coco_eval.summarize()
            return list(zip(self.stats_names, coco_eval.stats))

        stat_names, stats = summarize_dynamic(coco_eval, self.iou_thrs)
        return list(zip(stat_names, stats))
