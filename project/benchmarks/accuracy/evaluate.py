from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
import mim
import numpy as np
from pycocotools.coco import COCO
from mmdeploy.codebase.mmpose.deploy.pose_detection import _get_dataset_metainfo
from mmengine.config import Config
from mmpose.structures.pose_data_sample import PoseDataSample
from mmpose import __file__ as mmpose_root
import torch
from tqdm import tqdm

sys.path.insert(0, Path(__file__).parents[3].as_posix())
from project.label_studio.pipelines.pipeline import MMPipeline

from config import ModelConfig, EvalConfig
from maps import SKELETON_SUBSETS
from mmengine.evaluator.metric import BaseMetric
from tools import logger, generate_radar_plot, save_metrics_xlsx


_HALPE2CWB: dict[str, tuple[list[int], list[int]]] = {
    "body": (list(range(17)), list(range(17))),
    "foot": ([20, 22, 24, 21, 23, 25], [17, 18, 19, 20, 21, 22]),
    "left_hand": (list(range(94, 115)), list(range(91, 112))),
    "right_hand": (list(range(115, 136)), list(range(112, 133))),
}
_COCO2CWB: dict[str, tuple[list[int], list[int]]] = {
    "body": (list(range(17)), list(range(17))),
}

SurrogateCallback = Callable[["PoseDataSample", dict[str, Any], str], "PoseDataSample"]


class PoseMetricEvaluator(object):
    def __init__(self, pipeline: MMPipeline) -> None:
        self.pipeline: MMPipeline = pipeline
        self.coco: COCO | None = None
        self.is_wholebody: bool = False
        self.evaluator: BaseMetric | None = None

        self._callbacks = {
            21: self._hands21_callback,
            26: self._halpe_callback
        }

    def _build_metric(self, ann_file: str, anns_schema: str, converter: str) -> BaseMetric:
        self.coco = COCO(ann_file)
        num_joints = len(SKELETON_SUBSETS[anns_schema]["all"])
        self.is_wholebody = (num_joints == 133)

        params = dict(
            ann_file=ann_file,
            iou_type="keypoints",
            score_mode="bbox",
            keypoint_score_thr=0.2,
            nms_mode="none",
            nms_thr=0.5,
            format_only=False,
            use_area=False,
        )

        if self.is_wholebody:
            from patches import CocoWholeBodyMetric as Metric
        else:
            from mmpose.evaluation.metrics import CocoMetric as Metric
            import maps

            mapping = getattr(maps, converter)
            converter = dict(
                type="KeypointConverter",
                num_keypoints=num_joints,
                mapping=mapping,
            )
            params.update(dict(gt_converter=converter))

        return Metric(**params)

    def _resolve_meta(self, config: Config) -> dict[str, Any]:
        meta = _get_dataset_metainfo(config)

        if self.is_wholebody:
            meta = Config.fromfile(
                f"{os.path.dirname(mmpose_root)}/.mim/configs/_base_/datasets/coco_wholebody.py"
            )
        elif "from_file" in meta:
            meta = Config.fromfile(
                f"{os.path.dirname(mmpose_root)}/.mim/{meta['from_file']}"
            )

        if "dataset_info" not in meta:
            sigmas = np.array(meta["sigmas"])
            meta["dataset_info"] = {"sigmas": sigmas}
        else:
            sigmas = np.array(meta.dataset_info.sigmas)

        meta["dataset_info"]["sigmas"] = sigmas
        meta["dataset_info"]["num_keypoints"] = len(sigmas)
        return meta["dataset_info"]

    @staticmethod
    def _attempt_download_default() -> tuple[str, str]:
        default_root = Path.cwd().parent.joinpath("models")
        pose_config = default_root.joinpath("rtmw-x_8xb320-270e_cocktail14-384x288.py")
        weights = list(default_root.glob("*.pth"))

        if not pose_config.exists() or not len(weights):
            default_root.mkdir(parents=True, exist_ok=True)
            checkpoint_filename = mim.download("mmpose", [pose_config.stem], dest_root=default_root.as_posix())[0]
            pose_checkpoints = default_root.joinpath(checkpoint_filename)
        else:
            pose_checkpoints = weights[0]

        return pose_checkpoints.as_posix(), pose_config.as_posix()

    def _halpe_callback(self, result: PoseDataSample, ann: dict[str, Any], img_path: str) -> PoseDataSample:
        inst = result.pred_instances

        halpe_inst = self.pipeline(img_path, [ann["bbox"]])[0].pred_instances

        for key, values in _HALPE2CWB.items():
            src_ids, dst_ids = values
            if max(src_ids) > halpe_inst.keypoints.shape[1]:
                # case when using halpe26 instead of halpe136
                continue
            inst.keypoints[:, dst_ids] = halpe_inst.keypoints[:, src_ids]
            inst.keypoint_scores[:, dst_ids] = halpe_inst.keypoint_scores[:, src_ids]
            inst.keypoints_visible[:, dst_ids] = halpe_inst.keypoints_visible[:, src_ids]

        return result

    def _hands21_callback(self, result: PoseDataSample, ann: dict[str, Any], img_path: str) -> PoseDataSample:
        hand_bboxes: list[list[float]] = []
        hand_indices: list[tuple[str, int]] = []

        for name, pts in [("lefthand", 91), ("righthand", 112)]:
            box = ann.get(f"{name}_box", [0, 0, 0, 0])
            if ann.get(f"{name}_valid", False) and any(value != 0 for value in box):
                hand_bboxes.append(box)
                hand_indices.append((name, pts))

        if hand_bboxes:
            inst = result.pred_instances
            hand_results = self.pipeline(img_path, hand_bboxes)
            for idx, (name, pts) in enumerate(hand_indices):
                num_kp = hand_results[idx].pred_instances.keypoints.shape[1]
                inst.keypoints[:, pts: pts + num_kp] = hand_results[idx].pred_instances.keypoints
                inst.keypoint_scores[:, pts: pts + num_kp] = hand_results[idx].pred_instances.keypoint_scores
                inst.keypoints_visible[:, pts: pts + num_kp] = hand_results[idx].pred_instances.keypoints_visible

        return result

    def _surrogate_estimation(
        self,
        dataset_folder: str,
        pipeline: MMPipeline,
        inner_callback_func: SurrogateCallback | None = None,
    ) -> None:
        if inner_callback_func is None:
            inner_callback_func = lambda result, *args, **kwargs: result

        for img_id in tqdm(self.coco.getImgIds(), desc="Processing images", ncols=70):
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(dataset_folder, img_info["file_name"])

            for ann in self.coco.imgToAnns.get(img_id, []):
                batch_results_body = pipeline(img_path, [ann["bbox"]])
                result = batch_results_body[0]

                result = inner_callback_func(result, ann, img_path)

                result.id = ann["id"]
                result.img_id = ann["image_id"]
                self.evaluator.process({}, [result.to_dict()])

    @torch.no_grad()
    def evaluate(
        self,
        dataset_folder: str,
        ann_file: str,
        converter: str | None = None,
        anns_schema: str = "coco_wholebody",
        expected_joints: int | None = None,
        num_samples: int | None = None
    ) -> dict[str, float]:
        self.evaluator = self._build_metric(ann_file, anns_schema, converter)
        self.evaluator.dataset_meta = self._resolve_meta(self.pipeline.model_cfg)
        self.coco.imgs = dict(list(self.coco.imgs.items())[: num_samples])
        self.coco.anno_file = [ann_file]
        self.evaluator.coco = self.coco

        if anns_schema == "coco_wholebody" and expected_joints != 133:
            local_estimator = MMPipeline(*self._attempt_download_default())
            self._surrogate_estimation(dataset_folder, pipeline=local_estimator,
                                       inner_callback_func=self._callbacks[expected_joints])
        else:
            self._surrogate_estimation(dataset_folder, pipeline=self.pipeline)


        return self.evaluator.compute_metrics(self.evaluator.results)


def launch_evaluation(
    model_configs: list[ModelConfig],
    save_dir: str | None = None,
    show_plot: bool = True,
    radar_xticks: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    exp_metrics: dict[str, dict[str, float]] = {}
    is_wholebody = False

    for config in model_configs:
        logger.info(f"Evaluating: {config.legend}")
        pipeline = MMPipeline(config.model_path, config.config_path)
        evaluator = PoseMetricEvaluator(pipeline)
        metrics = evaluator.evaluate(
            dataset_folder=config.dataset_folder,
            ann_file=config.ann_file,
            converter=config.converter,
            anns_schema=config.anns_schema,
            expected_joints=config.expected_joints
        )
        exp_metrics[config.legend] = metrics
        is_wholebody = evaluator.is_wholebody
        logger.info(f"Results: {metrics}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = save_dir.joinpath("metrics.json")
        metrics_path.write_text(json.dumps(exp_metrics, indent=2))
        logger.info(f"Metrics JSON saved: '{metrics_path}'")

        xlsx_path = save_dir.joinpath("metrics.xlsx")
        save_metrics_xlsx(exp_metrics, xlsx_path.as_posix(), is_wholebody=is_wholebody)

    plot_path = save_dir.joinpath("radar.png") if save_dir else None
    generate_radar_plot(
        exp_metrics,
        is_wholebody=is_wholebody,
        xticks=radar_xticks,
        save_path=plot_path,
        show=show_plot,
        title="Pose Model Comparison",
    )

    return exp_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose model evaluation — mAP / mAR with flexible keypoint subsets")
    parser.add_argument("--config", "-c", default="./eval-config.yaml", type=str, help="Path to eval-config.yaml")
    parser.add_argument("--save-dir", "-o", type=str, default=None, help="Directory to save metrics.json, metrics.xlsx and radar.png")
    parser.add_argument("--no-show", action="store_true", help="Do not display the radar plot interactively")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = EvalConfig.load(args.config)
    launch_evaluation(
        cfg.models,
        save_dir=args.save_dir or cfg.save_dir,
        show_plot=(not args.no_show) and cfg.show_plot,
        radar_xticks=cfg.radar_xticks,
    )
