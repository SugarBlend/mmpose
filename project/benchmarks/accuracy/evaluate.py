from __future__ import annotations
import sys
from pathlib import Path
import torch


sys.path.insert(0, Path(__file__).parents[3].as_posix())
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load


import argparse
import json
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import mim
from mmdeploy.codebase.mmpose.deploy.pose_detection import _get_dataset_metainfo
from mmengine.config import Config
from mmengine.evaluator.metric import BaseMetric
from mmpose.structures.pose_data_sample import PoseDataSample
from mmpose import __file__ as mmpose_root
import torch
from tqdm import tqdm
from typing import Any, Callable

from config import ModelConfig, EvalConfig
from maps import SKELETON_SUBSETS, halpe2coco_wholebody, coco2coco_wholebody
from project.label_studio.pipelines.pipeline import MMPipeline
from tools import logger, generate_radar_plot, save_metrics_xlsx


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

    def _build_metric(
        self,
        ann_file: str,
        anns_schema: str,
        gt_converter: str,
        pred_converter: str,
        iou_thrs: list[float] | np.ndarray | None
    ) -> BaseMetric:
        self.coco = COCO(ann_file)
        num_joints = len(SKELETON_SUBSETS[anns_schema]["all"])
        self.is_wholebody = (num_joints == 133)

        params = dict(
            ann_file=ann_file,
            iou_type="keypoints",
            score_mode="keypoint",
            keypoint_score_thr=0.2,
            nms_mode="none",
            nms_thr=0.9,
            format_only=False,
            use_area=True,
            iou_thrs=iou_thrs,
        )

        if self.is_wholebody:
            from patches import CocoWholeBodyMetric as Metric
        else:
            from patches import CocoMetric as Metric
            import maps

            if gt_converter is not None:
                converter = dict(
                    type="KeypointConverter",
                    num_keypoints=num_joints,
                    mapping=getattr(maps, gt_converter),
                )
                params.update(dict(gt_converter=converter))

            if pred_converter is not None:
                converter = dict(
                    type="KeypointConverter",
                    num_keypoints=num_joints,
                    mapping=getattr(maps, pred_converter),
                )
                params.update(dict(pred_converter=converter))

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

    def _halpe_callback(self, result: PoseDataSample, ann: dict[str, Any], data: str | np.ndarray) -> PoseDataSample:
        inst = result.pred_instances

        halpe_inst = self.pipeline(data, [ann["bbox"]])[0].pred_instances

        for key, values in halpe2coco_wholebody.items():
            src_ids, dst_ids = values
            if max(src_ids) > halpe_inst.keypoints.shape[1]:
                # case when using halpe26 instead of halpe136
                continue
            inst.keypoints[:, dst_ids] = halpe_inst.keypoints[:, src_ids]
            inst.keypoint_scores[:, dst_ids] = halpe_inst.keypoint_scores[:, src_ids]
            inst.keypoints_visible[:, dst_ids] = halpe_inst.keypoints_visible[:, src_ids]

        return result

    def _hands21_callback(self, result: PoseDataSample, ann: dict[str, Any], data: str | np.ndarray) -> PoseDataSample:
        hand_bboxes: list[list[float]] = []
        hand_indices: list[tuple[str, int]] = []

        for name, pts in [("lefthand", 91), ("righthand", 112)]:
            box = ann.get(f"{name}_box", [0, 0, 0, 0])
            if ann.get(f"{name}_valid", False) and any(value != 0 for value in box):
                hand_bboxes.append(box)
                hand_indices.append((name, pts))

        if hand_bboxes:
            inst = result.pred_instances
            hand_results = self.pipeline(data, hand_bboxes)
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

        if dataset_folder.startswith("minio://"):
            from project.hooks.minio_backend import MinIOBackend
            client = MinIOBackend()

        for img_id in tqdm(self.coco.getImgIds(), desc="Processing images", ncols=70):
            img_info = self.coco.loadImgs(img_id)[0]

            if dataset_folder.startswith("minio://"):
                bytes_data = client.get(img_info["file_name"])
                arr = np.frombuffer(bytes_data, dtype=np.uint8)
                data = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                data = os.path.join(dataset_folder, img_info["file_name"])

            for ann in self.coco.imgToAnns.get(img_id, []):
                batch_results_body = pipeline(data, [ann["bbox"]]) # bboxes in xywh format
                result = batch_results_body[0]

                result = inner_callback_func(result, ann, data)

                result.id = ann["id"]
                result.img_id = ann["image_id"]
                self.evaluator.process({}, [result.to_dict()])

    @torch.no_grad()
    def evaluate(
        self,
        dataset_folder: str,
        ann_file: str,
        gt_converter: str | None = None,
        pred_converter: str | None = None,
        anns_schema: str = "coco_wholebody",
        expected_joints: int | None = None,
        num_samples: int | None = None,
        iou_thrs: list[float] | np.ndarray | None = None
    ) -> dict[str, float]:
        self.evaluator = self._build_metric(ann_file, anns_schema, gt_converter, pred_converter, iou_thrs)

        if isinstance(self.pipeline, MMPipeline):
            dataset_meta = self._resolve_meta(self.pipeline.model_cfg)
        else:
            # TODO: Now this is bad hardcode to halpe sigmas
            target_meta = Config.fromfile(
                f"{os.path.dirname(mmpose_root)}/.mim/configs/_base_/datasets/halpe.py"
            )
            sigmas = np.array(target_meta.dataset_info.sigmas)
            dataset_meta = {
                "sigmas": sigmas,
                "num_keypoints": len(sigmas),
            }

        self.evaluator.dataset_meta = dataset_meta
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
    iou_thrs: list[float] | np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    exp_metrics: dict[str, dict[str, float]] = {}
    is_wholebody = False

    for config in model_configs:
        logger.info(f"Evaluating: {config.legend}")

        if "sapiens2" in config.config_path:
            from project.label_studio.pipelines.sapiens2 import Sapiens2
            pipeline = Sapiens2(config.model_path, config.config_path)
        else:
            pipeline = MMPipeline(config.model_path, config.config_path)

        evaluator = PoseMetricEvaluator(pipeline)
        metrics = evaluator.evaluate(
            dataset_folder=config.dataset_folder,
            ann_file=config.ann_file,
            gt_converter=config.gt_converter,
            pred_converter=config.pred_converter,
            anns_schema=config.anns_schema,
            expected_joints=config.expected_joints,
            iou_thrs=iou_thrs,
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
    parser.add_argument("--config", "-c", type=str, help="Path to eval-config.yaml")
    parser.add_argument("--save-dir", "-o", type=str, help="Directory to save metrics.json, metrics.xlsx and radar.png")
    parser.add_argument("--no-show", action="store_true", help="Do not display the radar plot interactively")
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.joinpath("../../../tools/.env"))
    args = _parse_args()
    cfg = EvalConfig.load(args.config)
    launch_evaluation(
        cfg.models,
        save_dir=args.save_dir or cfg.save_dir,
        show_plot=(not args.no_show) and cfg.show_plot,
        radar_xticks=cfg.radar_xticks,
        iou_thrs=cfg.iou_thrs
    )
