import argparse
import json
import os
import sys
from math import pi
from pathlib import Path
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mmdeploy.codebase.mmpose.deploy.pose_detection import _get_dataset_metainfo
from mmengine.config import Config

import mmpose.evaluation.metrics
from mmpose import __file__ as mmpose_root
from mmpose.utils.logger import MMLogger
import torch
from tqdm import tqdm
from typing import Any

sys.path.insert(0, Path(__file__).parents[3].as_posix())
from project.annotations.pipelines.pipeline import MMPipeline
from config import ModelConfig, load_eval_config

matplotlib.use("Qt5Agg")
logger = MMLogger.get_instance("Accuracy")


class PoseMetricEvaluator(object):
    _HALPE26_MAPPING = [(i, i) for i in range(17)] + [(17, 20), (18, 22), (19, 24),
                                                      (20, 21), (21, 23), (22, 25)]

    def __init__(self, pipeline: MMPipeline) -> None:
        self.pipeline = pipeline

    def _build_metric(self, ann_file: str, num_joints: int):
        if num_joints == 133:
            from mmpose.evaluation.metrics import CocoWholeBodyMetric as Evaluator
        else:
            from mmpose.evaluation.metrics import CocoMetric as Evaluator

        converter = None
        if num_joints == 26:
            converter = dict(
                type="KeypointConverter",
                num_keypoints=num_joints,
                mapping=self._HALPE26_MAPPING,
            )

        return Evaluator(
            ann_file=ann_file,
            iou_type="keypoints",
            score_mode="bbox_keypoint",
            keypoint_score_thr=0.2,
            nms_mode="oks_nms",
            nms_thr=0.5,
            format_only=False,
            gt_converter=converter,
        )

    @staticmethod
    def _resolve_meta(config: Config, num_joints: int) -> dict[str, Any]:
        meta = _get_dataset_metainfo(config)

        if "from_file" in meta:
            meta = Config.fromfile(f"{os.path.dirname(mmpose_root)}/.mim/{meta['from_file']}")

        if "dataset_info" not in meta:
            meta["dataset_info"] = {"sigmas": np.array(meta["sigmas"])}
        else:
            meta.dataset_info.sigmas = np.array(meta.dataset_info.sigmas)

        meta.dataset_info.num_keypoints = num_joints
        return meta.dataset_info

    @torch.no_grad()
    def evaluate(
        self,
        dataset_folder: str,
        ann_file: str,
        num_joints: int = 26,
    ) -> dict[str, float]:
        coco_metric = self._build_metric(ann_file, num_joints)
        coco_metric.dataset_meta = self._resolve_meta(self.pipeline.model_cfg, num_joints)
        coco = COCO(ann_file)

        for img_id in tqdm(coco.getImgIds(), desc="Processing images", ncols=70):
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(dataset_folder, img_info["file_name"])
            annotations = coco.imgToAnns.get(img_id, [])

            if not annotations:
                continue

            bboxes = [ann["bbox"] for ann in annotations]
            batch_results = self.pipeline(img_path, bboxes)

            for ann, result in zip(annotations, batch_results):
                result.id = ann["id"]
                result.img_id = ann["image_id"]
                coco_metric.process({}, [result.to_dict()])

        return coco_metric.compute_metrics(coco_metric.results)


def generate_radar_plot(
    dfs: list[pd.DataFrame],
    xticks: list[float] | None = None,
    save_path: str | None = None,
    show: bool = True,
    title: str = "Pose Model Comparison",
) -> plt.Figure:
    if not dfs:
        raise ValueError("No dataframes to plot")

    all_keys = dfs[0].index.tolist()
    for df in dfs[1:]:
        if df.index.tolist() != all_keys:
            logger.warning("Metric keys differ between experiments — results may be misleading")

    xticks = xticks or [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    N = len(all_keys)
    angles = [n / N * 2 * pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_keys, size=9)
    ax.set_yticks(xticks)
    ax.set_yticklabels([str(t) for t in xticks], size=7)
    ax.set_ylim(min(xticks), max(xticks))
    ax.set_title(title, pad=20, size=13, weight="bold")

    colors = plt.cm.tab10.colors
    for idx, df in enumerate(dfs):
        name = df.columns[0]
        values = df[name].tolist() + [df[name].iloc[0]]
        color = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=name, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved → {save_path}")

    if show:
        plt.show()

    return fig


def launch_evaluation(
    model_configs: list[ModelConfig],
    save_dir: str | None = None,
    show_plot: bool = True,
    radar_xticks: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    exp_metrics: dict[str, dict[str, float]] = {}

    for cfg in model_configs:
        logger.info(f"Evaluating: {cfg.legend}")
        pipeline = MMPipeline(cfg.model_path, cfg.config_path)
        evaluator = PoseMetricEvaluator(pipeline)
        metrics = evaluator.evaluate(cfg.dataset_folder, cfg.ann_file, cfg.num_joints)
        exp_metrics[cfg.legend] = metrics

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = Path(save_dir).joinpath("metrics.json")
        with metrics_path.open("w") as file:
            json.dump(exp_metrics, file, indent=2)
        logger.info(f"Metrics saved → {metrics_path}")

    dfs = [
        pd.DataFrame(list(m.values()), index=list(m.keys()), columns=[legend])
        for legend, m in exp_metrics.items()
    ]
    plot_path = f"{save_dir}/radar.png" if save_dir else None
    generate_radar_plot(dfs, xticks=radar_xticks, save_path=plot_path, show=show_plot)

    return exp_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose model evaluation & accuracy")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to eval-config.yaml")
    parser.add_argument("--save-dir", "-o", type=str, default=None,
                        help="Directory to save metrics.json and radar.png")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the radar plot interactively")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_eval_config(args.config)
    launch_evaluation(cfg.models, save_dir=args.save_dir, show_plot=not args.no_show)
