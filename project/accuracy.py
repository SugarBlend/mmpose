import os
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from mmdeploy.codebase.mmpose.deploy.pose_detection import _get_dataset_metainfo
from typing import List
from mmengine.config.config import Config
from pathlib import Path
from typing import Any, Dict, Optional
from mmpose import __file__ as mmpose_root
from mmpose.structures import merge_data_samples
import pandas as pd
from math import pi
import matplotlib
import matplotlib.pyplot as plt
import random
from project.pipeline import MMPipeline


matplotlib.use("Qt5Agg")
matplotlib.rcParams["font.sans-serif"] = "Comic Sans MS"
matplotlib.rcParams["font.family"] = "sans-serif"


class PoseMetricEvaluator(object):
    def __init__(
        self,
        pipeline: MMPipeline
    ):
        self._pipeline: MMPipeline = pipeline
        self.converter: Optional[Dict[str, Any]] = None

    @property
    def pipeline(self) -> MMPipeline:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: MMPipeline) -> None:
        if not isinstance(value, MMPipeline):
            raise TypeError
        self._pipeline = value

    @torch.no_grad()
    def evaluate_model(self, dataset_folder: str, ann_file: str, joints: int) -> Dict[str, float]:
        if joints == 26:
            mapping = [(i, i) for i in range(17)] + [(17, 20), (18, 22), (19, 24), (20, 21), (21, 23), (22, 25)]
            self.converter = dict(
                type="KeypointConverter",
                num_keypoints=joints,
                mapping=mapping
            )

        if joints == 133:
            from mmpose.evaluation.metrics import CocoWholeBodyMetric as Evaluator
        else:
            from mmpose.evaluation.metrics import CocoMetric as Evaluator

        coco_metric = Evaluator(
            ann_file=ann_file,
            iou_type="keypoints",
            score_mode="bbox_keypoint",
            keypoint_score_thr=0.2,
            nms_mode="oks_nms",
            nms_thr=0.5,
            format_only=False,
            gt_converter=self.converter
        )

        config = Config(self.pipeline.model_cfg.to_dict())
        meta_data = _get_dataset_metainfo(config)
        if "from_file" in meta_data:
            meta_data = Config().fromfile(f"{os.path.dirname(mmpose_root)}/../{meta_data['from_file']}")

        if "dataset_info" not in meta_data:
            meta_data["dataset_info"] = {}
            meta_data["dataset_info"]["sigmas"] = np.array(meta_data["sigmas"])
        else:
            meta_data.dataset_info.sigmas = np.array(meta_data.dataset_info.sigmas)
        meta_data["dataset_info"]["num_keypoints"] = joints
        coco_metric.dataset_meta = meta_data["dataset_info"]

        coco = COCO(coco_metric.ann_file)
        cat_ids = coco.getCatIds(catNms=["person"])
        img_ids = coco.getImgIds(catIds=cat_ids)[:]

        results = []
        for img_id in tqdm(img_ids, desc="Processing images"):
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(dataset_folder, img_info["file_name"])

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            if not ann_ids:
                continue

            annotations = coco.loadAnns(ann_ids)
            for ann in annotations:
                batch_results = self.pipeline(img_path, [ann["bbox"]])
                result = merge_data_samples(batch_results)
                result.id = ann["id"]
                result.img_id = ann["image_id"]
                results.append(result)

        for result in results:
            coco_metric.process({}, [result.to_dict()])

        metrics = coco_metric.compute_metrics(coco_metric.results)
        return metrics


def generate_radar_plot(
    dfs: List[pd.DataFrame],
    xticks: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    if not len(dfs):
        raise Exception("Bring empty list with dataframes structs")
    if xticks is None:
        xticks = [-1, 0, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]

    categories = dfs[0].index.values
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)

    plt.yticks(xticks, list(map(str, xticks)), color="grey", size=7)
    plt.ylim(min(xticks), max(xticks))

    for idx, df in enumerate(dfs):
        name = df.columns[0]
        values = df[name].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=name)
        ax.fill(angles, values, color=(random.randint(0, 255) / 255, random.randint(0, 255) / 255,
                                       random.randint(0, 255) / 255), alpha=0.1)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.show(block=True)

    if save_path is not None:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        ax.get_figure().savefig(save_path, facecolor=ax.get_facecolor(), dpi=300)


def calculate_metrics(meta_infos: List[Config]) -> None:
    dfs: List[pd.DataFrame] = []
    results: List[Dict[str, float]] = []
    for idx, item in enumerate(meta_infos):
        pipeline = MMPipeline(item.model_path, item.config_path)
        if not idx:
            worker = PoseMetricEvaluator(pipeline)
        else:
            worker.pipeline = pipeline
        metrics = worker.evaluate_model(item.dataset_folder, item.ann_file, 26)
        results.append(metrics)

    for idx, metrics in enumerate(results):
        legend = meta_infos[idx].get("legend")
        if legend:
            column_headers = [legend]
        else:
            column_headers = [f"Experiment {idx}"]
        dfs.append(pd.DataFrame(metrics.values(), index=list(metrics.keys()), columns=column_headers))

    generate_radar_plot(dfs, xticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])


if __name__ == "__main__":
    meta_infos = [
        Config(
            dict(
                model_path="checkpoints/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth",
                config_path = "../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py",
                dataset_folder = "D:/NewPoseCustom/2025-10-20 12-50-24",
                ann_file = "E:/Projects/mmpose/project/annotations/pose/output.json",
                legend="Default"
            )
        ),
        Config(
            dict(
                model_path="E:/ViTrackerStaff/models/pose_estimation/mmpose/TensorRT/rtmpose-halpe-m_256x192_fp32.engine",
                config_path = "../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py",
                dataset_folder = "D:/NewPoseCustom/2025-10-20 12-50-24",
                ann_file = "E:/Projects/mmpose/project/annotations/pose/output.json",
                legend="TRT_fp32"
            )
        ),
        Config(
            dict(
                model_path="E:/ViTrackerStaff/models/pose_estimation/mmpose/TensorRT/rtmpose-halpe-m_256x192_fp16.engine",
                config_path = "../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py",
                dataset_folder = "D:/NewPoseCustom/2025-10-20 12-50-24",
                ann_file = r"E:/Projects/mmpose/project/annotations/pose/output.json",
                legend="TRT_fp16"
            )
        ),
        Config(
            dict(
                model_path="E:/Projects/mmpose/work_dirs/rtmpose-m_8xb512-700e_body8-halpe26-256x192/best_AUC_epoch_29.pth",
                config_path = "../configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py",
                dataset_folder = "D:/NewPoseCustom/2025-10-20 12-50-24",
                ann_file = r"E:/Projects/mmpose/project/annotations/pose/output.json"
            )
        ),
    ]
    calculate_metrics(meta_infos)


    # >>> from mmpose.evaluation.metrics import PCKAccuracy
    # >>> import numpy as np
    # >>> from mmengine.structures import InstanceData
    # >>> num_keypoints = 15
    # >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
    # >>> gt_instances = InstanceData()
    # >>> gt_instances.keypoints = keypoints
    # >>> gt_instances.keypoints_visible = np.ones(
    # ...     (1, num_keypoints, 1)).astype(bool)
    # >>> gt_instances.bboxes = np.random.random((1, 4)) * 20
    # >>> pred_instances = InstanceData()
    # >>> pred_instances.keypoints = keypoints
    # >>> data_sample = {
    # ...     'gt_instances': gt_instances.to_dict(),
    # ...     'pred_instances': pred_instances.to_dict(),
    # ... }
    # >>> data_samples = [data_sample]
    # >>> data_batch = [{'inputs': None}]
    # >>> pck_metric = PCKAccuracy(thr=0.5, norm_item='bbox')
    # ...: UserWarning: The prefix is not set in metric class PCKAccuracy.
    # >>> pck_metric.process(data_batch, data_samples)
    # >>> pck_metric.evaluate(1)
