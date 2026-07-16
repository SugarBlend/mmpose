import abc
import os
import numpy as np
from mmdeploy.codebase.mmpose.deploy.pose_detection import _get_dataset_metainfo
from mmengine.config import Config
from mmengine.evaluator.metric import BaseMetric
from mmpose import __file__ as mmpose_root
from typing import Any

from maps import SKELETON_SUBSETS
from project.label_studio.pipelines.pipeline import MMPipeline
from project.hooks.extend_pck import GroupedPCKAccuracy


class BaseMetricConfigurator(abc.ABC):
    def __init__(self, anns_schema: str) -> None:
        self._num_joints = len(SKELETON_SUBSETS[anns_schema]["all"])

    @abc.abstractmethod
    def _configurate_metric(self, *args, **kwargs) -> list[BaseMetric] | BaseMetric:
        pass

    @abc.abstractmethod
    def calculate_results(self, results) -> dict[str, float]:
        pass

    @property
    def is_whole_body(self) -> bool:
        return self._num_joints == 133

    @property
    def num_joints(self) -> int:
        return self._num_joints


class PCKMetricConfigurator(BaseMetricConfigurator):
    def __init__(
        self, anns_schema: str, **params
    ) -> None:
        super().__init__(anns_schema)
        self.evaluators = self._configurate_metric(**params)

    @staticmethod
    def _configurate_metric(params: dict[str, Any]) -> list[BaseMetric]:
        thresholds = params.pop("thresholds")
        # TODO: is they still need for correct computation?
        params.pop("gt_converter")
        params.pop('pred_converter')
        params.pop('ann_file')
        return [
            GroupedPCKAccuracy(thr=threshold, **params) for threshold in thresholds
        ]

    def calculate_results(self, results):
        [evaluator.process({}, results) for evaluator in self.evaluators]
        res = {}
        for evaluator in self.evaluators:
            dictionary = evaluator.evaluate(size=len(evaluator.results))
            for key, value in dictionary.items():
                res[f'{key}@{evaluator.thr}'] = value
        return res


class CocoMetricConfigurator(BaseMetricConfigurator):
    def __init__(self, anns_schema: str, **params) -> None:
        super().__init__(anns_schema)
        self.evaluator = self._configurate_metric(**params)

    def _configurate_metric(self, params: dict[str, Any]) -> BaseMetric:

        gt_converter = params.pop('gt_converter')
        pred_converter = params.pop('pred_converter')

        if self.is_whole_body:
            from patches import CocoWholeBodyMetric as Metric
        else:
            from patches import CocoMetric as Metric
            import maps

            if gt_converter is not None:
                converter = dict(
                    type="KeypointConverter",
                    num_keypoints=self._num_joints,
                    mapping=getattr(maps, gt_converter),
                )
                params.update(dict(gt_converter=converter))

            if pred_converter is not None:
                converter = dict(
                    type="KeypointConverter",
                    num_keypoints=self._num_joints,
                    mapping=getattr(maps, pred_converter),
                )
                params.update(dict(pred_converter=converter))

        return Metric(**params)

    def _resolve_meta(self, config: Config) -> dict[str, Any]:
        meta = _get_dataset_metainfo(config)

        if self.is_whole_body:
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

    def update_metadata(self, pipeline: MMPipeline | Any) -> None:
        if isinstance(pipeline, MMPipeline):
            dataset_meta = self._resolve_meta(pipeline.model_cfg)
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

    def calculate_results(self, results):
        self.evaluator.process({}, results)
        return self.evaluator.compute_metrics(self.evaluator.results)


correspondence = {
    'COCO': CocoMetricConfigurator,
    "PCK": PCKMetricConfigurator
}
