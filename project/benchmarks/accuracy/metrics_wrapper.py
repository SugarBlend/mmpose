import abc
import numpy as np
from mmengine.config import Config
from mmengine.evaluator.metric import BaseMetric
from typing import Any
import patches
import maps
from project.hooks.extend_pck import GroupedPCKAccuracy


class BaseMetricConfigurator(abc.ABC):
    def __init__(self, metapath: str):
        self.dataset_metainfo = self._resolve_meta(metapath)

    @abc.abstractmethod
    def _configurate_metric(self, *args, **kwargs) -> list[BaseMetric] | BaseMetric:
        pass

    @abc.abstractmethod
    def calculate_results(self, results) -> dict[str, float]:
        pass

    @staticmethod
    def _resolve_meta(metapath: str) -> dict[str, Any]:
        meta = Config.fromfile(metapath)

        meta.dataset_info.update({"num_keypoints": len(meta.dataset_info.sigmas)})
        meta.dataset_info.sigmas = np.array(meta.dataset_info.sigmas)
        return meta.dataset_info


class PCKMetricConfigurator(BaseMetricConfigurator):
    def __init__(self, metapath: str, params: dict[str, Any]) -> None:
        super().__init__(metapath)
        self.evaluators = self._configurate_metric(params)
        for evaluator in self.evaluators:
            evaluator.dataset_meta = self.dataset_metainfo

    def _configurate_metric(self, params: dict[str, Any]) -> list[BaseMetric]:
        params.pop("ann_file")
        thresholds = params.pop("thresholds")
        converter_params = params.pop("gt_converter")

        if converter_params is not None:
            gt_converter = dict(
                type="KeypointConverter",
                num_keypoints=converter_params["num_keypoints"],
                mapping=getattr(maps, converter_params["mapping"]),
            )
            params.update(dict(gt_converter=gt_converter))

        return [
            GroupedPCKAccuracy(threshold, **params) for threshold in thresholds
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
    def __init__(self, metapath: str, params: dict[str, Any]) -> None:
        super().__init__(metapath)
        self.evaluator = self._configurate_metric(params)
        self.evaluator.dataset_meta = self.dataset_metainfo

    def _configurate_metric(self, params: dict[str, Any]) -> BaseMetric:
        eval_class = params.pop("type")
        converter_params = params.pop("gt_converter")

        cls = getattr(patches, eval_class)

        if converter_params is not None:
            gt_converter = dict(
                type="KeypointConverter",
                num_keypoints=converter_params["num_keypoints"],
                mapping=getattr(maps, converter_params["mapping"]),
            )
            params.update(dict(gt_converter=gt_converter))

        return cls(**params)

    def calculate_results(self, results):
        self.evaluator.process({}, results)
        return self.evaluator.compute_metrics(self.evaluator.results)


correspondence = {
    "COCO": CocoMetricConfigurator,
    "PCK": PCKMetricConfigurator
}
