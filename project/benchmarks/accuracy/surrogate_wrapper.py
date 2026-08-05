import os
import cv2
from dataclasses import dataclass
from enum import Enum
import numpy as np
import maps
from mmpose.registry import TRANSFORMS
from mmpose.structures import PoseDataSample
from mmpose.datasets.transforms.converting import KeypointConverter
from pycocotools.coco import COCO
from tqdm import tqdm
from typing import Any
from urllib.parse import urlparse

from project.label_studio.pipelines.pipeline import MMPipeline
from project.hooks.minio_backend import MinIOBackend


@dataclass
class PartSpec:
    ann_field: str
    converter: KeypointConverter


class AnnFormats(Enum):
    CocoWholeBody = "CocoWholeBody"
    Coco = "Coco"


class SurrogateEstimatorWrapper:
    def __init__(self) -> None:
        self._coco: COCO | None = None
        self._pipeline: MMPipeline | None = None
        self._part_specs: list[PartSpec] = []

    @property
    def coco(self) -> COCO:
        return self._coco

    @coco.setter
    def coco(self, value: COCO) -> None:
        self._coco = value

    @property
    def pipeline(self) -> MMPipeline:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: MMPipeline) -> None:
        self._pipeline = value

    @property
    def predict_converters(self) -> list[PartSpec]:
        return self._part_specs

    @predict_converters.setter
    def predict_converters(self, values: list[dict[str, Any]] | None) -> None:
        if values:
            self._part_specs = [
                PartSpec(
                    ann_field=value["ann_field"],
                    converter=TRANSFORMS.build(
                        dict(
                            type="KeypointConverter",
                            num_keypoints=value["num_keypoints"],
                            mapping=getattr(maps, value["mapping"]),
                        )
                    ),
                )
                for value in values
            ]

    def _convert(self, result: PoseDataSample, converter: KeypointConverter) -> PoseDataSample:
        converted = converter(
            dict(
                keypoints=result.pred_instances.keypoints,
                keypoints_visible=result.pred_instances.keypoints_visible,
            )
        )
        result.pred_instances.keypoints = converted["keypoints"]
        result.pred_instances.keypoints_visible = converted["keypoints_visible"]

        scores = result.pred_instances.keypoint_scores
        converted_scores = converter(
            dict(keypoints=scores[..., None], keypoints_visible=np.ones_like(scores))
        )
        result.pred_instances.keypoint_scores = converted_scores["keypoints"][..., 0]
        return result

    def _part_callback(self, ann: dict[str, Any], data: str | np.ndarray) -> PoseDataSample | None:
        bboxes: list[list[float]] = []
        specs: list[PartSpec] = []

        if not self._part_specs:
            bboxes.append(ann["bbox"])
        else:
            for spec in self._part_specs:
                box = ann.get(spec.ann_field, [])
                if any(v != 0 for v in box):
                    bboxes.append(box)
                    specs.append(spec)

            if not bboxes:
                return None

        raw_results = self._pipeline(data, bboxes)

        if specs:
            results = [self._convert(r, spec.converter) for r, spec in zip(raw_results, specs)]

            merged = results[0]
            for extra in results[1:]:
                merged.pred_instances.keypoints += extra.pred_instances.keypoints
                merged.pred_instances.keypoints_visible += extra.pred_instances.keypoints_visible
                merged.pred_instances.keypoint_scores += extra.pred_instances.keypoint_scores

            return merged
        else:
            return raw_results[0]

    def __call__(self, dataset_folder: str, eval_format: str | AnnFormats) -> list[dict[str, Any]]:
        object_storages = ["minio://", "s3://"]
        if any(dataset_folder.startswith(prefix) for prefix in object_storages):
            client = MinIOBackend()

        results: list[dict[str, Any]] = []
        for img_id in tqdm(self._coco.getImgIds(), desc="Processing images", ncols=70):
            img_info = self._coco.loadImgs(img_id)[0]

            if any(dataset_folder.startswith(prefix) for prefix in object_storages):
                path = urlparse(dataset_folder).path.lstrip("/")
                bytes_data = client.get(f"{path}/{img_info['file_name']}")
                arr = np.frombuffer(bytes_data, dtype=np.uint8)
                data = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                data = os.path.join(dataset_folder, img_info["file_name"])

            for ann in self._coco.imgToAnns.get(img_id, []):
                result = self._part_callback(ann, data)
                if result:
                    result.id = ann["id"]
                    result.img_id = ann["image_id"]

                    if eval_format == AnnFormats.CocoWholeBody.value:
                        gt_keypoints = np.asarray([
                            *ann["keypoints"],
                            *ann["foot_kpts"],
                            *ann["face_kpts"],
                            *ann["lefthand_kpts"],
                            *ann["righthand_kpts"]
                        ]).reshape(1, -1, 3)
                    else:
                        gt_keypoints = np.asarray(ann["keypoints"]).reshape(1, -1, 3)

                    result.gt_instances.set_field(gt_keypoints[:, :, :2], "keypoints")
                    result.gt_instances.set_field(gt_keypoints[:, :, 2], "keypoints_visible")

                    results.append(result.to_dict())

        # reset states
        self._predict_converter = None
        self._pipeline = None
        return results
