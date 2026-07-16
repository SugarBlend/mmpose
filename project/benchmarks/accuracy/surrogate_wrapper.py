from pathlib import Path
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import mim
from mmpose.structures.pose_data_sample import PoseDataSample
from tqdm import tqdm
from typing import Any
from urllib.parse import urlparse

from maps import halpe2coco_wholebody
from project.label_studio.pipelines.pipeline import MMPipeline
from project.hooks.minio_backend import MinIOBackend
from typing import Callable


SurrogateCallback = Callable[["PoseDataSample", dict[str, Any], str], "PoseDataSample"]


def attempt_download_default() -> tuple[str, str]:
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


class SurrogateEstimatorWrapper(object):
    def __init__(self) -> None:
        self._coco: COCO | None = None
        self._pipeline: MMPipeline | None = None

        self._callbacks = {
            21: self._hands21_callback,
            26: self._halpe_callback,
        }

    @property
    def pipeline(self) -> MMPipeline:
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: MMPipeline) -> None:
        self._pipeline = value

    @property
    def coco(self) -> COCO:
        return self._coco

    @coco.setter
    def coco(self, value: COCO) -> None:
        self._coco = value

    def _halpe_callback(self, result: PoseDataSample, ann: dict[str, Any], data: str | np.ndarray) -> PoseDataSample:
        inst = result.pred_instances

        halpe_inst = self._pipeline(data, [ann["bbox"]])[0].pred_instances

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
            hand_results = self._pipeline(data, hand_bboxes)
            for idx, (name, pts) in enumerate(hand_indices):
                num_kp = hand_results[idx].pred_instances.keypoints.shape[1]
                inst.keypoints[:, pts: pts + num_kp] = hand_results[idx].pred_instances.keypoints
                inst.keypoint_scores[:, pts: pts + num_kp] = hand_results[idx].pred_instances.keypoint_scores
                inst.keypoints_visible[:, pts: pts + num_kp] = hand_results[idx].pred_instances.keypoints_visible

        return result

    def __call__(
        self,
        dataset_folder: str,
        expected_joints: int | None = None
    ) -> list[dict[str, Any]]:
        inner_callback_func = self._callbacks.get(expected_joints)
        if inner_callback_func is None:
            inner_callback_func = lambda result, *args, **kwargs: result

        if dataset_folder.startswith("minio://"):
            client = MinIOBackend()

        results: list[dict[str, Any]] = []
        for img_id in tqdm(self._coco.getImgIds(), desc="Processing images", ncols=70):
            img_info = self._coco.loadImgs(img_id)[0]

            if dataset_folder.startswith("minio://"):
                parsed = urlparse(dataset_folder)
                path = parsed.path.lstrip("/")
                bytes_data = client.get(f"{path}/{img_info['file_name']}")
                arr = np.frombuffer(bytes_data, dtype=np.uint8)
                data = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                data = os.path.join(dataset_folder, img_info["file_name"])

            for ann in self._coco.imgToAnns.get(img_id, []):
                batch_results_body = self.pipeline(data, [ann["bbox"]]) # bboxes in xywh format
                result = batch_results_body[0]

                result = inner_callback_func(result, ann, data)

                result.id = ann["id"]
                result.img_id = ann["image_id"]
                gt_keypoints = np.asarray(ann["keypoints"]).reshape(1, -1, 3)
                result.gt_instances.set_field(gt_keypoints[:, :, :2], "keypoints")
                result.gt_instances.set_field(gt_keypoints[:, :, 2], "keypoints_visible")
                results.append(result.to_dict())
        return results
