import logging
import torch
from typing import Any

def _patched_torch_load(*args, **kwargs) -> Any:
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

_orig_torch_load = torch.load
torch.load = _patched_torch_load

import os
import threading
import cv2
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from pathlib import Path
import uuid
from ultralytics import YOLO
import mim
import gdown


HALPE26_KEYPOINTS: list[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
    "head",            # 17
    "neck",            # 18
    "hip",             # 19
    "left_big_toe",    # 20
    "right_big_toe",   # 21
    "left_small_toe",  # 22
    "right_small_toe", # 23
    "left_heel",       # 24
    "right_heel",      # 25
]

# correspondence to labeling configuration file - "config-halpe.xml"
KPT_LABEL_FIELD: dict[str, str] = {
    kpt: ("label_body_keypoints" if idx < 20 else "label_foot_keypoints")
    for idx, kpt in enumerate(HALPE26_KEYPOINTS)
}


def make_rectanglelabels(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    rect_id: str,
    score: float,
) -> dict[str, Any]:
    return {
        "id": rect_id,
        "type": "rectanglelabels",
        "from_name": "label_rectangles",
        "to_name": "image",
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": x1 / img_w * 100,
            "y": y1 / img_h * 100,
            "width": (x2 - x1) / img_w * 100,
            "height": (y2 - y1) / img_h * 100,
            "rotation": 0,
            "rectanglelabels": ["person"],
        },
        "score": score,
    }


def make_keypointlabels(
    kx: float, ky: float,
    img_w: int, img_h: int,
    kpt_name: str,
    from_name: str,
    rect_id: str,
    score: float,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": "keypointlabels",
        "from_name": from_name,
        "to_name": "image",
        "parentID": rect_id,
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": kx / img_w * 100,
            "y": ky / img_h * 100,
            "width": 0.5,
            "keypointlabels": [kpt_name],
        },
        "score": score,
    }


class YOLODetector(object):
    def __init__(self, model_path: str, device: str, score_threshold: float = 0.3) -> None:
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info(f"Loading '{model_path}' on {device}")
        self.score_threshold = score_threshold
        self.model = YOLO(model_path, verbose=False)
        if "cuda" in device:
            self.model.cuda()
        self.model.eval()
        self.model.fuse()
        self.logger.info("Initialization successfully completed")

    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> np.ndarray:
        raw = self.model(image, verbose=False)[0].boxes.data.cpu().numpy()
        if raw.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float32)
        mask = (raw[:, 5] == 0) & (raw[:, 4] >= self.score_threshold)
        return raw[mask, :4].astype(np.float32)


class MMPoseEstimator(object):
    def __init__(self, config: str, checkpoint: str, device: str) -> None:
        from mmpose.apis import init_model as init_pose_estimator
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info(f"Loading '{checkpoint}' on {device}")
        self.pose_estimator = init_pose_estimator(config, checkpoint, device=device)
        self.logger.info("Initialization successfully completed")

    @torch.inference_mode()
    def estimate(self, image: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from mmpose.apis import inference_topdown
        results = inference_topdown(self.pose_estimator, image, bboxes)
        keypoints = np.concatenate(
            [r.pred_instances.keypoints for r in results], axis=0,
        )
        scores = np.concatenate(
            [r.pred_instances.keypoint_scores for r in results], axis=0,
        )
        return keypoints, scores


class PoseEstimationModel(LabelStudioMLBase):
    _pose_config: Path | None = Path(os.getenv("MMPOSE_CONFIG"))
    _pose_checkpoint: Path | None = Path(os.getenv("MMPOSE_CHECKPOINT"))
    _detector_checkpoint: Path = Path(os.getenv("YOLO_MODEL", "/app/models/yolo/yolo12x.pt"))
    score_thresh: float = float(os.getenv("SCORE_THRESHOLD", "0.3"))
    kpt_thresh: float = float(os.getenv("KPT_THRESHOLD", "0.3"))
    _device: str = os.getenv("DEVICE", "cuda:0")
    _dest: str | None = os.getenv("MODEL_DIR")
    _model_lock = threading.Lock()
    logger = logging.getLogger(__name__)

    def __init__(self, *args, **kwargs):
        self._detector: YOLODetector | None = None
        self._pose_estimator: MMPoseEstimator | None = None
        self.logger = logging.getLogger(__class__.__name__)
        super().__init__(*args, **kwargs)

    def setup(self) -> None:
        self.set("model_version", "1.0.0")

        if not self._pose_config or not self._pose_checkpoint:
            raise OSError("MMPOSE_CONFIG and MMPOSE_CHECKPOINT env vars must be set.",)

        with self._model_lock:
            if not self._pose_config.exists() or not self._pose_checkpoint.exists():
                mim.download("mmpose", [self._pose_config.stem],
                             dest_root=f"{self._dest}/mmpose")
                if not self._pose_checkpoint.exists():
                    raise FileNotFoundError(f"Doesn't found such file for env variable 'MMPOSE_CHECKPOINT'='{self._pose_checkpoint}'")
                if not self._pose_config.exists():
                    raise FileNotFoundError(f"Doesn't found such file for env variable 'MMPOSE_CONFIG'='{self._pose_config}'")

            if not self._detector_checkpoint.exists():
                self._detector_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(url=f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{self._detector_checkpoint.name}",
                               output=self._detector_checkpoint.as_posix(), quiet=False)
                if not self._detector_checkpoint.exists():
                    raise FileNotFoundError(f"Doesn't found such file for env variable 'YOLO_MODEL'='{self._detector_checkpoint}'")

        self.logger.info(f"device={self._device}, yolo={self._detector_checkpoint}, mmpose={self._pose_checkpoint}. "
                         f"Models load on first predict().")

    def _ensure_models_loaded(self) -> None:
        with self._model_lock:
            if self._detector is not None:
                return

            self.logger.info("Lazy models loading")
            self._detector = YOLODetector(self._detector_checkpoint.as_posix(), self._device, self.score_thresh,)
            self._pose_estimator = MMPoseEstimator(
                self._pose_config.as_posix(), self._pose_checkpoint.as_posix(), self._device,
            )
            self.logger.info("Models successfully loaded")

    def predict(self, tasks: list[dict[str, Any]], context: dict | None = None, **kwargs) -> ModelResponse:
        self._ensure_models_loaded()

        predictions: list[dict[str, Any]] = []
        for task in tasks:
            try:
                predictions.append(self._predict_single(task))
            except Exception as exc:
                self.logger.exception(f"Task {task.get('id')} failed: {exc}")
                predictions.append({"result": [], "score": 0.0})

        return ModelResponse(predictions=predictions)

    def _predict_single(self, task: dict[str, Any]) -> dict[str, Any]:
        image_url = task["data"].get("image")
        if not image_url:
            raise ValueError("No 'image' key in task data.")
        self.logger.debug(f"Task={task.get('id')}, url={image_url}")
        local_path = self.get_local_path(image_url, task_id=task.get("id"))
        self.logger.debug(f"local_path={local_path}")

        image = cv2.imread(local_path)
        if image is None:
            raise ValueError(f"Cannot load image from local path: {local_path}")

        img_h, img_w = image.shape[:2]
        self.logger.debug(f"Image loaded: {img_w}x{img_h}")

        bboxes = self._detector.detect(image)
        self.logger.debug(f"Detected {len(bboxes)} persons")
        if len(bboxes) == 0:
            return {"result": [], "score": 0.0}

        keypoints, scores = self._pose_estimator.estimate(image, bboxes)
        self.logger.debug(f"Pose done for {len(keypoints)} persons")

        results: list[dict[str, Any]] = []
        for bbox, kpts, kpt_scores in zip(bboxes, keypoints, scores):
            x1, y1, x2, y2 = map(float, bbox)
            rect_id = str(uuid.uuid4())
            mean_score = float(np.mean(kpt_scores))

            results.append(make_rectanglelabels(x1, y1, x2, y2, img_w, img_h, rect_id, mean_score))

            for kpt_name, kpt_xy, kpt_score in zip(HALPE26_KEYPOINTS, kpts, kpt_scores):
                if float(kpt_score) < self.kpt_thresh:
                    continue

                results.append(
                    make_keypointlabels(
                        kx=float(kpt_xy[0]), ky=float(kpt_xy[1]),
                        img_w=img_w, img_h=img_h,
                        kpt_name=kpt_name,
                        from_name=KPT_LABEL_FIELD[kpt_name],
                        rect_id=rect_id,
                        score=float(kpt_score),
                    ),
                )

        valid = scores[scores >= self.kpt_thresh]
        overall_score = float(np.mean(valid)) if len(valid) else 0.0

        self.logger.debug(f"Received {len(results)} items, score={overall_score:.3f}")
        return {
            "result": results,
            "score": overall_score,
            "model_version": self.get("model_version"),
        }

    def fit(self, event: str, data: dict, **kwargs) -> None:
        self.logger.warning(f"Event='{event}' — not implemented.")
