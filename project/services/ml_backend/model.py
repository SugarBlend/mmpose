import gc

from utils import make_rectanglelabels, make_keypointlabels, get_ls_fields
from logger import get_logger
import torch
from typing import Any

import os
import threading
import cv2
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from pathlib import Path
import uuid
from ultralytics import YOLO
import mim
import gdown


class YOLODetector(object):
    def __init__(self, model_path: str, device: str, score_threshold: float) -> None:
        self.logger = get_logger(__class__.__name__)
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
        self.logger = get_logger(__class__.__name__)
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
    score_thresh: float = float(os.getenv("SCORE_THRESHOLD", "0."))
    kpt_thresh: float = float(os.getenv("KPT_THRESHOLD", "0."))
    _device: str = os.getenv("DEVICE", "cuda:0")
    # Resolve paths at instance creation so they can be overridden via reload
    _pose_config: Path = Path(
        os.getenv("MMPOSE_CONFIG", f"{os.getcwd()}/models/mmpose/rtmw-x_8xb320-270e_cocktail14-384x288.py")
    )
    _pose_checkpoint: Path = Path(
        os.getenv("MMPOSE_CHECKPOINT", f"{os.getcwd()}/models/mmpose/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth")
    )
    _detector_checkpoint: Path = Path(os.getenv("YOLO_MODEL", f"{os.getcwd()}/models/yolo/yolo12x.pt"))

    _model_lock = threading.Lock()
    logger = get_logger(__name__)

    _is_ready: bool = False # True once models are loaded
    _load_error: str | None = None # last error message if loading failed
    _num_joints: int | None = None

    def __init__(self, *args, **kwargs) -> None:
        self._detector: YOLODetector | None = None
        self._pose_estimator: MMPoseEstimator | None = None

        super().__init__(*args, **kwargs)

    @classmethod
    @property
    def _status(cls) -> str:
        if cls._load_error:
            return "error"

        if cls._is_ready:
            return "ready"

        return "loading"

    def setup(self) -> None:
        self.set("model_version", "1.0.0")
        self._attempt_download()
        # Eager load — models are ready before the first request
        self._load_models(
            pose_config=self._pose_config,
            pose_checkpoint=self._pose_checkpoint,
            detector_checkpoint=self._detector_checkpoint,
        )

    def _attempt_download(self) -> None:
        with self._model_lock:
            if not self._pose_config.exists() or not self._pose_checkpoint.exists():
                mim.download("mmpose", [self._pose_config.stem], dest_root=f"{os.getcwd()}/models/mmpose")
                if not self._pose_checkpoint.exists():
                    raise FileNotFoundError(
                        f"File not found for MMPOSE_CHECKPOINT='{self._pose_checkpoint}'"
                    )
                if not self._pose_config.exists():
                    raise FileNotFoundError(
                        f"File not found for MMPOSE_CONFIG='{self._pose_config}'"
                    )

            if not self._detector_checkpoint.exists():
                self._detector_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                gdown.download(
                    url=f"https://github.com/ultralytics/assets/releases/download/v8.3.0/"
                        f"{self._detector_checkpoint.name}",
                    output=self._detector_checkpoint.as_posix(),
                    quiet=False,
                )
                if not self._detector_checkpoint.exists():
                    raise FileNotFoundError(
                        f"File not found for YOLO_MODEL='{self._detector_checkpoint}'"
                    )

    def _unload_models(self) -> None:
        self.logger.info("Try to unload previous models")
        if self._detector is not None:
            del self._detector.model
            del self._detector
            self._detector = None

        if self._pose_estimator is not None:
            del self._pose_estimator.pose_estimator
            del self._pose_estimator
            self._pose_estimator = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Previous models unloaded from memory")

    def _load_models(
        self,
        pose_config: Path,
        pose_checkpoint: Path,
        detector_checkpoint: Path,
    ) -> None:
        PoseEstimationModel._is_ready = False
        PoseEstimationModel._load_error = None

        try:
            with self._model_lock:
                self._unload_models()

                self.logger.info("Loading models")
                self._detector = YOLODetector(
                    detector_checkpoint.as_posix(), self._device, self.score_thresh
                )
                self._pose_estimator = MMPoseEstimator(
                    pose_config.as_posix(), pose_checkpoint.as_posix(), self._device
                )

                # Update instance paths so predict() uses the new files
                self._pose_config = pose_config
                self._pose_checkpoint = pose_checkpoint
                self._detector_checkpoint = detector_checkpoint

            PoseEstimationModel._is_ready = True
            self.logger.info("Models successfully loaded and ready")

        except Exception as exc:
            PoseEstimationModel._load_error = str(exc)
            self.logger.exception(f"Model loading failed: {exc}")
            raise

    def reload_models(
        self,
        pose_config: str | None = None,
        pose_checkpoint: str | None = None,
        detector_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        self.logger.debug("Try to reload models!")
        new_pose_config = Path(pose_config) if pose_config else self._pose_config
        new_pose_checkpoint = Path(pose_checkpoint) if pose_checkpoint else self._pose_checkpoint
        new_detector_ckpt = Path(detector_checkpoint) if detector_checkpoint else self._detector_checkpoint
        self.logger.debug(f"Updated detector checkpoints: '{new_detector_ckpt}'")
        self.logger.debug(f"Updated pose estimator config: '{new_pose_config}'")
        self.logger.debug(f"Updated pose estimator checkpoints: '{new_pose_checkpoint}'")

        # Validate paths before touching anything
        missing = []
        for label, path in [
            ("pose_config", new_pose_config),
            ("pose_checkpoint", new_pose_checkpoint),
            ("detector_checkpoint", new_detector_ckpt),
        ]:
            if not path.exists():
                missing.append(f"{label}='{path}'")
        if missing:
            raise FileNotFoundError(f"Files not found: {', '.join(missing)}")

        self.logger.info(
            f"Hot-reload requested — "
            f"config={new_pose_config}, "
            f"checkpoint={new_pose_checkpoint}, "
            f"detector={new_detector_ckpt}"
        )
        self._load_models(new_pose_config, new_pose_checkpoint, new_detector_ckpt)

        return {
            "pose_config": str(new_pose_config),
            "pose_checkpoint": str(new_pose_checkpoint),
            "detector_checkpoint": str(new_detector_ckpt),
        }

    def predict(self, tasks: list[dict[str, Any]], context: dict | None = None, **kwargs) -> ModelResponse:
        if not PoseEstimationModel._is_ready:
            raise RuntimeError("Models are not ready yet. Check /health-ready for status.")

        predictions: list[PredictionValue] = []
        for task in tasks:
            try:
                predictions.append(self._predict_single(task))
            except Exception as exc:
                self.logger.exception(f"Task {task.get('id')} failed: {exc}")
                predictions.append(PredictionValue(result=[], score=0.))

        return ModelResponse(predictions=predictions)

    def _predict_single(self, task: dict[str, Any]) -> PredictionValue:
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
        if not len(bboxes):
            self.logger.debug("Doesn't found any joints, maybe you need to increase score threshold?")
            return PredictionValue(result=[], score=0.)

        keypoints, scores = self._pose_estimator.estimate(image, bboxes)
        self.logger.debug(f"Pose done for {len(keypoints)} persons")

        if len(keypoints):
            self._num_joints = len(keypoints[0, :, 0])
        else:
            self.logger.warning("In theory this shouldn't happen, but it might be worth checking the implementation "
                                "of the pose estimation pipeline, since the boxes are not empty.")
            return PredictionValue(result=[], score=0.)

        label_fields, joint_names = get_ls_fields(self._num_joints)

        results: list[dict[str, Any]] = []
        for bbox, kpts, kpt_scores in zip(bboxes, keypoints, scores):
            x1, y1, x2, y2 = map(float, bbox)
            rect_id = str(uuid.uuid4())
            mean_score = float(np.mean(kpt_scores))

            results.append(make_rectanglelabels(x1, y1, x2, y2, img_w, img_h, rect_id, mean_score))

            for kpt_name, kpt_xy, kpt_score in zip(joint_names, kpts, kpt_scores):
                if float(kpt_score) < self.kpt_thresh:
                    continue

                results.append(
                    make_keypointlabels(
                        kx=float(kpt_xy[0]), ky=float(kpt_xy[1]),
                        img_w=img_w, img_h=img_h,
                        kpt_name=kpt_name,
                        from_name=label_fields[kpt_name],
                        rect_id=rect_id,
                        score=float(kpt_score),
                    ),
                )

        valid = scores[scores >= self.kpt_thresh]
        overall_score = float(np.mean(valid)) if len(valid) else 0.0

        self.logger.debug(f"Received {len(results)} items, score={overall_score:.3f}")

        return PredictionValue(result=results, score=overall_score, model_version=self.get("model_version"))

    def fit(self, event: str, data: dict, **kwargs) -> None:
        self.logger.warning(f"Event='{event}' — not implemented.")
