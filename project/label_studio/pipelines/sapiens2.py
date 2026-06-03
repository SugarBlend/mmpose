import sys
import os

sys.modules["mmpretrain"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from mmengine.structures import InstanceData
from mmpose.structures.pose_data_sample import PoseDataSample
from pathlib import Path
from sapiens.pose import __file__ as sapiens2_root
from sapiens.pose.src.datasets import UDPHeatmap, parse_pose_metainfo
from sapiens.pose.src.models import init_model
from sapiens.pose.src.models.core.pose_topdown_estimator import PoseTopdownEstimator
import torch
from typing import Any, List


CONFIGS_DIR = Path(sapiens2_root).parent.joinpath("configs")

_CONFIG_TO_SIZE = {
    "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py": "0.4B",
    "sapiens2_0.8b_keypoints308_shutterstock_goliath_3po-1024x768.py": "0.8B",
    "sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py": "1B",
    "sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py": "5B",
}


class Sapiens2(object):
    POSE_MODELS = {
        "0.4B": {
            "repo": "facebook/sapiens2-pose-0.4b",
            "filename": "sapiens2_0.4b_pose.safetensors",
            "config": os.path.join(CONFIGS_DIR, "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"),
        },
        "0.8B": {
            "repo": "facebook/sapiens2-pose-0.8b",
            "filename": "sapiens2_0.8b_pose.safetensors",
            "config": os.path.join(CONFIGS_DIR, "sapiens2_0.8b_keypoints308_shutterstock_goliath_3po-1024x768.py"),
        },
        "1B": {
            "repo": "facebook/sapiens2-pose-1b",
            "filename": "sapiens2_1b_pose.safetensors",
            "config": os.path.join(CONFIGS_DIR, "sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py"),
        },
        "5B": {
            "repo": "facebook/sapiens2-pose-5b",
            "filename": "sapiens2_5b_pose.safetensors",
            "config": os.path.join(CONFIGS_DIR, "sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py"),
        },
    }

    def __init__(
        self,
        pose_checkpoint: str,
        pose_config: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.pose_checkpoint = pose_checkpoint
        self.pose_config = pose_config
        self.device = device
        self.dtype = dtype

        self._metainfo_cache: dict[str, Any] | None = None
        self.samples_list: list = []

        config_basename = os.path.basename(pose_config)
        self._size = _CONFIG_TO_SIZE.get(config_basename)

        self.model = self._load_model()
        self.model_cfg = self.model.cfg

    def _get_metainfo(self) -> dict[str, Any] | None:
        if self._metainfo_cache is None:
            meta_path = os.path.join(CONFIGS_DIR, "_base_", "keypoints308.py")
            self._metainfo_cache = parse_pose_metainfo(dict(from_file=meta_path))
        return self._metainfo_cache

    def _load_model(self) -> PoseTopdownEstimator:
        if os.path.isfile(self.pose_checkpoint):
            ckpt = self.pose_checkpoint
        else:
            assert self._size is not None, (
                f"Unable to determine model size from config '{self.pose_config}'. Please provide a local path to the "
                f"checkpoint or use the default config name."
            )
            spec = self.POSE_MODELS[self._size]
            ckpt = hf_hub_download(repo_id=spec["repo"], filename=spec["filename"])

        model = init_model(self.pose_config, ckpt, device=self.device)
        codec_cfg = dict(model.cfg.codec)
        assert codec_cfg.pop("type") == "UDPHeatmap"
        model.codec = UDPHeatmap(**codec_cfg)
        model.pose_metainfo = self._get_metainfo()
        return model

    def preprocess(self, image: np.ndarray, bboxes: np.ndarray) -> torch.Tensor:
        self.samples_list: list[dict[str, Any]] = []
        inputs_list: list[torch.Tensor] = []
        for bbox in bboxes:
            x, y, w, h = bbox
            bbox = np.array([x, y, x + w, y + h])
            data_info = dict(img=image, bbox=bbox[None], bbox_score=np.ones(1, dtype=np.float32))
            data = self.model.pipeline(data_info)
            data = self.model.data_preprocessor(data)
            inputs_list.append(data["inputs"])
            self.samples_list.append(data["data_samples"])
        return torch.cat(inputs_list, dim=0)

    def postprocess(self, heatmaps: torch.Tensor, bboxes: np.ndarray) -> List[PoseDataSample]:
        prediction = heatmaps.cpu().numpy()
        results: List[PoseDataSample] = []

        for i, sample in enumerate(self.samples_list):
            joints, scores = self.model.codec.decode(prediction[i])
            meta = sample["meta"]
            joints = joints / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"]
            visible = (scores > 0).astype(np.float32)

            pred_instances = InstanceData(
                keypoints=joints,
                keypoint_scores=scores,
                keypoints_visible=visible,
                bboxes=bboxes[i][None],
                bbox_scores=np.ones(1, dtype=np.float32),
            )

            # gt_instances needs for CocoMetric.process for bbox_scales and bbox_scores
            gt_instances = InstanceData(
                bboxes=bboxes[i][None],
                bbox_scores=np.ones(1, dtype=np.float32),
                bbox_scales=np.array(meta["bbox_scale"])[None],  # (1, 2)
            )

            data_sample = PoseDataSample()
            data_sample.pred_instances = pred_instances
            data_sample.gt_instances = gt_instances
            data_sample.set_metainfo(dict(
                input_size=meta["input_size"],
                input_center=meta["bbox_center"],
                input_scale=meta["bbox_scale"],
            ))
            results.append(data_sample)

        return results

    @torch.inference_mode()
    def __call__(
        self,
        data: np.ndarray | str,
        bboxes: np.ndarray | list[float],
    ) -> List[PoseDataSample]:
        if isinstance(data, str):
            data = cv2.imread(data)

        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)

        inputs = self.preprocess(data, bboxes)
        heatmaps = self.model(inputs)
        return self.postprocess(heatmaps, bboxes)
