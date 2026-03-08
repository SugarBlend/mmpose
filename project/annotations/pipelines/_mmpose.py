import numpy as np
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from typing import Union, Tuple
from pathlib import Path


class RTMPose(object):
    def __init__(
        self,
        pose_config: Union[str, Path],
        pose_checkpoint: Union[str, Path],
        device: str
    ) -> None:
        self.pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=device
        )

    def __call__(self, image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pose_results = inference_topdown(self.pose_estimator, image, bboxes)
        joints = np.concatenate([item.pred_instances.keypoints for item in pose_results], axis=0)
        joint_scores = np.concatenate([item.pred_instances.keypoint_scores for item in pose_results], axis=0)
        return joints, joint_scores
