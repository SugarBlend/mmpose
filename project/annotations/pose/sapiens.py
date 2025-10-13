from deploy2serve.deployment.projects.sapiens.model.preprocess import PosePreprocessor
from deploy2serve.deployment.projects.sapiens.model.postprocess import udp_decode
from deploy2serve.deployment.core.executors.backends.tensrt import TensorRTExecutor
import numpy as np
import torch
from typing import Optional, Tuple, List


class Sapiens(object):
    def __init__(
        self,
        checkpoints_path: str,
        input_shape: Tuple[int, int] = (768, 1024),
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16
    ) -> None:
        self.input_shape: Tuple[int, int] = input_shape
        self.device: str = device
        self.dtype: torch.dtype = dtype

        self.batched_data: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.centers: Optional[torch.Tensor] = None

        self.preprocessor = PosePreprocessor(
            input_shape, torch.tensor([123.675, 116.28, 103.53]), torch.tensor([58.395, 57.12, 57.375]),
        )
        self.model = TensorRTExecutor(checkpoints_path, self.device, "ERROR")

    def preprocess(self, tensor: torch.Tensor, bboxes: torch.Tensor) -> None:
        self.batched_data, self.centers, self.scales = self.preprocessor(tensor.to(self.device), bboxes.to(self.device))

    def postprocess(self, heatmaps: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        skeletons: List[np.ndarray] = []
        joints_scores: List[np.ndarray] = []
        for idx in range(heatmaps.shape[0]):
            keypoints, keypoint_scores = udp_decode(
                heatmaps[idx].float().cpu().numpy(),  # type: ignore[attr-defined]
                self.input_shape,
                np.array(self.input_shape) / 4,
            )
            keypoints = ((keypoints / self.preprocessor.input_shape) * self.scales[idx] + self.centers[idx]
                         - 0.5 * self.scales[idx])
            skeletons.append(keypoints.reshape(-1, 133, 2))
            joints_scores.append(keypoint_scores.reshape(-1, 133))
        return np.concatenate(skeletons, axis=0), np.concatenate(joints_scores, axis=0)

    @torch.inference_mode()
    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(image).to(device=self.device, dtype=self.dtype).permute(2, 0, 1)
        bboxes = torch.from_numpy(bboxes).to(device=self.device, dtype=self.dtype)
        self.preprocess(tensor, bboxes)
        output = self.model.infer(input_feed={"input": self.batched_data.to(self.dtype)},
                                  asynchronous=False)[0][:bboxes.shape[0]]
        joints, keypoint_scores = self.postprocess(output)
        return joints, keypoint_scores
