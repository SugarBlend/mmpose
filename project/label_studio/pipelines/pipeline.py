import cv2
from deploy2serve.deployment.core.executors.backends.tensrt import TensorRTExecutor
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_codebase_config, load_config
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.structures import BaseDataElement, InstanceData
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.models import builder
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xywh2xyxy
import numpy as np
from itertools import zip_longest
from pathlib import Path
from typing import List, Sequence
import torch.cuda


class MMPipeline(object):
    def __init__(
        self,
        pose_checkpoint: str,
        pose_config: str,
        device: str = "cuda:0"
    ) -> None:
        self.pose_checkpoint: str = pose_checkpoint
        self.pose_config: str = pose_config

        self.model_cfg, = load_config(self.pose_config)
        deploy_cfg = dict(
            backend_config = dict(type="tensorrt"),
            codebase_config = dict(type="mmpose", task="PoseDetection")
        )

        self.task_processor = build_task_processor(self.model_cfg, Config(deploy_cfg), device)

        if self.pose_checkpoint.endswith(".pth"):
            self.model = init_pose_estimator(self.pose_config, self.pose_checkpoint, device=device)
            self.model.eval()
            self.model.cuda()
        elif self.pose_checkpoint.endswith(".engine"):
            self.model = TensorRTExecutor(self.pose_checkpoint, device, "ERROR")
        else:
            raise NotImplementedError(f"Weight file extension '{Path(pose_checkpoint).suffix}' is not supported. "
                                      f"Only 'pth', 'engine' formats are currently supported.")

        self.head = builder.build_head(self.model_cfg.model.head) if hasattr(self.model_cfg.model, 'head') else None
        self.codebase_cfg = get_codebase_config(Config(deploy_cfg))
        self.codec = self.model_cfg.codec
        if isinstance(self.codec, (list, tuple)):
            self.codec = self.codec[-1]

        self.pipeline = Compose(self.model_cfg.test_dataloader.dataset.pipeline)

        mean = self.model_cfg.model.data_preprocessor.mean
        std = self.model_cfg.model.data_preprocessor.std
        self._mean = torch.tensor(mean, dtype=torch.float32).reshape(3, 1, 1).cuda()
        self._std = torch.tensor(std, dtype=torch.float32).reshape(3, 1, 1).cuda()

    # bboxes in "xywh" format
    def __call__(self, image_path: str, bboxes: List[List[float]]) -> List[PoseDataSample]:
        if self.pose_checkpoint.endswith("pth"):
            results = inference_topdown(self.model, image_path, bboxes, bbox_format="xywh")
        else:
            img = cv2.imread(image_path)
            if isinstance(bboxes, list):
                bboxes = np.array(bboxes)
            bboxes = bbox_xywh2xyxy(bboxes)

            data_list = []
            for bbox in bboxes:
                if isinstance(img, str):
                    data_info = dict(img_path=img)
                else:
                    data_info = dict(img=img)
                data_info['bbox'] = bbox[None]
                data_info['bbox_score'] = np.ones(1, dtype=np.float32)
                data_list.append(self.pipeline(data_info))
            batch = pseudo_collate(data_list)
            inputs = torch.stack(batch["inputs"]).float().cuda()
            batch["inputs"] = (inputs[:, [2, 1, 0]] - self._mean) / self._std

            batch_outputs = self.model.infer({"input": batch["inputs"]})
            if self.codec.type == "YOLOXPoseAnnotationProcessor":
                raise NotImplementedError("Check realization in mmdeploy")
            elif self.codec.type == "SimCCLabel":
                export_postprocess = self.codebase_cfg.get("export_postprocess", False)
                if export_postprocess:
                    keypoints, scores = [_.cpu().numpy() for _ in batch_outputs]
                    predicts = [
                        InstanceData(keypoints=keypoints, keypoint_scores=scores)
                    ]
                else:
                    batch_predict_x, batch_predict_y = batch_outputs
                    predicts = self.head.decode((batch_predict_x, batch_predict_y))
            elif self.codec.type in ["RegressionLabel", "IntegralRegressionLabel"]:
                predicts = self.head.decode(batch_outputs)
            else:
                predicts = self.head.decode(batch_outputs[0])

            results = self.pack_result(predicts, batch["data_samples"])
        return results

    @staticmethod
    def pack_result(
            predicts: Sequence[InstanceData],
            data_samples: List[BaseDataElement],
            convert_coordinate: bool = True):
        """Pack pred results to mmpose format
        Args:
            predicts (Sequence[InstanceData]): Prediction of keypoints.
            data_samples (List[BaseDataElement]): A list of meta info for
                image(s).
            convert_coordinate (bool): Whether to convert keypoints
                coordinates to original image space. Default is True.
        Returns:
            data_samples (List[BaseDataElement]):
                updated data_samples with predictions.
        """
        if isinstance(predicts, tuple):
            batch_predict_instances, predicts_fields = predicts
        else:
            batch_predict_instances = predicts
            predicts_fields = None
        assert len(batch_predict_instances) == len(data_samples)
        if predicts_fields is None:
            predicts_fields = []

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_predict_instances, predicts_fields, data_samples):

            gt_instances = data_sample.gt_instances
            # convert keypoint coordinates from input space to image space
            if convert_coordinate:
                input_size = data_sample.metainfo['input_size']
                input_center = data_sample.metainfo['input_center']
                input_scale = data_sample.metainfo['input_scale']
                keypoints = pred_instances.keypoints
                keypoints = keypoints / input_size * input_scale
                keypoints += input_center - 0.5 * input_scale
                pred_instances.keypoints = keypoints

            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                data_sample.predicts_fields = pred_fields

        return data_samples
