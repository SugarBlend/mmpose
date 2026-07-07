import cv2
import numpy as np
import torch.cuda
from typing import Any
from mmpose.models import RTMCCHead, HeatmapHead
from project.label_studio.pipelines.pipeline import MMPipeline


class HookedPipeline(object):
    def __init__(self, pipeline: MMPipeline) -> None:
        self.pipeline: MMPipeline = pipeline
        self.captured: dict[str, Any] = {}
        self._heatmap: np.ndarray | None = None
        self._overlay: np.ndarray | None = None
        self._monkey_patch()

    def _monkey_patch(self) -> None:
        model = self.pipeline.model

        if hasattr(model, "test_cfg"):
            model.test_cfg["flip_test"] = False

        original_extract = model.extract_feat
        original_forward = model.head.forward

        def extract(inputs, *args, **kwargs):
            self.captured["input"] = inputs.detach().cpu()
            return original_extract(inputs, *args, **kwargs)

        def forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            self.captured["head_output"] = out
            return out

        model.extract_feat = extract
        model.head.forward = forward

    def reverse_transform(self) -> None:
        mean = np.array(self.pipeline.model.cfg.model.data_preprocessor.mean, dtype=np.float32)
        std = np.array(self.pipeline.model.cfg.model.data_preprocessor.std, dtype=np.float32)
        img_np = self.input_tensor[0].permute(1, 2, 0).numpy()
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        if getattr(self.pipeline.model.cfg.model.data_preprocessor, 'bgr_to_rgb', False):
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        self._overlay = np.ascontiguousarray(img_np)

    def head_postprocess(self) -> None:
        w, h = self.pipeline.model_cfg.codec.input_size
        if isinstance(self.pipeline.head, RTMCCHead):
            pred_x, pred_y = self.head_output
            heatmap_2d = torch.einsum('kh,kw->khw', pred_y[0], pred_x[0])
            heatmap_2d = heatmap_2d / (heatmap_2d.amax() + 1e-9)
            _heatmap_2d = torch.nn.functional.interpolate(heatmap_2d[None], (h, w))
            self._heatmap = _heatmap_2d.squeeze().clamp(0, 1).detach().cpu()
        elif isinstance(self.pipeline.head, HeatmapHead):
            heatmap = self.head_output[0]
            h_min = heatmap.amin(dim=(1, 2), keepdim=True)
            h_max = heatmap.amax(dim=(1, 2), keepdim=True)
            self._heatmap = ((heatmap - h_min) / (h_max - h_min + 1e-9)).detach().cpu()

    def __call__(self, *args, **kwargs) -> Any:
        result = self.pipeline(*args, **kwargs)
        self.head_postprocess()
        self.reverse_transform()
        return result

    @property
    def heatmap(self) -> np.ndarray | None:
        return self._heatmap

    @property
    def input_tensor(self) -> torch.Tensor | None:
        return self.captured.get("input")

    @property
    def overlay(self) -> np.ndarray | None:
        return self._overlay

    @property
    def head_output(self) -> tuple[torch.Tensor] | torch.Tensor | None:
        return self.captured.get("head_output")
