from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Optional
import yaml
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import maps


@dataclass
class ModelConfig:
    legend: str = field(
        metadata={
            "description": "Legend which will be used for separating different runs."
        }
    )
    model_path: str = field(
        metadata={
            "description": "Path to the model weights."
        }
    )
    config_path: str = field(
        metadata={
            "description": "Path to the configuration path of the current model."
        }
    )
    dataset_folder: str = field(
        metadata={
            "description": "Path to base folder, paths which provided in coco-annotations will be relative them folder."
                           "This folder may indicate on local filesystem or cloud object storage such as S3."
        }
    )
    ann_file: str = field(
        metadata={
            "description": "Path to the annotations file which created in coco-style."
        }
    )
    expected_joints: int = field(
        metadata={
            "description": "Expected number of output keypoints, this number - have undefined position in the model's "
                           "config, that's why need to pass manually. "
        }
    )
    anns_schema: str = field(
        default="coco_wholebody",
        metadata={
            "description": "Skeleton format of the annotation file. Must be registered in 'maps.SKELETON_SUBSETS'",
            "allowed_values": maps.SKELETON_SUBSETS.keys()
        }
    )

    gt_converter: str | None = field(
        default=None,
        metadata={
            "description": "The name of the key for converting ground truth keypoints in different forms between each "
                           "other, described in the structure: 'maps.converters'."
        }
    )
    pred_converter: str | None = field(
        default=None,
        metadata={
            "description": "The name of the key for converting prediction keypoints in different forms between each "
                           "other, described in the structure: 'maps.converters'."
        }
    )

    def __post_init__(self) -> None:
        # Basic path validation (warn, not crash — paths may be on remote FS)
        for attr in ("config_path", "ann_file"):
            path = Path(getattr(self, attr))
            if not path.exists():
                warnings.warn(f"[ModelConfig] '{attr}' does not exist: {path}", stacklevel=2)
                raise

        if self.anns_schema not in maps.SKELETON_SUBSETS.keys():
            warnings.warn("[ModelConfig] Value of 'anns_schema' is not possible, allowed "
                          f"combinations: {maps.SKELETON_SUBSETS}", stacklevel=2)
            raise

        # for attr in ("gt_converter", "pred_converter"):
        #     value = getattr(self, attr)
        #     if not maps.converters.get(value):
        #         warnings.warn(f"[ModelConfig] Value of '{attr}' is not possible, allowed "
        #                       f"combinations: {maps.converters.keys()}", stacklevel=2)
        #         raise


@dataclass
class RenderConfig:

    show_plot: bool = field(
        metadata={
            "description": "Flag for enable/disable visualization step."
        }
    )
    generate_html: bool = field(
        metadata={
            "description": "Flag for enable/disable storing results into html format."
        }
    )
    save_dir: str | None = field(
        metadata={
            "description": "Path to saved results."
        }
    )
    radar_xticks: List[float] = field(
        default_factory=list,
        metadata={
            "description": "Range for X axis."
        }
    )

    def __post_init__(self) -> None:
        if not self.save_dir:
            warnings.warn(f"[{self.__class__.__name__}] 'save_dir' is not provided, save step will be skipped.",
                          stacklevel=2)
        else:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        if not self.radar_xticks:
            warnings.warn(f"[{self.__class__.__name__}] 'radar_xticks' is not provided, program will "
                          f"generated them further.", stacklevel=2)


@dataclass
class COCOMetrics:
    iou_type: str = "keypoints"
    score_mode: str = "keypoint"
    keypoint_score_thr: float = 0.2
    nms_mode: float = 0.9
    format_only: bool = False
    use_area: bool = True
    iou_thresholds: List[float] = field(
        default_factory=list,
        metadata={
            "description": "Thresholds by which quality should be assessed"
        }
    )

    def __post_init__(self) -> None:
        if any(item > 1. for item in self.iou_thresholds):
            warnings.warn(f"[ModelConfig] Value of 'iou_thresholds' should not contain values outside the "
                          f"range [0, 1]", stacklevel=2)
            raise



@dataclass
class PCKMetrics:
    thresholds: List[float] = field(default_factory=list)
    norm_item: str = "bbox"
    keypoint_groups: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    COCO: COCOMetrics = field(default_factory=COCOMetrics)
    PCK: PCKMetrics = field(default_factory=PCKMetrics)


@dataclass
class EvalConfig:
    """
    Top-level evaluation configuration loaded from YAML.

    Attributes
    ----------
    models:
        List of ModelConfig instances to evaluate in sequence.
    radar_xticks:
        Y-axis ticks for the radar plot (values 0–1).
    save_dir:
        Directory where metrics.json and radar.png will be saved.
        None means do not save.
    show_plot:
        Whether to display the interactive radar plot window.
    """

    models: list[ModelConfig]
    visualization: RenderConfig
    metrics: MetricsConfig

    @classmethod
    def load(cls, path: str) -> "EvalConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        models: list[ModelConfig] = []
        for model in raw.get("models", []):
            models.append(ModelConfig(**model))

        return cls(
            models=models,
            visualization=RenderConfig(**raw.get("visualization")),
            metrics=MetricsConfig(**raw.get("metrics"))
        )
