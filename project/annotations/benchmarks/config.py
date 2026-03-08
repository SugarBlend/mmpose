from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ModelConfig:
    model_path: str
    config_path: str
    legend: str
    dataset_folder: str
    ann_file: str
    num_joints: int = 26

    def __post_init__(self):
        if not self.legend:
            self.legend = Path(self.model_path).stem

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ComparisonConfig:
    models: list[ModelConfig]
    dataset_path: str
    ann_file: str
    window_name:  str = "Pose Comparison"


@dataclass
class EvalConfig:
    models: list[ModelConfig]


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_eval_config(path: str) -> EvalConfig:
    data = load_yaml(path)
    models = [ModelConfig.from_dict(m) for m in data["models"]]
    return EvalConfig(models=models)


def load_comparison_config(path: str) -> ComparisonConfig:
    data = load_yaml(path)
    models = [ModelConfig.from_dict(m) for m in data["models"]]
    cmp = data.get("comparison", {})
    return ComparisonConfig(
        models=models,
        dataset_path=cmp.get("dataset_path", ""),
        ann_file=cmp.get("ann_file", ""),
        window_name=cmp.get("window_name", "Pose Comparison"),
    )
