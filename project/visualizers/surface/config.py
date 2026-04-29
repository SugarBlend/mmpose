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
class EvalConfig:
    models: list[ModelConfig]

    @classmethod
    def load(cls, path: str) -> "EvalConfig":
        with open(path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
        models = [ModelConfig.from_dict(m) for m in data["models"]]
        return EvalConfig(models=models)
