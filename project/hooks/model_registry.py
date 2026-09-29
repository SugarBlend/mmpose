import mlflow
import mlflow.pytorch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
import torch
import torch.distributed as dist
import subprocess
import re
from pathlib import Path
import json


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


@HOOKS.register_module()
class MLflowModelRegistryHook(Hook):
    def __init__(
        self,
        register_on_metric: str,
    ) -> None:
        self.register_on_metric = register_on_metric

        self._run_id = None
        self._vis_backend = None

        self._best_metric = -1.0
        self._best_epoch = None
        self._best_ckpt_path: Path | None = None
        self._dataset_tag = None

    def log_dataset_info(self) -> None:
        git_commit = sh("git rev-parse HEAD")
        git_commit_short = sh("git rev-parse --short HEAD")
        git_branch = sh("git rev-parse --abbrev-ref HEAD")
        git_tag = sh("git tag --points-at HEAD") or "no-tag"
        git_dataset_tag = sh("git tag --list 'dataset/v*' --sort=-version:refname | head -1") or "unknown"
        self._dataset_tag = git_dataset_tag
        split_dir = Path("project/annotations/custom")

        stats = {}
        total_images = 0
        total_annotations = 0
        for split in ["train", "val", "test"]:
            path = split_dir / f"{split}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            images_count = len(data.get("images", []))
            annotations_count = len(data.get("annotations", []))
            categories_count = len(data.get("categories", []))
            total_images += images_count
            total_annotations += annotations_count
            stats[split] = {
                "images": images_count,
                "annotations": annotations_count,
                "categories": categories_count,
            }

        mlflow.set_tags({
            "git.commit": git_commit,
            "git.commit_short": git_commit_short,
            "git.branch": git_branch,
            "git.tag": git_tag,
            "dataset.version": git_dataset_tag,
            "dataset.train_images": stats.get("train", {}).get("images", 0),
            "dataset.train_annotations": stats.get("train", {}).get("annotations", 0),
            "dataset.val_images": stats.get("val", {}).get("images", 0),
            "dataset.val_annotations": stats.get("val", {}).get("annotations", 0),
            "dataset.test_images": stats.get("test", {}).get("images", 0),
            "dataset.test_annotations": stats.get("test", {}).get("annotations", 0),
            "dataset.total_images": total_images,
            "dataset.total_annotations": total_annotations,
            "repro.restore_cmd": f"git checkout {git_commit} && dvc pull",
        })

        if total_images > 0:
            mlflow.set_tag("dataset.split_ratio", "{}/{}/{}".format(
                round(stats.get("train", {}).get("images", 0) / total_images, 2),
                round(stats.get("val", {}).get("images", 0) / total_images, 2),
                round(stats.get("test", {}).get("images", 0) / total_images, 2),
            ))

    def before_run(self, runner: Runner) -> None:
        if not is_main_process():
            return

        self.log_dataset_info()

        for vis_backend in runner.visualizer._vis_backends.values():
            if hasattr(vis_backend, "_mlflow"):
                active = vis_backend._mlflow.active_run()
                if active:
                    self._run_id = active.info.run_id
                    self._vis_backend = vis_backend
                    break

        if self._run_id is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] MLflow active run not found. "
                "Check that SafeMLflowVisBackend integrate in config visualizer."
            )

    def _resolve_best_checkpoint(self, runner: Runner) -> None:
        best_score = runner.message_hub.get_info("best_score")
        best_ckpt = runner.message_hub.get_info("best_ckpt")

        if best_ckpt:
            self._best_metric = float(best_score) if best_score is not None else self._best_metric
            self._best_ckpt_path = Path(best_ckpt)
            match = re.search(r"epoch_(\d+)", Path(best_ckpt).name)
            self._best_epoch = int(match.group(1)) if match else runner.epoch + 1
            return

        candidates = sorted(Path(runner.work_dir).glob("best_*.pth"))
        if candidates:
            ckpt_path = candidates[-1]
            self._best_ckpt_path = ckpt_path
            match = re.search(r"epoch_(\d+)", ckpt_path.name)
            self._best_epoch = int(match.group(1)) if match else runner.epoch + 1
            if best_score is not None:
                self._best_metric = float(best_score)
            return

        self._best_ckpt_path = None
        self._best_epoch = None

    def after_run(self, runner: Runner) -> None:
        if not is_main_process():
            return

        self._resolve_best_checkpoint(runner)

        if self._best_epoch is None or self._best_ckpt_path is None:
            runner.logger.warning(
                f"[{self.__class__.__name__}] best checkpoint not found (check that "
                "CheckpointHook has save_best enabled), skip MLflow registration."
            )
            return

        if not self._best_ckpt_path.exists():
            runner.logger.warning(
                f"[{self.__class__.__name__}] best checkpoint file not found at "
                f"{self._best_ckpt_path}, skip MLflow registration."
            )
            return

        runner.logger.info(
            f"[{self.__class__.__name__}] loading best checkpoint (epoch {self._best_epoch}, "
            f"{self.register_on_metric}={self._best_metric:.4f}) and registering in MLflow."
        )

        ckpt = torch.load(self._best_ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        model = runner.model
        model.eval()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            runner.logger.warning(
                f"[{self.__class__.__name__}] load_state_dict mismatch — "
                f"missing: {len(missing)}, unexpected: {len(unexpected)}"
            )

        self._register_model(runner)

        model.train()

    def _register_model(self, runner: Runner) -> None:
        model = runner.model

        artifact_name = f"epoch_{self._best_epoch:03d}"

        try:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name=artifact_name,
            )
            model_uri = model_info.model_uri
            model_id = model_info.model_id

            train_dataset = mlflow.data.meta_dataset.MetaDataset(
                source=mlflow.data.http_dataset_source.HTTPDatasetSource(url="local://project/annotations/custom"),
                name=self._dataset_tag or "unknown",
            )
            mlflow.log_metrics(
                metrics={self.register_on_metric: self._best_metric},
                model_id=model_id,
                dataset=train_dataset,
            )
        except (AttributeError, TypeError) as e:
            runner.logger.warning(
                f"[{self.__class__.__name__}] LoggedModel API unavailable ({e}), "
                "falling back to legacy artifact_path-based logging."
            )
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"checkpoints/{artifact_name}",
            )
            model_uri = f"runs:/{self._run_id}/checkpoints/{artifact_name}"

        mv = mlflow.register_model(
            model_uri=model_uri,
            name=self._vis_backend._exp_name,
            tags={
                self.register_on_metric: str(round(self._best_metric, 4)),
                "epoch": str(self._best_epoch),
                # 'dvc_hash': self.dataset_dvc_hash,
            }
        )

        client = mlflow.tracking.MlflowClient()

        for v in client.get_latest_versions(self._vis_backend._exp_name, stages=["Staging"]):
            client.transition_model_version_stage(
                name=v.name, version=v.version, stage="Archived"
            )

        client.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="Staging"
        )

        runner.logger.info(
            f"[{self.__class__.__name__}] Registered: {mv.name} - v{mv.version}"
        )
