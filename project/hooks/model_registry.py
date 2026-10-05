import json
import re
import subprocess
import sys
import hashlib
import warnings
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.distributed as dist
from mlflow.tracking import MlflowClient
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


# paths relative to the git root, same as in dataset_release.sh
DVC_DIR = "project/annotations" # directory with dvc.yaml / dvc.lock
SPLITS_ROOT = "project/annotations/custom" # <SPLITS_ROOT>/<dataset_set>/train.json ...
SPLITS = ("train", "val", "test")
MLFLOW_3 = int(mlflow.__version__.split(".")[0]) >= 3


def run(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    # list of args, no shell: identical behaviour in cmd / PowerShell / Git Bash / Linux
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, check=check)


def git(*args: str, cwd: Path | None = None) -> str:
    return run("git", *args, cwd=cwd).stdout.strip()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_current(dataset_set: str) -> dict:
    root = Path(git("rev-parse", "--show-toplevel"))
    split_dir = f"{SPLITS_ROOT}/{dataset_set}"
    hashes = {s: md5(root / split_dir / f"{s}.json") for s in SPLITS}

    version = None
    tags = git("tag", "-l", f"dataset/{dataset_set}/v*", "--sort=-v:refname", cwd=root).splitlines()
    for tag in tags:
        msg = git("tag", "-l", "--format=%(contents)", tag, cwd=root)
        fields = dict(line.strip().split("=", 1) for line in msg.splitlines() if "=" in line)
        if all(fields.get(f"{s}_md5") == hashes[s] for s in SPLITS):
            version = tag
            break

    # files == committed dvc.lock  ->  the run can be restored with git checkout + dvc pull
    # dvc from the same interpreter: it may not be on PATH (Windows, IDE run configs)
    status = run(sys.executable, "-m", "dvc", "status", f"split_coco@{dataset_set}", "--quiet",
                 cwd=root / DVC_DIR, check=False)
    if status.returncode not in (0, 1):
        warnings.warn(f"`dvc status` failed (exit {status.returncode}): {status.stderr.strip()[:300]}")
    lock_committed = run("git", "diff", "--quiet", "HEAD", "--", f"{DVC_DIR}/dvc.lock",
                         cwd=root, check=False).returncode == 0
    clean = status.returncode == 0 and lock_committed
    return {
        "version": version or ("unreleased" if clean else "dirty"),
        "released": version is not None,
        "clean": clean,
        "hashes": hashes,
        "commit": git("rev-parse", "HEAD", cwd=root),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD", cwd=root),
        "root": root,
        "split_dir": split_dir,
    }


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    # MMDistributedDataParallel / DDP keep the real model in .module
    return model.module if hasattr(model, "module") else model


@HOOKS.register_module()
class MLflowModelRegistryHook(Hook):
    priority = "LOWEST" # after CheckpointHook has saved the best checkpoint

    def __init__(
        self,
        register_on_metric: str,
        dataset_set: str, # halpe26 | halpe136 — required, no silent default
        registered_model_name: str | None = None, # stable registry name, e.g. "RTMW-x-halpe136"
    ) -> None:
        self.register_on_metric = register_on_metric
        self.dataset_set = dataset_set
        self.registered_model_name = registered_model_name

        self._run_id: str | None = None
        self._vis_backend = None
        self._info: dict | None = None

        self._best_metric = -1.0
        self._best_epoch: int | None = None
        self._best_ckpt_path: Path | None = None

    def _find_mlflow_run(self, runner: Runner) -> None:
        for vis_backend in runner.visualizer._vis_backends.values():
            if hasattr(vis_backend, "_mlflow"):
                active = vis_backend._mlflow.active_run()
                if active:
                    self._run_id = active.info.run_id
                    self._vis_backend = vis_backend
                    return
        raise RuntimeError(
            f"[{self.__class__.__name__}] MLflow active run not found. "
            "Check that SafeMLflowVisBackend integrate in config visualizer."
        )

    def log_dataset_info(self) -> None:
        info = resolve_current(self.dataset_set)
        self._info = info
        split_dir = info["root"] / info["split_dir"]

        stats = {}
        for split in SPLITS:
            data = json.loads((split_dir / f"{split}.json").read_text(encoding="utf-8"))
            stats[split] = {"images": len(data.get("images", [])),
                            "annotations": len(data.get("annotations", []))}
        total_images = sum(v["images"] for v in stats.values())

        tags = {
            "dataset.set": self.dataset_set,
            "dataset.version": info["version"], # dataset/halpe26/v1.0 | unreleased | dirty
            "dataset.released": str(info["released"]),
            "dataset.reproducible": str(info["clean"]),
            "git.commit": info["commit"],
            "git.commit_short": info["commit"][:7],
            "git.branch": info["branch"],
            "repro.restore_cmd": (f"git checkout {info['version'] if info['released'] else info['commit']}"
                                  f" && cd {DVC_DIR} && dvc pull"),
        }
        for split in SPLITS:
            tags[f"dataset.{split}_md5"] = info["hashes"][split]
            tags[f"dataset.{split}_images"] = str(stats[split]["images"])
            tags[f"dataset.{split}_annotations"] = str(stats[split]["annotations"])
        if total_images:
            tags["dataset.split_ratio"] = "/".join(
                str(round(stats[s]["images"] / total_images, 2)) for s in SPLITS)
        mlflow.set_tags(tags)

        if not info["released"]:
            warnings.warn(f"Dataset version is '{info['version']}': run `./dataset_release.sh {self.dataset_set}` "
                          f"after committing dvc.lock, otherwise this run is not tied to a numbered dataset.")

        ref = info["version"] if info["released"] else info["commit"]
        for split in SPLITS:
            mlflow.log_input(self._dataset(split, ref), context=split)

    def _dataset(self, split: str, ref: str):
        return mlflow.data.meta_dataset.MetaDataset(
            source=mlflow.data.http_dataset_source.HTTPDatasetSource(
                url=f"dvc://{ref}/{self._info['split_dir']}/{split}.json"),
            name=f"{self.dataset_set}_{split}@{self._info['version']}",
            digest=self._info["hashes"][split][:8],
        )

    def before_run(self, runner: Runner) -> None:
        if not is_main_process():
            return
        # find the run FIRST: mlflow.set_tags without an active run would silently start a new one
        self._find_mlflow_run(runner)
        self.log_dataset_info()

    def _resolve_best_checkpoint(self, runner: Runner) -> None:
        best_score = runner.message_hub.get_info("best_score")
        best_ckpt = runner.message_hub.get_info("best_ckpt")

        if not best_ckpt:
            # newest file by modification time (lexicographic sort puts epoch_98 after epoch_268)
            candidates = sorted(Path(runner.work_dir).glob("best_*.pth"), key=lambda p: p.stat().st_mtime)
            best_ckpt = candidates[-1] if candidates else None
        if not best_ckpt:
            self._best_ckpt_path = None
            self._best_epoch = None
            return

        self._best_ckpt_path = Path(best_ckpt)
        if best_score is not None:
            self._best_metric = float(best_score)
        match = re.search(r"epoch_(\d+)", self._best_ckpt_path.name)
        self._best_epoch = int(match.group(1)) if match else runner.epoch + 1

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

        ckpt = torch.load(self._best_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)   # with EMAHook it already holds the EMA weights

        model = unwrap(runner.model)
        model.eval()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            runner.logger.warning(
                f"[{self.__class__.__name__}] load_state_dict mismatch — "
                f"missing: {len(missing)}, unexpected: {len(unexpected)}"
            )

        self._register_model(runner, model)

    def _register_model(self, runner: Runner, model: torch.nn.Module) -> None:
        artifact_name = f"epoch_{self._best_epoch:03d}"
        version_tag = self._info["version"]
        ref = version_tag if self._info["released"] else self._info["commit"]

        if MLFLOW_3:
            model_info = mlflow.pytorch.log_model(pytorch_model=model, name=artifact_name)
            model_uri = model_info.model_uri
            # metric is computed by the validation loop -> bind it to the val split
            mlflow.log_metrics(
                metrics={self.register_on_metric: self._best_metric},
                model_id=model_info.model_id,
                dataset=self._dataset("val", ref),
            )
        else:
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path=f"checkpoints/{artifact_name}")
            model_uri = f"runs:/{self._run_id}/checkpoints/{artifact_name}"

        name = self.registered_model_name or self._vis_backend._exp_name
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            tags={
                f"val.{self.register_on_metric}": str(round(self._best_metric, 4)),
                "epoch": str(self._best_epoch),
                "dataset.set": self.dataset_set,
                "dataset.version": version_tag,
                "dataset.test_md5": self._info["hashes"]["test"],
                "candidate": "true",
            },
        )
        # Stages (Staging/Archived) are deprecated and removed in MLflow 3. A new version is a candidate;
        # @champion is assigned by hand after evaluation on the test set:
        #   MlflowClient().set_registered_model_alias(name, "champion", <version>)
        MlflowClient().set_registered_model_alias(name, f"latest-{self.dataset_set}", mv.version)

        runner.logger.info(f"[{self.__class__.__name__}] Registered: {mv.name} v{mv.version} "
                           f"(@latest-{self.dataset_set}, dataset {version_tag})")
