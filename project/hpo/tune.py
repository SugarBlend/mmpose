import argparse
import os
import sys
from pathlib import Path

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
sys.path.insert(0, Path(__file__).parents[2].as_posix())

import json
import tempfile
from datetime import datetime
from mmengine.config import Config
import importlib
import hashlib
from search_space import SEARCH_SPACES
import logging
import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.integration.ray_train import TuneReportCallback
from tools.train import main as trainable, parse_args as train_parser
from typing import Any, Callable


logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.Logger(__name__)

MMPOSE_ROOT = Path(__file__).parents[2]
HPO_TMP_DIR = os.path.join(tempfile.gettempdir(), "hpo_trials")
os.makedirs(HPO_TMP_DIR, exist_ok=True)


def train_fn_per_worker(train_loop_config: dict) -> None:
    tune_config = train_loop_config["tune_config"]
    patch_script_path = train_loop_config["patch_script_path"]
    work_dir = train_loop_config["work_dir"]

    patch_module = dynamic_loading(patch_script_path)

    cfg = Config.fromfile(MMPOSE_ROOT.joinpath(patch_module.BASE_CONFIG))
    cfg = patch_module.patch_config(
        cfg, tune_config, work_dir, max_epochs=patch_module.PROBE_EPOCHS
    )

    rank = ray.train.get_context().get_world_rank()
    tmp_cfg_path = os.path.join(work_dir, f"trial_config_rank{rank}.py")
    cfg.dump(tmp_cfg_path)

    targs = train_parser([])
    targs.config = tmp_cfg_path
    targs.work_dir = work_dir
    targs.launcher = "pytorch"
    trainable(targs)


def make_trial_fn(patch_script_path: str, n_gpus: int) -> Callable[[dict], None]:
    def trial_fn(tune_config: dict[str, Any]) -> None:
        work_dir = tempfile.mkdtemp(prefix="rtmpose_trial_", dir=HPO_TMP_DIR)

        trainer = TorchTrainer(
            train_fn_per_worker,
            train_loop_config={
                "tune_config": tune_config,
                "patch_script_path": patch_script_path,
                "work_dir": work_dir,
            },
            scaling_config=ScalingConfig(num_workers=n_gpus, use_gpu=True),
            run_config=ray.train.RunConfig(
                name=f"train-trial_id={ray.tune.get_context().get_trial_id()}",
                callbacks=[TuneReportCallback()],
            ),
        )
        result = trainer.fit()
        metrics = result.metrics or {}
        tune.report(metrics)
    return trial_fn


def search_hyperparameters(patch: "module") -> dict[str, Any] | None:
    ray.init(
        num_gpus=patch.N_GPUS,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}},
    )
    grace_period = int(patch.PROBE_EPOCHS / 2)

    searcher = OptunaSearch(metric="AUC", mode="max")
    scheduler = ASHAScheduler(
        metric="AUC",
        mode="max",
        max_t=patch.PROBE_EPOCHS,
        grace_period=grace_period,
        reduction_factor=3,
    )

    storage_path = MMPOSE_ROOT.joinpath(f"project/hpo/ray_results_{patch.name}_{start_time}")
    trial_fn = make_trial_fn(f"{Path(__file__).parent}/{args.patch_script}", patch.N_GPUS)
    tuner = tune.Tuner(
        # tune.with_resources(trial_fn, resources={"gpu": patch.N_GPUS, "cpu": patch.N_CPUS}),
        trial_fn,
        param_space=SEARCH_SPACES[patch.space_name],
        tune_config=tune.TuneConfig(
            num_samples=patch.N_TRIALS,
            scheduler=scheduler,
            search_alg=searcher,
            max_concurrent_trials=1,
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:8]}",
        ),
        run_config=RunConfig(
            name=f"{patch.name}:{ray.tune.get_context().get_trial_id()}",
            storage_path=storage_path.as_posix(),
            verbose=1,
        ),
    )

    logger.info(f"[HPO] {patch.N_TRIALS} trial's × {patch.PROBE_EPOCHS} epochs  (final: {patch.FULL_EPOCHS} epochs)")
    logger.info(f"[HPO] Results by: {storage_path}")

    results = tuner.fit()
    best = results.get_best_result(metric="AUC", mode="max")

    logger.info(f"Best AUC (by {patch.PROBE_EPOCHS} epochs): {best.metrics['AUC']:.4f}")
    logger.info("Best parameters:")
    for k, v in sorted(best.config.items()):
        logger.info(f"{k:22s} = {v:.6f}" if isinstance(v, float) else f"{k:22s} = {v}")

    logger.info("Top-5 trial's:")
    df = results.get_dataframe()
    cols = ["AUC"] + [c for c in df.columns if c.startswith("config/")]
    cols = [c for c in cols if c in df.columns]
    logger.info(df.nlargest(5, "AUC")[cols].to_string(index=False))

    out_path = MMPOSE_ROOT.joinpath(f"project/hpo/best_params_{patch.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best.config, f, indent=2, ensure_ascii=False)

    logger.info(f"Best parameters: {out_path}")
    return best.config


def parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_script",
                        default="patches/rtmpose_halpe.py",
                        help="Path to patch file which consider all variables for tuner and patch function "
                             "for mmpose configuration.")
    return parser.parse_args()


def dynamic_loading(source: str) -> None:
    spec_name = "_dynpatch_" + hashlib.md5(source.encode()).hexdigest()[:10]
    if spec_name in sys.modules:
        return sys.modules[spec_name]

    spec = importlib.util.spec_from_file_location(spec_name, source)
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = package
    spec.loader.exec_module(package)
    return package

# simple version without supporting ddp
# def make_trial_fn(patch_script_path) -> Callable[[dict], None]:
#     def trial_fn(tune_config: dict[str, Any]) -> None:
#         patch_module = dynamic_loading(patch_script_path)
#         work_dir = tempfile.mkdtemp(prefix="rtmpose_trial_", dir=HPO_TMP_DIR)
#
#         cfg = Config.fromfile(MMPOSE_ROOT.joinpath(patch_module.BASE_CONFIG))
#         cfg = patch_module.patch_config(
#             cfg, tune_config, work_dir, max_epochs=patch_module.PROBE_EPOCHS
#         )
#         tmp_cfg_path = os.path.join(work_dir, "trial_config.py")
#         cfg.dump(tmp_cfg_path)
#
#         targs = train_parser([])
#         targs.config = tmp_cfg_path
#         targs.work_dir = work_dir
#
#         trainable(targs)
#     return trial_fn


if __name__ == "__main__":
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    args = parse_args()
    search_hyperparameters(dynamic_loading(Path(__file__).parent.joinpath(args.patch_script).as_posix()))
