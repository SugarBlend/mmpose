import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from mmengine.config import Config

from search_space import SEARCH_SPACES
import logging
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.Logger(__name__)

MMPOSE_ROOT = Path(__file__).parents[2]
HPO_TMP_DIR = os.path.join(tempfile.gettempdir(), "hpo_trials")
os.makedirs(HPO_TMP_DIR, exist_ok=True)


def parse_best_metric(work_dir: str) -> float:
    import re
    best_auc = 0.0
    work_path = Path(work_dir)

    # Primary way: scalars.json (MMEngine vis_data)
    for scalars_file in work_path.rglob("scalars.json"):
        try:
            with open(scalars_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        auc = data.get("AUC", data.get("auc", 0.0))
                        best_auc = max(best_auc, float(auc))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
        except OSError:
            continue

    # Fallback: text .log file
    if best_auc == 0.0:
        for log_file in work_path.rglob("*.log"):
            try:
                with open(log_file, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "AUC" in line:
                            match = re.search(r"AUC[:\s]+([0-9.]+)", line)
                            if match:
                                best_auc = max(best_auc, float(match.group(1)))
            except OSError:
                continue

    return best_auc


def trainable(params: dict) -> None:
    work_dir = tempfile.mkdtemp(prefix="rtmpose_hpo_", dir=HPO_TMP_DIR)

    cfg = Config.fromfile(MMPOSE_ROOT.joinpath(patch.BASE_CONFIG))
    cfg = patch.patch_config(cfg, params, work_dir, max_epochs=patch.PROBE_EPOCHS)

    tmp_cfg = os.path.join(work_dir, "trial_config.py")
    cfg.dump(tmp_cfg)

    train_script = MMPOSE_ROOT.joinpath("tools/train.py")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = MMPOSE_ROOT.as_posix() + (os.pathsep + existing if existing else "")

    proc = subprocess.run(
        [sys.executable, train_script, "--config", tmp_cfg, "--work-dir", work_dir],
        cwd=MMPOSE_ROOT,
        env=env
    )

    if proc.returncode != 0:
        logger.error(f"[Trial ERROR] stdout: {proc.stdout}")
        logger.error(f"[Trial ERROR] stderr: {proc.stderr}")
        tune.report({"AUC": 0.0})
        return

    auc = parse_best_metric(work_dir)

    report = " | ".join([f"{k}:{v}" for k, v in params.items()])
    report += f" | AUC={auc:.4f}"
    logger.info(report)
    tune.report({"AUC": auc, "work_dir": work_dir})


def run_hpo(patch: "module") -> dict[str, Any] | None:
    ray.init(num_gpus=patch.N_GPUS, ignore_reinit_error=True)

    searcher = OptunaSearch(metric="AUC", mode="max")

    grace_period = int(patch.PROBE_EPOCHS / 3)
    scheduler = ASHAScheduler(
        metric="AUC",
        mode="max",
        max_t=patch.PROBE_EPOCHS,
        grace_period=grace_period,
        reduction_factor=3,
    )

    storage_path = MMPOSE_ROOT.joinpath(f"project/hpo/ray_results_{patch.name}")
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"gpu": patch.N_GPUS, "cpu": patch.N_CPUS}),
        param_space=SEARCH_SPACES[patch.space_name],
        tune_config=tune.TuneConfig(
            num_samples=patch.N_TRIALS,
            scheduler=scheduler,
            search_alg=searcher,
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:8]}",
        ),
        run_config=RunConfig(
            name=patch.name,
            storage_path=storage_path.as_posix(),
            verbose=1,
        ),
    )

    logger.info(f"[HPO] {patch.N_TRIALS} trial's × {patch.PROBE_EPOCHS} epochs  (final: {patch.FULL_EPOCHS} epochs)")
    logger.info(f"[HPO] ASHA grace_period={grace_period}, reduction_factor={scheduler._reduction_factor}")
    logger.info(f"[HPO] Results by: {storage_path}")

    results = tuner.fit()

    best = results.get_best_result(metric="AUC", mode="max")

    logger.info(f"Best AUC (by {patch.PROBE_EPOCHS} epochs): {best.metrics['AUC']:.4f}")
    logger.info("Best parameters:")
    for k, v in sorted(best.config.items()):
        logger.info(f"{k:22s} = {v:.6f}" if isinstance(v, float) else f"{k:22s} = {v}")

    logger.info("Top-5 trial's:")
    df = results.get_dataframe()
    cols = [
        "AUC",
        "config/lr", "config/weight_decay",
        "config/frozen_stages_init",
        "config/unfreeze_epoch_1", "config/unfreeze_epoch_2",
        "config/kl_beta",
        "config/sigma_x", "config/sigma_y",
        "config/halpe_ratio", "config/coco_ratio",
        "config/crowdpose_ratio", "config/humanart_ratio",
    ]
    cols = [c for c in cols if c in df.columns]
    logger.info(df.nlargest(5, "AUC")[cols].to_string(index=False))

    out_path = MMPOSE_ROOT.joinpath(f"project/hpo/best_params_{patch.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best.config, f, indent=2, ensure_ascii=False)

    logger.info(f"Best parameters: {out_path}")
    return best.config


def parse_args() -> "argparse.Namespace":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_script",
                        default="patches/rtmpose_halpe.py",
                        help="Path to patch file which consider all variables for tuner and patch function "
                             "for mmpose configuration.")
    return parser.parse_args()


def dynamic_loading(spec_name: str, source: str) -> None:
    import importlib
    spec = importlib.util.spec_from_file_location(spec_name, source)
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = package
    spec.loader.exec_module(package)
    return package


if __name__ == "__main__":
    args = parse_args()
    patch = dynamic_loading("patch", args.patch_script)
    run_hpo(patch)
