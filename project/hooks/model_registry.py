import mlflow
import mlflow.pytorch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
import torch.distributed as dist


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


@HOOKS.register_module()
class MLflowModelRegistryHook(Hook):
    mlflow.enable_system_metrics_logging()

    def __init__(
            self,
            register_on_metric: str,
            # dataset_dvc_hash: str,
    ) -> None:
        self.register_on_metric = register_on_metric
        self._best_metric = -1.0
        self._run_id = None
        self._vis_backend = None

    def before_run(self, runner: Runner) -> None:
        if not is_main_process():
            return

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

    def after_val_epoch(self, runner: Runner, metrics: dict[str, float]) -> None:
        if not is_main_process():
            return

        current = metrics.get(self.register_on_metric, -1.0)
        previous_best = runner.message_hub.get_info("best_score") or -1.0
        if current <= previous_best:
            runner.logger.info(
                f"[{__class__.__name__}] skip to register model in mlflow, reason: current metric "
                f"'{self.register_on_metric}' - {round(current, 4)}, but highest - {round(previous_best, 4)}"
            )
            return

        self._best_metric = current
        runner.logger.info(
            f"[{__class__.__name__}] New best evaluation for '{self.register_on_metric}' - {current:.4f}, "
            f"try to register in mlflow."
        )
        self._register_model(runner)

    def _register_model(self, runner: Runner) -> None:
        model = runner.model
        model.eval()

        artifact_path = f'checkpoints/epoch_{runner.epoch + 1:03d}'

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
        )

        model.train()

        mv = mlflow.register_model(
            model_uri=f'runs:/{self._run_id}/{artifact_path}',
            name=self._vis_backend._exp_name,
            tags={
                self.register_on_metric: str(round(self._best_metric, 4)),
                'epoch': str(runner.epoch + 1),
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
            f"[{__class__.__name__}] Registered: {mv.name} - v{mv.version}"
        )
