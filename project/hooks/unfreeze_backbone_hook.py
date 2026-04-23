from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class StageUnfreezeHook(Hook):
    def __init__(self, unfreeze_epoch: int, frozen_stages: int) -> None:
        self.unfreeze_epoch: int = unfreeze_epoch
        self.frozen_stages: int = frozen_stages
        self._done = False

    def before_train_epoch(self, runner) -> None:
        if runner.epoch >= self.unfreeze_epoch and not self._done:
            model = runner.model
            if hasattr(model, "module"):
                model = model.module

            backbone = model.backbone

            runner.logger.info(
                f"Unfreezing backbone at epoch {runner.epoch}, frozen_stages={self.frozen_stages}"
            )
            backbone.frozen_stages = self.frozen_stages

            if hasattr(backbone, "_freeze_stages"):
                backbone._freeze_stages()

            for i, block in enumerate(backbone.layers):
                requires_grad = (i >= self.frozen_stages)
                for p in block.parameters():
                    p.requires_grad = requires_grad

            self._done = True
