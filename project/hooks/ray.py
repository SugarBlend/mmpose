import ray.train
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class RayReporter(Hook):

    def after_val_epoch(self, runner, metrics=None):
        ray.train.report(metrics or {})
