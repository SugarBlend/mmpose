from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class CompileModelHook(Hook):
    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.backbone.compile(mode='max-autotune-no-cudagraphs')
        model.neck.compile(mode='max-autotune-no-cudagraphs')
        model.head.compile(mode='max-autotune-no-cudagraphs')
