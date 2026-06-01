import copy
from mmengine.config import Config

space_name = "vitpose-base-simple"
name = f"{space_name}-256x192"
BASE_CONFIG = "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py"

PROBE_EPOCHS = 15   # epochs per trial  (≈37 % of full 40-epoch run)
FULL_EPOCHS = 40   # epochs for the final training run
N_TRIALS = 30   # fewer trials than rtmpose: ViT is heavier per epoch
N_GPUS = 2
N_CPUS = 12

# Number of custom datasets in train CombinedDataset (datasets_custom list)
N_CUSTOM_DATASETS = 11


def patch_config(cfg: Config, params: dict, work_dir: str, max_epochs: int) -> Config:
    cfg = copy.deepcopy(cfg)

    cfg.visualizer = dict(
        type="PoseLocalVisualizer",
        vis_backends=[dict(type="LocalVisBackend")],
        name="visualizer",
    )

    cfg.optim_wrapper.optimizer.lr = params["lr"]
    cfg.optim_wrapper.optimizer.weight_decay = params["weight_decay"]
    cfg.optim_wrapper.paramwise_cfg["layer_decay_rate"] = params["layer_decay_rate"]

    # Base config uses LinearLR warmup + MultiStepLR with milestones [170, 200]
    # designed for 210 epochs. For probe runs we scale milestones proportionally
    # so the shape of the schedule stays the same (drops at ~81% and ~95% of run).
    stage2_num_epochs = max(5, max_epochs // 4)
    milestone_1 = max(1, int(round(max_epochs * 170 / 210)))  # ~81%
    milestone_2 = max(milestone_1 + 1, int(round(max_epochs * 200 / 210)))  # ~95%
    cfg.param_scheduler = [
        dict(
            type="LinearLR",
            begin=0,
            end=500,
            start_factor=0.001,
            by_epoch=False,  # warmup — identical to base config
        ),
        dict(
            type="MultiStepLR",
            begin=0,
            end=max_epochs,
            milestones=[milestone_1, milestone_2],
            gamma=0.1,
            by_epoch=True,
        ),
    ]

    cfg.model.backbone.drop_path_rate = params["drop_path_rate"]
    cfg.model.backbone.frozen_stages = params["frozen_stages_init"]

    # Ensure ue1 < ue2 and both fit within max_epochs
    ue1 = params["unfreeze_epoch_1"]
    ue2 = params["unfreeze_epoch_2"]
    if ue2 <= ue1:
        ue2 = ue1 + 5
    ue1 = min(ue1, max_epochs - 6)
    ue2 = min(ue2, max_epochs - 3)

    existing_hooks = [
        h for h in cfg.custom_hooks
        if h.get("type") not in ("StageUnfreezeHook", "MLflowModelRegistryHook")
    ]
    for h in existing_hooks:
        if h.get("type") == "mmdet.PipelineSwitchHook":
            h["switch_epoch"] = max_epochs - stage2_num_epochs

    # Partial unfreeze: reduce frozen_stages by 2 at ue1, full unfreeze at ue2
    partial_frozen = max(0, params["frozen_stages_init"] - 2)
    cfg.custom_hooks = existing_hooks + [
        dict(
            type="StageUnfreezeHook",
            unfreeze_epoch=ue1,
            frozen_stages=partial_frozen,
            priority="VERY_HIGH",
        ),
        dict(
            type="StageUnfreezeHook",
            unfreeze_epoch=ue2,
            frozen_stages=0,
            priority="VERY_HIGH",
        ),
    ]

    cfg.codec["sigma"] = params["sigma"]
    # HeatmapHead.decoder references the same codec dict
    cfg.model.head.decoder["sigma"] = params["sigma"]

    # Order matches base config train CombinedDataset:
    # [custom×N, halpe, coco, ochuman, humanart_dance, humanart_drama,
    #  humanart_acrobatics, crowdpose]
    cfg.train_dataloader.dataset["sample_ratio_factor"] = [
        *[1.0] * N_CUSTOM_DATASETS,
        params["halpe_ratio"],
        params["coco_ratio"],
        params["humanart_ratio"], # ochuman
        params["humanart_ratio"], # humanart_dance
        params["humanart_ratio"], # humanart_drama
        params["humanart_ratio"], # humanart_acrobatics
        params["crowdpose_ratio"],
    ]

    cfg.train_cfg.max_epochs = max_epochs
    cfg.work_dir = work_dir

    cfg.log_processor = dict(
        type="LogProcessor",
        window_size=50,
        by_epoch=True,
        custom_cfg=[dict(data_src="", method="max", window_size="global")],
    )
    cfg.default_hooks["checkpoint"] = dict(
        type="CheckpointHook",
        save_best="AUC",
        rule="greater",
        max_keep_ckpts=1,
    )
    cfg.default_hooks["visualization"] = dict(
        type="PoseVisualizationHook",
        enable=False,
    )

    return cfg
