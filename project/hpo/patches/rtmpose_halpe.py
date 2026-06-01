import copy
from mmengine.config import Config

space_name = "rtmpose-halpe"
name = f"{space_name}-x-384x288"
BASE_CONFIG = "configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py"

PROBE_EPOCHS = 12 # epoch on ine trial (≈30% from full 40)
FULL_EPOCHS = 40 # number of epochs for final training
N_TRIALS = 40 # number of configuration
N_GPUS = 2
N_CPUS = 12

# Number of custom datasets in train CombinedDataset (datasets_custom)
N_CUSTOM_DATASETS = 11


def patch_config(cfg: Config, params: dict, work_dir: str, max_epochs: int) -> Config:
    cfg = copy.deepcopy(cfg)

    cfg.visualizer = dict(
        type="PoseLocalVisualizer",
        vis_backends=[dict(type="LocalVisBackend")],
        name="visualizer"
    )

    cfg.optim_wrapper.optimizer.lr = params["lr"]
    cfg.optim_wrapper.optimizer.weight_decay = params["weight_decay"]

    # Save original struct for rtmpose-halpe-x schedulers LinearLR + CosineAnnealingLR,
    stage2_num_epochs = max(5, max_epochs // 4)  # ~25% epochs for stage2

    cfg.param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=1.0e-5,
            by_epoch=False,
            begin=0,
            end=100, # ~1-2 epochs warmup 63 iter/epoch
        ),
        dict(
            type="CosineAnnealingLR",
            eta_min=params["lr"] * 0.05,
            begin=max_epochs // 2,
            end=max_epochs,
            T_max=max_epochs // 2,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
    ]

    # Start freezing for backbone layers
    cfg.model.backbone.frozen_stages = params["frozen_stages_init"]

    # Gargantua for unfreeze_epoch_1 < unfreeze_epoch_2 in StageUnfreezeHook
    ue1 = params["unfreeze_epoch_1"]
    ue2 = params["unfreeze_epoch_2"]
    if ue2 <= ue1:
        ue2 = ue1 + 5

    # Both hooks must be executed within max_epochs
    # Margins: ue1 at least 6 epochs before the end, ue2 at least 3
    ue1 = min(ue1, max_epochs - 6)
    ue2 = min(ue2, max_epochs - 3)

    # Overwrite custom_hooks: don't touch EMA and PipelineSwitch, replace StageUnfreezeHook
    existing_hooks = [
        h for h in cfg.custom_hooks
        if h.get("type") not in ("StageUnfreezeHook", "MLflowModelRegistryHook")
    ]

    # Update switch_epoch for PipelineSwitch
    for h in existing_hooks:
        if h.get("type") == "mmdet.PipelineSwitchHook":
            h["switch_epoch"] = max_epochs - stage2_num_epochs

    cfg.custom_hooks = existing_hooks + [
        dict(
            type="StageUnfreezeHook",
            unfreeze_epoch=ue1,
            frozen_stages=max(0, params["frozen_stages_init"] - 2),
            priority="VERY_HIGH",
        ),
        dict(
            type="StageUnfreezeHook",
            unfreeze_epoch=ue2,
            frozen_stages=0,
            priority="VERY_HIGH",
        ),
    ]

    # Update loss weights
    cfg.model.head.loss["beta"] = params["kl_beta"]

    # update sigmas for gaussians in codec SimCC
    cfg.codec["sigma"] = (params["sigma_x"], params["sigma_y"])
    # Synchronize sigma in head.decoder (he refs on same codec dictionary)
    cfg.model.head.decoder["sigma"] = cfg.codec["sigma"]

    # Balance for datasets, order in train CombinedDataset
    # [custom×11, halpe, coco, ochuman, humanart_dance, humanart_drama, humanart_acrobatics, crowdpose]
    cfg.train_dataloader.dataset["sample_ratio_factor"] = [
        *[1.0] * N_CUSTOM_DATASETS, # custom — always 1.0
        params["halpe_ratio"], # dataset_halpe
        params["coco_ratio"], # dataset_coco
        params["humanart_ratio"], # dataset_ochuman
        params["humanart_ratio"], # humanart_dance
        params["humanart_ratio"], # humanart_drama
        params["humanart_ratio"], # humanart_acrobatics
        params["crowdpose_ratio"], # dataset_crowdpose
    ]

    # system settings
    cfg.train_cfg.max_epochs = max_epochs
    cfg.work_dir = work_dir

    cfg.log_processor = dict(
        type="LogProcessor",
        window_size=50,
        by_epoch=True,
        custom_cfg=[dict(data_src="", method="max", window_size="global")]
    )

    cfg.default_hooks["checkpoint"] = dict(
        type="CheckpointHook",
        save_best="AUC",
        rule="greater",
        max_keep_ckpts=1,
    )

    # disable visualization
    cfg.default_hooks["visualization"] = dict(
        type="PoseVisualizationHook",
        enable=False,
    )

    return cfg
