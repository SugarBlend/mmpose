import copy
from mmengine.config import Config

space_name = "rtmpose-halpe"
name = f"{space_name}-x-384x288"
BASE_CONFIG = "configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py"

PROBE_EPOCHS = 25 # epoch on ine trial (≈30% from full 40)
FULL_EPOCHS = 100 # number of epochs for final training
N_TRIALS = 80 # number of configuration
N_GPUS = 2
N_CPUS = 12

# Number of custom datasets in train CombinedDataset (datasets_custom)
N_CUSTOM_DATASETS = 1


def patch_config(cfg: Config, params: dict, work_dir: str, max_epochs: int) -> Config:
    cfg = copy.deepcopy(cfg)

    cfg.visualizer = dict(
        type='PoseLocalVisualizer',
        vis_backends=[
            dict(
                type='SafeMLflowVisBackend',
                exp_name='rtmpose-halpe-x-raytune',
                tracking_uri='http://127.0.0.1:8082',
            )
        ],
        name='visualizer'
    )
    cfg.train_dataloader.batch_size = params["batch_size"]
    cfg.auto_scale_lr.base_batch_size = params["batch_size"]
    cfg.optim_wrapper.optimizer.lr = params["lr"]
    cfg.optim_wrapper.optimizer.weight_decay = params["weight_decay"]
    cfg.optim_wrapper.clip_grad["max_norm"] = params["clip_grad_norm"]

    stage2_num_epochs = max_epochs * params["stage2_num_epochs"]  # ~25% epochs for stage2

    # Save original struct for rtmpose-halpe-x schedulers LinearLR + CosineAnnealingLR,
    cosine_begin = int(max_epochs * params["cosine_begin_ratio"])
    cfg.param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=1.0e-5,
            by_epoch=False,
            begin=0,
            end=params["warmup_iters"], # ~1-2 epochs warmup 63 iter/epoch
        ),
        dict(
            type="CosineAnnealingLR",
            eta_min=params["lr"] * params["eta_min_ratio"],
            begin=cosine_begin,
            end=max_epochs,
            T_max=max_epochs - cosine_begin,
            by_epoch=True,
            convert_to_iter_based=True,
        ),
    ]

    def _patch_pipeline(pipeline: list) -> None:
        for t in pipeline:
            if not isinstance(t, dict):
                continue

            if t.get("type") == "RandomBBoxTransform":
                t["rotate_factor"] = params["rotate_factor"]
                t["scale_factor"] = [
                    params["scale_factor_min"],
                    params["scale_factor_max"],
                ]

            if t.get("type") == "RandomHalfBody":
                t["min_total_keypoints"] = 8

            if t.get("type") == "GenerateTarget":
                t["use_dataset_keypoint_weights"] = params["use_kp_weights"]

            if t.get("type") == "Albumentation":
                for aug in t.get("transforms", []):
                    if aug.get("type") == "CoarseDropout":
                        aug["p"] = params["coarse_dropout_p"]
                        aug["max_height"] = params["coarse_dropout_max_h"]
                        aug["max_width"] = params["coarse_dropout_max_w"]

    _patch_pipeline(cfg.train_dataloader.dataset.pipeline)

    # Overwrite custom_hooks: don't touch EMA and PipelineSwitch, replace StageUnfreezeHook
    existing_hooks = []
    for h in cfg.custom_hooks:
        htype = h.get("type")
        if htype in ("StageUnfreezeHook", "MLflowModelRegistryHook"):
            continue

        if htype == "EMAHook":
            h["momentum"] = params["ema_momentum"]

        if htype == "mmdet.PipelineSwitchHook":
            # Update switch_epoch for PipelineSwitch
            h["switch_epoch"] = max_epochs - stage2_num_epochs
            if h.get("switch_pipeline"):
                _patch_pipeline(h["switch_pipeline"])

        existing_hooks.append(h)

    # Start freezing for backbone layers
    cfg.model.backbone.frozen_stages = params["frozen_stages_init"]

    if params["frozen_stages_init"] > 0:
        # Gargantua for unfreeze_epoch_1 < unfreeze_epoch_2 in StageUnfreezeHook
        ue1 = int(max_epochs * params["unfreeze_stage_ratio_1"])
        new_frozen_stages = max(0, params["frozen_stages_init"] - 2)
        cfg.custom_hooks = existing_hooks + [
            dict(
                type="StageUnfreezeHook",
                unfreeze_epoch=ue1,
                frozen_stages=new_frozen_stages,
                priority="VERY_HIGH",
            ),
        ]

        if new_frozen_stages > 0:
            cfg.custom_hooks.append(
                dict(
                    type="StageUnfreezeHook",
                    unfreeze_epoch=ue1 + (max_epochs - ue1) * params["unfreeze_stage_ratio_2"],
                    frozen_stages=0,
                    priority="VERY_HIGH"
                )
            )

    # Update loss weights
    cfg.model.head.loss["beta"] = params["kl_beta"]
    cfg.model.head.loss["label_softmax"] = params["label_softmax"]

    cfg.model.head.gau_cfg["dropout_rate"] = params["gau_dropout"]
    cfg.model.head.gau_cfg["drop_path"] = params["gau_drop_path"]

    # update sigmas for gaussians in codec SimCC
    cfg.codec["sigma"] = (params["sigma_x"], params["sigma_y"])
    # Synchronize sigma in head.decoder (he refs on same codec dictionary)
    cfg.model.head.decoder["sigma"] = cfg.codec["sigma"]

    # Balance for datasets, order in train CombinedDataset
    # [custom×11, halpe, coco, ochuman, humanart_dance, humanart_drama, humanart_acrobatics, crowdpose]
    cfg.train_dataloader.dataset["sample_ratio_factor"] = [
        *[1.0] * N_CUSTOM_DATASETS,
        params["halpe_ratio"],
        params["coco_ratio"],
        params["humanart_ratio"],
        params["humanart_ratio"],
        params["humanart_ratio"],
        params["humanart_ratio"],
        params["crowdpose_ratio"],
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
