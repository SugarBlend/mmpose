from ray import tune


# Shared dataset-ratio space (same for both architectures)
_DATASET_RATIO_SPACE = dict(
    # Standard public datasets — tune down to avoid drowning 3k custom samples
    halpe_ratio=tune.uniform(0.02, 0.10), # 47k * 0.05 ≈ 2.3k ≈ custom
    coco_ratio=tune.uniform(0.01, 0.08), # 65k * 0.04 ≈ 2.6k ≈ custom
    crowdpose_ratio=tune.uniform(0.05, 0.25), # 12k * 0.15 ≈ 1.8k ≈ custom
    humanart_ratio=tune.uniform(0.20, 0.80), # ~5k * 0.5  ≈ 2.5k ≈ custom
)

RTMPOSE_SPACE = dict(
    lr=tune.loguniform(5e-6, 5e-5), # base_lr neighbourhood
    weight_decay=tune.uniform(0.01, 0.1),
    # RTMPose backbone (CSPNeXt) uses integer stage indices
    frozen_stages_init=tune.choice([0, 1, 2, 3, 4]),
    unfreeze_epoch_1=tune.choice([3, 5, 8, 10]), # first unfreeze
    unfreeze_epoch_2=tune.choice([6, 12, 15, 20, 25]), # full unfreeze (clamped in patch)
    # SimCC KL-divergence beta (temperature-like sharpness)
    kl_beta=tune.uniform(5.0, 20.0),
    sigma_x=tune.uniform(4.0, 8.0), # horizontal spread (px-level)
    sigma_y=tune.uniform(4.0, 8.5), # vertical spread
    **_DATASET_RATIO_SPACE,
)

VITPOSE_SPACE = dict(
    lr=tune.loguniform(5e-6, 2e-4), # config default 4.15e-5
    weight_decay=tune.uniform(0.02, 0.25), # default 0.05
    layer_decay_rate=tune.uniform(0.60, 0.90), # default 0.75
    drop_path_rate=tune.uniform(0.1, 0.5), # default 0.3; higher = more reg
    # ViT-Base has 12 transformer blocks (stages 0-11)
    frozen_stages_init=tune.choice([4, 6, 8, 10, 12]), # start heavily frozen on small data
    unfreeze_epoch_1=tune.randint(5, 15), # partial unfreeze
    unfreeze_epoch_2=tune.randint(15, 30), # full unfreeze (clamped in patch)

    # Single isotropic sigma (px in heatmap space 48×64)
    sigma=tune.uniform(1.5, 3.5), # default 2.0
    **_DATASET_RATIO_SPACE,
)

SEARCH_SPACES = dict(
    rtmpose_halpe=RTMPOSE_SPACE,
    vitpose_halpe=VITPOSE_SPACE,
)
