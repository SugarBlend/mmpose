from ray import tune


# Shared dataset-ratio space (same for both architectures)
_DATASET_RATIO_SPACE = dict(
    # Standard public datasets — tune down to avoid drowning 3k custom samples
    halpe_ratio=tune.uniform(0.02, 0.12), # 47k * 0.05 ≈ 2.3k ≈ custom
    coco_ratio=tune.uniform(0.01, 0.15), # 65k * 0.04 ≈ 2.6k ≈ custom
    crowdpose_ratio=tune.uniform(0.05, 0.25), # 12k * 0.15 ≈ 1.8k ≈ custom
    humanart_ratio=tune.uniform(0.08, 0.35), # ~5k * 0.5  ≈ 2.5k ≈ custom
    ochuman_ratio  = tune.uniform(0.05, 0.80),
)

RTMPOSE_SPACE = dict(
    batch_size=tune.choice([32, 48, 64, 96, 128, 192, 256]),
    stage2_num_epochs=tune.choice([0.1, 0.25, 0.4, 0.65, 0.8, 0.95]), # begin of second stage in percent relative to max epochs
    lr=tune.loguniform(3e-6, 1e-4), # base_lr neighbourhood
    weight_decay=tune.uniform(0.005, 0.2), # for optimizer
    clip_grad_norm= tune.choice([10, 20, 35, 50]),

    # Scheduler
    # for LinerLR
    warmup_iters=tune.choice([50, 100, 200, 300]), # peace from start to end point when lr increase linearly
    # for CosineAnnealingLR
    cosine_begin_ratio=tune.uniform(0.3, 0.8),  # begin=max_epochs * ratio
    eta_min_ratio=tune.uniform(0.01, 0.1),  # eta_min = lr * ratio

    # RTMPose backbone (CSPNeXt) uses integer stage indices
    frozen_stages_init=tune.choice([0, 1, 2, 3, 4]),
    unfreeze_stage_ratio_1=tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]), # first unfreeze
    unfreeze_stage_ratio_2=tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]), # full unfreeze (clamped in patch)

    # params for one-hot coding by using SimCC codec, KL-divergence beta (temperature-like sharpness)
    kl_beta=tune.uniform(5.0, 20.0),
    sigma_x=tune.uniform(2.0, 9.0), # horizontal spread (px-level)
    sigma_y=tune.uniform(2.0, 9.0), # vertical spread
    label_softmax = tune.choice([True, False]),

    # GAU in head, parameter for regularization
    gau_dropout=tune.uniform(0.0, 0.15),
    gau_drop_path=tune.uniform(0.0, 0.15),

    # EMA
    ema_momentum=tune.loguniform(5e-4, 5e-3),

    # Augmentation on first stage
    coarse_dropout_p=tune.uniform(0.1, 1.0),
    coarse_dropout_max_h=tune.uniform(0.2, 0.5),
    coarse_dropout_max_w=tune.uniform(0.2, 0.5),
    rotate_factor=tune.choice([15, 30, 45, 60, 90]),
    scale_factor_min=tune.uniform(0.3, 0.7),
    scale_factor_max=tune.uniform(1.15, 1.8),
    use_kp_weights=tune.choice([True, False]),
    half_body_prob=tune.uniform(0.1, 0.4),

    **_DATASET_RATIO_SPACE,
)

#TODO: refresh under new/updated fields
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

SEARCH_SPACES = {
    "rtmpose-halpe": RTMPOSE_SPACE,
    "vitpose-halpe": VITPOSE_SPACE,
}
