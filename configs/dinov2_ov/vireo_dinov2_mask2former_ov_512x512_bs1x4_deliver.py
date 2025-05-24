# dataset config
_base_ = [
    "../_base_/ov_datasets/dg_citys2deliver_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/vireo_dinov2_mask2former.py",
]

num_classes = 25

model = dict(
    backbone=dict(
        vireo_config=dict(
            class_json="open_vocab/deliver.json",
        ),
    ),
    decode_head=dict(
        class_json="open_vocab/deliver.json",
        num_classes=num_classes,
    ),
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
