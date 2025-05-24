_base_ = [
    "./deliver-fog_512x512.py",
    "./deliver-night_512x512.py",
    "./deliver-rain_512x512.py",
    "./deliver-cloud_512x512.py",
    "./deliver-sun_512x512.py",
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_night_deliver}},
            {{_base_.val_cloud_deliver}},
            {{_base_.val_fog_deliver}},
            {{_base_.val_rain_deliver}},
            {{_base_.val_sun_deliver}},
        ],
    ),
)
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type="DefaultSampler", shuffle=False),
#     dataset={{_base_.val_cityscapes}},
# )
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["night/", "cloud/", "fog/", "rain/", "sun/"],
    mean_used_keys=["night/", "cloud/", "fog/", "rain/", "sun/"],
)
# test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
