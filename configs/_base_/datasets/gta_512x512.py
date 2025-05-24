gta_type = "CityscapesDataset"
gta_root = "./data/gta"
# gta_root = "data/gta/"
gta_crop_size = (512, 512)
gta_train_pipeline = [
    dict(type="LoadImageFromFile", imdecode_backend="pillow"),
    dict(type="LoadAnnotations", imdecode_backend="pillow"),
    dict(type="Resize", scale=(1280, 720), backend="pillow"),
    dict(type="RandomCrop", crop_size=gta_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
gta_test_pipeline = [
    dict(type="LoadImageFromFile", imdecode_backend="pillow"),
    dict(type="LoadAnnotations", imdecode_backend="pillow"),
    dict(type="Resize", scale=(1280, 720), backend="pillow"),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="PackSegInputs"),
]
train_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_train_pipeline,
)
val_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_test_pipeline,
)
