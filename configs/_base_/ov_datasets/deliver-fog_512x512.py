fog_deliver_type = "DELIVER"
fog_deliver_root = "./data/DELIVER/"

fog_deliver_test_pipeline = [
    dict(type="LoadImageFromFile", imdecode_backend="pillow"),
    dict(type="Resize", scale=(1024, 1024)),
    dict(type="LoadAnnotations", imdecode_backend="pillow"),
    dict(type="PackSegInputs"),
]

val_fog_deliver = dict(
    type=fog_deliver_type,
    data_root=fog_deliver_root,
    data_prefix=dict(
        img_path="img/fog/test",
        seg_map_path="semantic/fog/test",
    ),
    img_suffix="_rgb_front.png",
    seg_map_suffix="_semantic_front.png",
    pipeline=fog_deliver_test_pipeline,
)
