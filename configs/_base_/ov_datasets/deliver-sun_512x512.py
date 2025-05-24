sun_deliver_type = "DELIVER"
sun_deliver_root = "./data/DELIVER/"

sun_deliver_test_pipeline = [
    dict(type="LoadImageFromFile", imdecode_backend="pillow"),
    dict(type="Resize", scale=(1024, 1024)),
    dict(type="LoadAnnotations", imdecode_backend="pillow"),
    dict(type="PackSegInputs"),
]

val_sun_deliver = dict(
    type=sun_deliver_type,
    data_root=sun_deliver_root,
    data_prefix=dict(
        img_path="img/sun/test",
        seg_map_path="semantic/sun/test",
    ),
    img_suffix="_rgb_front.png",
    seg_map_suffix="_semantic_front.png",
    pipeline=sun_deliver_test_pipeline,
)
