rain_deliver_type = "DELIVER"
rain_deliver_root = "./data/DELIVER/"

rain_deliver_test_pipeline = [
    dict(type="LoadImageFromFile", imdecode_backend="pillow"),
    dict(type="Resize", scale=(1024, 1024)),
    dict(type="LoadAnnotations", imdecode_backend="pillow"),
    dict(type="PackSegInputs"),
]

val_rain_deliver = dict(
    type=rain_deliver_type,
    data_root=rain_deliver_root,
    data_prefix=dict(
        img_path="img/rain/test",
        seg_map_path="semantic/rain/test",
    ),
    img_suffix="_rgb_front.png",
    seg_map_suffix="_semantic_front.png",
    pipeline=rain_deliver_test_pipeline,
)
