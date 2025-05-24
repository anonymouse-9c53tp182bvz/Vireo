from mmseg.registry import DATASETS
from mmseg.datasets import CityscapesDataset


@DATASETS.register_module()
class DELIVER(CityscapesDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    METAINFO = dict(
        classes=[
            "Building",
            "Fence",
            "Other",
            "Pedestrian",
            "Pole",
            "RoadLine",
            "Road",
            "SideWalk",
            "Vegetation",
            "Cars",
            "Wall",
            "TrafficSign",
            "Sky",
            "Ground",
            "Bridge",
            "RailTrack",
            "GroundRail",
            "TrafficLight",
            "Static",
            "Dynamic",
            "Water",
            "Terrain",
            "TwoWheeler",
            "Bus",
            "Truck",
        ],
        palette=[
            [70, 70, 70],
            [100, 40, 40],
            [55, 90, 80],
            [220, 20, 60],
            [153, 153, 153],
            [157, 234, 50],
            [128, 64, 128],
            [244, 35, 232],
            [107, 142, 35],
            [0, 0, 142],
            [102, 102, 156],
            [220, 220, 0],
            [70, 130, 180],
            [81, 0, 81],
            [150, 100, 100],
            [230, 150, 140],
            [180, 165, 180],
            [250, 170, 30],
            [110, 190, 160],
            [170, 120, 50],
            [45, 60, 150],
            [145, 170, 100],
            [0, 0, 230],
            [0, 60, 100],
            [0, 0, 70],
        ],
    )

    def __init__(
        self,
        img_suffix="_rgb_front.png",
        seg_map_suffix="_semantic_front.png",
        **kwargs
    ) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
