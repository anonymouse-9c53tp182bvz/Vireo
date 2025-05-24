from mmseg.registry import DATASETS
from mmseg.datasets import ADE20KDataset
from .adefull_class import ADE20K_CLASSES
from .adefull_palette import ADE20K_PALETTE


@DATASETS.register_module()
class OpenADE20kfullDataset(ADE20KDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    METAINFO = dict(
        classes=ADE20K_CLASSES,
        palette=ADE20K_PALETTE,
    )

    def __init__(
        self,
        img_suffix=".jpg",
        seg_map_suffix=".tif",
        reduce_zero_label=False,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )
