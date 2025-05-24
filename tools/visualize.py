# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)

from mmengine.config import Config
from mmseg.utils import get_classes, get_palette
from mmengine.runner.checkpoint import _load_checkpoint
from vireo.utils import init_model
from mmseg.apis import inference_model
import vireo
import tqdm
import mmengine
import torch
import numpy as np
from PIL import Image

import warnings

warnings.filterwarnings("ignore")
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("config", help="Path to the training configuration file.")
    parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint file for both the vireo and head models.",
    )
    parser.add_argument(
        "images", help="Directory or file path of images to be processed."
    )
    parser.add_argument(
        "--suffix",
        default="_rgb_anon.png",
        help="File suffix to filter images in the directory. Default is '.png'.",
    )
    parser.add_argument(
        "--not-recursive",
        action="store_false",
        help="Whether to search images recursively in subfolders. Default is recursive.",
    )
    parser.add_argument(
        "--search-key",
        default="",
        help="Keyword to filter images within the directory. Default is no filtering.",
    )
    parser.add_argument(
        "--backbone",
        default="checkpoints/dinov2_vitl14_converted_1024x1024.pth",
        help="Path to the backbone model checkpoint. Default is 'checkpoints/dinov2_vitl14_converted_1024x1024.pth'.",
    )
    parser.add_argument(
        "--save_dir",
        default="work_dirs/show",
        help="Directory to save the output images. Default is 'work_dirs/show'.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test time augmentation. Default is disabled.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use for computation. Default is 'cuda:0'.",
    )
    args = parser.parse_args()
    return args


def load_backbone(checkpoint: dict, backbone_path: str) -> None:
    converted_backbone_weight = _load_checkpoint(backbone_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint["state_dict"].update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )
    else:
        checkpoint.update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )


classes = get_classes("cityscapes")
palette = get_palette("cityscapes")

# classes = [
#     "Building",
#     "Fence",
#     "Other",
#     "Pedestrian",
#     "Pole",
#     "RoadLine",
#     "Road",
#     "SideWalk",
#     "Vegetation",
#     "Cars",
#     "Wall",
#     "TrafficSign",
#     "Sky",
#     "Ground",
#     "Bridge",
#     "RailTrack",
#     "GroundRail",
#     "TrafficLight",
#     "Static",
#     "Dynamic",
#     "Water",
#     "Terrain",
#     "TwoWheeler",
#     "Bus",
#     "Truck",
# ]
# palette = [
#     [70, 70, 70],
#     [100, 40, 40],
#     [55, 90, 80],
#     [220, 20, 60],
#     [153, 153, 153],
#     [157, 234, 50],
#     [128, 64, 128],
#     [244, 35, 232],
#     [107, 142, 35],
#     [0, 0, 142],
#     [102, 102, 156],
#     [220, 220, 0],
#     [70, 130, 180],
#     [81, 0, 81],
#     [150, 100, 100],
#     [230, 150, 140],
#     [180, 165, 180],
#     [250, 170, 30],
#     [110, 190, 160],
#     [170, 120, 50],
#     [45, 60, 150],
#     [145, 170, 100],
#     [0, 0, 230],
#     [0, 60, 100],
#     [0, 0, 70],
# ]


def draw_sem_seg(sem_seg: torch.Tensor):
    num_classes = len(classes)
    sem_seg = sem_seg.data.squeeze(0)
    H, W = sem_seg.shape
    ids = torch.unique(sem_seg).cpu().numpy()
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    colors = [torch.tensor(color, dtype=torch.uint8).view(1, 1, 3) for color in colors]
    result = torch.zeros([H, W, 3], dtype=torch.uint8)
    for label, color in zip(labels, colors):
        result[sem_seg == label, :] = color
    return result.cpu().numpy()


# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if "test_pipeline" not in cfg:
        cfg.test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    2048,
                    1024,
                ),
                type="Resize",
            ),
            dict(type="PackSegInputs"),
        ]
    model = init_model(cfg, args.checkpoint, device=args.device)
    model = model.cuda(args.device)
    state_dict = model.state_dict()
    load_backbone(state_dict, args.backbone)
    model.load_state_dict(state_dict, strict=False)
    mmengine.mkdir_or_exist(args.save_dir)
    images = []
    if osp.isfile(args.images):
        images.append(args.images)
    elif osp.isdir(args.images):
        for im in mmengine.scandir(
            args.images, suffix=args.suffix, recursive=args.not_recursive
        ):
            if args.search_key in im:
                images.append(osp.join(args.images, im))
    else:
        raise NotImplementedError()
    print(f"Collect {len(images)} images")
    # for im_path in tqdm.tqdm(images):
    #     result = inference_model(model, im_path)
    #     pred = draw_sem_seg(result.pred_sem_seg)
    #     img = Image.open(im_path).convert("RGB")
    #     pred = Image.fromarray(pred).resize(
    #         [img.width, img.height], resample=Image.NEAREST
    #     )
    #     vis = Image.new("RGB", [img.width * 2, img.height])
    #     vis.paste(img, (0, 0))
    #     vis.paste(pred, (img.width, 0))
    #     vis.save(osp.join(args.save_dir, osp.basename(im_path)))
    # print(f"Results are saved in {args.save_dir}")

    # for im_path in tqdm.tqdm(images):
    #     result = inference_model(model, im_path)
    #     mask = draw_sem_seg(result.pred_sem_seg)

    #     orig_img_pil = Image.open(im_path).convert("RGB")
    #     orig_img = np.array(orig_img_pil)

    #     orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

    #     mask_pil = Image.fromarray(mask).resize(
    #         (orig_img.shape[1], orig_img.shape[0]), resample=Image.NEAREST
    #     )
    #     mask_np = np.array(mask_pil)

    #     mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_RGB2BGR)

    #     blended_bgr = cv2.addWeighted(orig_img_bgr, 0.3, mask_bgr, 0.7, 0)

    #     blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    #     blended_img = Image.fromarray(blended_rgb)
    #     blended_img.save(osp.join(args.save_dir, osp.basename(im_path)))

    # print(f"Results are saved in {args.save_dir}")

    for im_path in tqdm.tqdm(images):
        result = inference_model(model, im_path)
        sem_seg = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)

        assert (
            sem_seg.ndim == 2
        ), f"Expected sem_seg to be 2D, but got {sem_seg.ndim}D array"

        img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]

        mask_resized = cv2.resize(sem_seg, (w, h), interpolation=cv2.INTER_NEAREST)

        assert (
            mask_resized.ndim == 2
        ), f"Expected sem_seg to be 2D, but got {sem_seg.ndim}D array"

        base = osp.splitext(osp.basename(im_path))[0]
        out_path = osp.join(args.save_dir, f"{base}.png")
        cv2.imwrite(out_path, mask_resized)

    print(f"Results are saved in {args.save_dir}")


if __name__ == "__main__":
    main()
