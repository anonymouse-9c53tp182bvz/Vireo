from mmseg.models.builder import BACKBONES, MODELS
from .vireo import Vireo
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train

import sys
from pathlib import Path

current_dir = Path(__file__).parent

depth_anything_dir = current_dir / "third_party" / "PromptDA"

sys.path.insert(0, str(depth_anything_dir))

from promptda.promptda import PromptDA

import torch
import torch.nn.functional as F

import types
from mmseg.structures import SegDataSample


def forward_features_extra(self, x, masks=None):
    if isinstance(x, list):
        return self.forward_features_list(x, masks)
    x = self.prepare_tokens_with_masks(x, masks)
    out = []
    for idx, blk in enumerate(self.blocks):
        x = blk(x)
        out.append(x)
    return out


@BACKBONES.register_module()
class VireoDinoVisionTransformerV2(DinoVisionTransformer):
    def __init__(
        self,
        Vireo_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Vireo: Vireo = MODELS.build(Vireo_config)

        DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # self.depth_anything = PromptDA(ckpt_path=f"checkpoints/promptda_vitl.ckpt").to(DEVICE).eval()
        self.depth_anything = PromptDA().to(DEVICE).eval()
        self.depth_anything.pretrained.forward_features_extra = types.MethodType(
            forward_features_extra, self.depth_anything.pretrained
        )

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        depth_features = self.depth_anything.pretrained.forward_features_extra(x)

        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)

        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.Vireo.forward(
                x,
                depth_features[idx],
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.Vireo.return_auto(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["vireo"])
        set_train(self, ["vireo"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "vireo" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
