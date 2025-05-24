from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from torch import Tensor
from typing import List, Tuple
import torch
import torch.nn as nn
from mmseg.models.builder import MODELS
import torch.nn.functional as F
from .third_party import clip
import json

from .third_party import imagenet_templates
from einops import rearrange, repeat
import torch.nn.init as init

from mmengine.structures import InstanceData, PixelData
from typing import Dict, List, Optional, Tuple, Union

from mmdet.models.utils import multi_apply
from mmdet.utils import (
    ConfigType,
    InstanceList,
    OptConfigType,
    OptMultiConfig,
    reduce_mean,
)
from mmseg.structures.seg_data_sample import SegDataSample


@MODELS.register_module()
class VireoMask2FormerHead(Mask2FormerHead):
    def __init__(
        self,
        class_json,
        replace_query_feat=False,
        prompt_depth=0,
        prompt_length=0,
        ignore_index=255,
        prompt_ensemble_type="imagenet_select",
        **kwargs,
    ):
        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        num_queries = kwargs["num_queries"]
        del self.query_embed
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)

        self.DQCM = DynamicQueryCorrectionModule(
            in_channels=1024, embedding_dim=1024, index=4
        )
        self.ignore_index = ignore_index

        self.tokenizer = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_depth = prompt_depth
        self.prompt_length = prompt_length

        clip_model, clip_preprocess = clip.load(
            "ViT-L/14@336px",
            device=device,
            jit=False,
            prompt_depth=prompt_depth,
            prompt_length=prompt_length,
        )

        self.prompt_ensemble_type = prompt_ensemble_type

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = [
                "A photo of a {} in the scene",
            ]
        else:
            raise NotImplementedError

        with open(class_json, "r") as f_in:
            self.class_texts = json.load(f_in)

        self.text_features = (
            self.class_embeddings(self.class_texts, prompt_templates, clip_model)
            .permute(1, 0, 2)
            .float()
        ).to(device)

        self.cls_embed = nn.Linear(feat_channels, 768)

        # self.mask_embed = nn.Sequential(
        #     nn.Linear(768, 768), nn.ReLU(inplace=True),
        #     nn.Linear(768, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, feat_channels))

        self.feat_conv = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(feat_channels, 1, kernel_size=1)

        self.prior = PriorGenerator(
            in_channels=1024,
            class_embed_dim=768,
            prompt_number=len(prompt_templates),
            num_queries=num_queries,
            transformer_embed_dim=256,
        )

        # self.temperature = nn.Parameter(torch.tensor(1.0))

        init.xavier_uniform_(self.cls_embed.weight)
        if self.cls_embed.bias is not None:
            init.constant_(self.cls_embed.bias, 0)

    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        x, query_embed = x

        text_features = self.text_features
        text_features = text_features.repeat(x[0].shape[0], 1, 1, 1)

        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)

        x[0] = self.DQCM(x) + x[0]
        mask_prior, query_prior = self.prior(x[0], text_features)

        query_embed = query_embed + query_prior

        text_features = F.normalize(text_features, dim=-1)  # B T P C2
        text_features = text_features.mean(dim=2)  # B T C2
        text_features = F.normalize(text_features, dim=-1)

        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        # query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:], text_features
        )

        # cls_pred, mask_pred, attn_mask = self._forward_head(
        #     query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        # )

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
                text_features,
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list, mask_prior

    def _forward_head(
        self,
        decoder_out: Tensor,
        mask_feature: Tensor,
        attn_mask_target_size: Tuple[int, int],
        text_feature: Tensor,
    ) -> Tuple[Tensor]:
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (batch_size, num_queries, c1)

        cls_embed = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c2)

        cls_pred = torch.einsum("bqc,btc->bqt", cls_embed, text_feature)
        # shape (batch_size, num_queries, num_classes)

        mask_embed = self.mask_embed(decoder_out)

        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)

        attn_mask = F.interpolate(
            mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False
        )
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = (
            attn_mask.flatten(2)
            .unsqueeze(1)
            .repeat((1, self.num_heads, 1, 1))
            .flatten(0, 1)
        )
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def prior_loss(
        self,
        mask_prior: torch.Tensor,  # (B, T, Hp, Wp)
        target: torch.Tensor,  # (B, H,  W )
        num_classes: int,
        ignore_index: int = 255,
    ):

        B, T, Hp, Wp = mask_prior.shape

        target_ds = (
            F.interpolate(
                target.unsqueeze(1).float(), size=(Hp, Wp), mode="nearest"  # (B,1,H,W)
            )
            .squeeze(1)
            .long()
        )  # (B, Hp, Wp)

        logits = mask_prior.permute(0, 2, 3, 1).reshape(-1, num_classes)
        flat_tgt = target_ds.view(-1)  # (B*Hp*Wp,)
        valid_idx = flat_tgt != ignore_index

        tgt_onehot = F.one_hot(flat_tgt[valid_idx], num_classes=num_classes).float()

        loss = F.binary_cross_entropy_with_logits(
            logits[valid_idx], tgt_onehot, reduction="mean"
        )
        return loss

    def loss_by_feat(
        self,
        all_cls_scores: Tensor,
        all_mask_preds: Tensor,
        mask_prior: Tensor,
        batch_gt_instances: List[InstanceData],
        target: Tensor,
        batch_img_metas: List[dict],
    ) -> Dict[str, Tensor]:
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [batch_gt_instances for _ in range(num_dec_layers)]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_mask, losses_dice = multi_apply(
            self._loss_by_feat_single,
            all_cls_scores,
            all_mask_preds,
            batch_gt_instances_list,
            img_metas_list,
        )

        losses_prior = self.prior_loss(
            mask_prior, target, self.num_classes, self.ignore_index
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_mask"] = losses_mask[-1]
        loss_dict["loss_dice"] = losses_dice[-1]
        loss_dict["losses_prior"] = losses_prior * 0.1
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
            losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_mask"] = loss_mask_i
            loss_dict[f"d{num_dec_layer}.loss_dice"] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(
        self,
        cls_scores: Tensor,
        mask_preds: Tensor,
        batch_gt_instances: List[InstanceData],
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            avg_factor,
        ) = self.get_targets(
            cls_scores_list, mask_preds_list, batch_gt_instances, batch_img_metas
        )
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights)

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        # upsample to shape of target
        # shape (num_total_gts, h, w)
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(1), target_shape, mode="bilinear", align_corners=False
        ).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(mask_preds, mask_targets, avg_factor=num_total_masks)

        # mask loss
        # FocalLoss support input of shape (n, num_class)
        h, w = mask_preds.shape[-2:]
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
        mask_preds = mask_preds.reshape(-1, 1)
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
        mask_targets = mask_targets.reshape(-1)
        # target is (1 - mask_targets) !!!
        loss_mask = self.loss_mask(
            mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w
        )

        return loss_cls, loss_mask, loss_dice

    def loss(
        self, x: Tuple[Tensor], batch_data_samples: SampleList, train_cfg: ConfigType
    ) -> dict:
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples
        )

        gt_list = [ds.gt_sem_seg.data for ds in batch_data_samples]
        target = torch.stack(gt_list, dim=0)  # shape -> [B, H, W] 或 [B, 1, H, W]
        target = target.long()

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # -> [B, H, W]

        # forward
        all_cls_scores, all_mask_preds, mask_prior = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(
            all_cls_scores,
            all_mask_preds,
            mask_prior,
            batch_gt_instances,
            target,
            batch_img_metas,
        )

        return losses

    def predict(
        self, x: Tuple[Tensor], batch_img_metas: List[dict], test_cfg: ConfigType
    ) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds, _ = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if "pad_shape" in batch_img_metas[0]:
            size = batch_img_metas[0]["pad_shape"]
        else:
            size = batch_img_metas[0]["img_shape"]
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode="bilinear", align_corners=False
        )
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum("bqc, bqhw->bchw", cls_score, mask_pred)
        return seg_logits

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ", " in classname:
                classname_splits = classname.split(", ")
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [
                    template.format(classname) for template in templates
                ]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(
                    len(templates), -1, class_embeddings.shape[-1]
                ).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights


class AdaptiveAttentionGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AdaptiveAttentionGate, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        spatial_weights = torch.sigmoid(self.spatial_attention(x))
        x = x * spatial_weights

        return x


class DynamicQueryCorrectionModule(nn.Module):
    def __init__(self, in_channels=1024, embedding_dim=1024, index=4, **kwargs):
        super(DynamicQueryCorrectionModule, self).__init__()
        self.in_channels = in_channels
        self.indexes = index

        self.attention_gate = AdaptiveAttentionGate(in_channels=embedding_dim)

        self.linear_fuse = nn.Conv2d(
            embedding_dim * self.indexes, embedding_dim, kernel_size=1
        )

    def forward(self, x_all):
        x_list = []
        _, _, h_last, w_last = x_all[0].shape

        for ind in range(len(x_all)):
            x = x_all[ind]

            _x = F.interpolate(
                x, size=(h_last, w_last), mode="bilinear", align_corners=False
            )

            _x = self.attention_gate(_x)

            x_list.append(_x)

        x_list = torch.cat(x_list, dim=1)
        x = self.linear_fuse(x_list)

        return x


class PriorGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        class_embed_dim: int,
        prompt_number: int,
        transformer_embed_dim: int,
        num_queries: int,
    ):
        super().__init__()
        self.feat_proj = nn.Conv2d(in_channels, class_embed_dim, kernel_size=1)
        self.coarse_embed = nn.Conv2d(prompt_number, 1, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(in_channels, transformer_embed_dim)
        self.query_proto = nn.Parameter(torch.randn(num_queries, transformer_embed_dim))

    def forward(
        self,
        mask_features: torch.Tensor,  # (B, C_feat, H, W)
        text_features: torch.Tensor,  # (B, T+1, prompt, C_text)
    ):
        B, C_feat, H, W = mask_features.shape
        text = text_features[:, :-1, :, :]  # (B, T, prompt, C_text)
        _, T, P, C_text = text.shape

        feat_aligned = self.feat_proj(mask_features)  # (B, C_text, H, W)
        coarse = torch.einsum("bchw,btpc->btphw", feat_aligned, text)  # (B, T, P, H, W)
        coarse = rearrange(coarse, "b t p h w -> (b t) p h w")
        coarse = self.coarse_embed(coarse)  # (B*T, 1, H, W)
        coarse = rearrange(coarse, "(b t) 1 h w -> b t h w", b=B)  # (B, T, H, W)
        prob = coarse.sigmoid()  # (B, T, H, W)

        prob_flat = prob.view(B, T, -1)  # (B, T, N)
        weights = torch.softmax(prob_flat, dim=-1)
        feats_flat = mask_features.view(B, C_feat, -1)  # (B, C_feat, N)
        class_feats = torch.einsum(
            "b t n, b c n -> b t c", weights, feats_flat
        )  # (B, T, C_feat)

        class_embeds = self.query_proj(class_feats)  # (B, T, E)

        sim = torch.einsum("b t e, q e -> b t q", class_embeds, self.query_proto)
        attn = torch.softmax(sim, dim=1)  # normalize over T → (B, T, Q)
        query_anchors = torch.einsum("b t q, b t e -> b q e", attn, class_embeds)

        return coarse, query_anchors
