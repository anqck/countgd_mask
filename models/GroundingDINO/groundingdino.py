# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
from collections.abc import Sequence
import copy
from typing import Any, Literal
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align
from torchvision.ops.boxes import nms
from transformers.tokenization_utils_base import BatchEncoding

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    get_world_size,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

from ..registry import MODULE_BUILD_FUNCS
from .backbone import Joiner, build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .matcher import build_matcher
from .positional_encoding_loca import PositionalEncodingsFixed
from .transformer import Transformer, build_transformer, MaskHead, build_maskhead
from .transformer_loca import TransformerEncoder
from .utils import MLP, ContrastiveEmbed

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_target_points(
    image,
    target,
    save_path=None,
    show=True,
    point_size=40,
    box_alpha=0.7,
    title=None,
):
    """
    Visualize target boxes/points on the original image.

    Args:
        image:
            - PIL.Image
            - numpy array [H, W, 3]
            - numpy array [H, W]
            - image path (str)

        target:
            target dict containing:
                target["boxes"] : [N, 4]
                    format [cx, cy, w, h]
                    normalized to [0, 1]

            Here cx, cy are treated as point annotations.

        save_path:
            Where to save visualization.

        show:
            Whether to display with matplotlib.

        point_size:
            Size of point marker.

        box_alpha:
            Transparency of bbox.

        title:
            Optional figure title.
    """

    # ============================================================
    # 1. Load image
    # ============================================================

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        image = np.asarray(image)

    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = np.asarray(image)

    elif torch.is_tensor(image):
        image = image.detach().cpu()

        # CHW -> HWC
        if image.ndim == 3:
            if image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)

        image = image.numpy()

    elif isinstance(image, np.ndarray):
        pass

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ============================================================
    # 2. Normalize image for visualization
    # ============================================================

    if image.dtype != np.uint8:
        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = ((image - image_min) / (image_max - image_min) * 255).astype(
                np.uint8
            )

        else:
            image = np.zeros_like(image, dtype=np.uint8)

    # Grayscale -> RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    H, W = image.shape[:2]

    # ============================================================
    # 3. Get target boxes
    # ============================================================

    boxes = target["boxes"]

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu()

    boxes = boxes.float().numpy()

    if boxes.ndim != 2 or boxes.shape[1] < 2:
        raise ValueError(f"Expected boxes [N,4], got {boxes.shape}")

    # ============================================================
    # 4. Create figure
    # ============================================================

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.imshow(image)

    # ============================================================
    # 5. Draw every target
    # ============================================================

    for i, box in enumerate(boxes):
        cx_norm = float(box[0])
        cy_norm = float(box[1])

        # --------------------------------------------------------
        # normalized [0,1] -> original image pixel
        # --------------------------------------------------------

        cx = cx_norm * W
        cy = cy_norm * H

        # --------------------------------------------------------
        # Draw point
        # --------------------------------------------------------

        ax.scatter(
            cx,
            cy,
            s=point_size,
            marker="x",
            linewidths=2,
        )

        # --------------------------------------------------------
        # Instance index
        # --------------------------------------------------------

        ax.text(
            cx + 4,
            cy - 4,
            str(i),
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                pad=1,
            ),
        )

        # --------------------------------------------------------
        # Draw bbox if width/height are meaningful
        #
        # Your task currently treats boxes as points.
        # Therefore if w/h == 0 or extremely small,
        # we DON'T draw a bbox.
        # --------------------------------------------------------

        if boxes.shape[1] >= 4:
            w_norm = float(box[2])
            h_norm = float(box[3])

            # If actual bbox exists
            if w_norm > 1e-6 and h_norm > 1e-6:
                bw = w_norm * W
                bh = h_norm * H

                x1 = cx - bw / 2
                y1 = cy - bh / 2

                rect = patches.Rectangle(
                    (
                        x1,
                        y1,
                    ),
                    bw,
                    bh,
                    fill=False,
                    linewidth=1.5,
                    alpha=box_alpha,
                )

                ax.add_patch(rect)

    # ============================================================
    # 6. Figure settings
    # ============================================================

    ax.set_xlim(
        0,
        W,
    )

    ax.set_ylim(
        H,
        0,
    )

    ax.set_xlabel("x (pixel)")

    ax.set_ylabel("y (pixel)")

    if title is None:
        title = f"Target points (N={len(boxes)})"

    ax.set_title(title)

    ax.grid(False)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(
            os.path.dirname(save_path) or ".",
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()

    else:
        plt.close(fig)

    return fig, ax


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone: Joiner,
        transformer: Transformer,
        mask_head: MaskHead,
        num_queries: int,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type: Literal["no", "standard"] = "no",
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # visual exemplar cropping
        self.feature_map_proj = nn.Conv2d(
            (256 + 512 + 1024), self.hidden_dim, kernel_size=1
        )

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(
            self.bert.config.hidden_size, self.hidden_dim, bias=True
        )
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )

        # prepare input projection layers
        # num_feature_levels = 4
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.num_channels[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            self.hidden_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:  # dead branch
            assert (
                two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            backbone.num_channels[-1], self.hidden_dim, kernel_size=1
                        ),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        nn.init.constant_(
            _bbox_embed.layers[-1].weight.data, 0
        )  # ty: ignore[invalid-argument-type]
        nn.init.constant_(
            _bbox_embed.layers[-1].bias.data, 0
        )  # ty: ignore[invalid-argument-type]

        # dec_pred_bbox_embed_share=True
        if dec_pred_bbox_embed_share:
            self.bbox_embed = nn.ModuleList(
                [_bbox_embed] * transformer.num_decoder_layers
            )
        else:  # dead branch
            self.bbox_embed = [
                copy.deepcopy(_bbox_embed)
            ] * transformer.num_decoder_layers
        self.class_embed = nn.ModuleList(
            [_class_embed] * transformer.num_decoder_layers
        )
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            "no",
            "standard",
        ], f"unknown param {two_stage_type} of two_stage_type"
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self.mask_head = mask_head
        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def add_exemplar_tokens(
        self,
        tokenized: dict[str, Any],
        text_dict: dict[str, Any],
        exemplar_tokens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Splice visual exemplar token embeddings into the text token sequence.

        For each sample in the batch, embeds the given exemplar tokens directly
        before the caption phrase corresponding to the sample's label (category
        index). Placeholder token id 1008 marks the exemplar positions in
        ``input_ids`` while the actual exemplar features are inserted at the same
        position in ``encoded_text``, keeping the token and embedding sequences
        aligned. Attention masks and position ids are regenerated from the
        modified token sequences so exemplars participate in text self-attention.

        Args:
            tokenized: BERT BatchEncoding containing "input_ids".
            text_dict: dict with "encoded_text", "text_token_mask", "position_ids"
                and "text_self_attention_masks" from the BERT encoder.
            exemplar_tokens: [batch_size, num_exemplars, hidden_dim] visual
                exemplar features (e.g. from RoIAlign of combined features).
            labels: per-sample category index whose phrase receives the exemplar
                insertion, shape [batch_size, 1].

        Returns:
            Updated text_dict with exemplar-spliced "encoded_text", an
            all-True "text_token_mask", and regenerated "position_ids" and
            "text_self_attention_masks".
        """
        input_ids: torch.Tensor = tokenized["input_ids"]

        device = input_ids.device
        new_input_ids = []
        encoded_text = text_dict["encoded_text"]
        new_encoded_text = []
        _text_token_mask = text_dict["text_token_mask"]
        new_text_token_mask = []
        position_ids = text_dict["position_ids"]
        _text_self_attention_masks = text_dict["text_self_attention_masks"]

        for sample_ind in range(len(labels)):
            label = labels[sample_ind][0]
            exemplars = exemplar_tokens[sample_ind]
            label_count = -1
            assert len(input_ids[sample_ind]) == len(position_ids[sample_ind])
            for token_ind in range(len(input_ids[sample_ind])):
                input_id = input_ids[sample_ind][token_ind]
                if (input_id not in self.specical_tokens) and (
                    token_ind == 0
                    or (input_ids[sample_ind][token_ind - 1] in self.specical_tokens)
                ):
                    label_count += 1
                if label_count == label:
                    # Get the index where to insert the exemplar tokens.
                    ind_to_insert_exemplar = token_ind
                    while (
                        input_ids[sample_ind][ind_to_insert_exemplar]
                        not in self.specical_tokens
                    ):
                        ind_to_insert_exemplar += 1
                    break

            # * token indicates exemplar.
            new_input_ids.append(
                torch.cat(
                    [
                        input_ids[sample_ind][:ind_to_insert_exemplar],
                        torch.tensor([1008] * exemplars.shape[0]).to(device),
                        input_ids[sample_ind][ind_to_insert_exemplar:],
                    ]
                )
            )
            new_encoded_text.append(
                torch.cat(
                    [
                        encoded_text[sample_ind][:ind_to_insert_exemplar, :],
                        exemplars,
                        encoded_text[sample_ind][ind_to_insert_exemplar:, :],
                    ]
                )
            )
            new_text_token_mask.append(
                torch.full((len(new_input_ids[sample_ind]),), True).to(device)
            )

        tokenized["input_ids"] = torch.stack(new_input_ids)

        _text_self_attention_masks, position_ids, _ = (
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.specical_tokens, None
            )
        )

        return {
            "encoded_text": torch.stack(new_encoded_text),
            "text_token_mask": torch.stack(new_text_token_mask),
            "position_ids": position_ids,
            "text_self_attention_masks": _text_self_attention_masks,
        }

    def combine_features(self, features: list[NestedTensor]) -> torch.Tensor:
        """
        Fuse multi-scale backbone features into a single unified feature map.

        Interpolates each input feature map to the spatial resolution of the first
        one (bilinear, align_corners=True), concatenates them along the channel
        dimension, and projects the result through a 1x1 convolution
        (``feature_map_proj``) to reduce the total channel count (256 + 512 + 1024)
        down to ``hidden_dim``.

        Args:
            features (NestedTensor): multi-scale feature maps from the backbone,
                one NestedTensor per level.

        Returns:
            Fused feature map of shape [batch_size, hidden_dim, h, w], where
            (h, w) are the spatial dimensions of the first input feature map.
            Used downstream to extract visual exemplar tokens via RoIAlign.
        """
        _bs, _c, h, w = (
            features[0].decompose()[0].shape[-4],
            features[0].decompose()[0].shape[-3],
            features[0].decompose()[0].shape[-2],
            features[0].decompose()[0].shape[-1],
        )

        x: torch.Tensor = torch.cat(
            [
                F.interpolate(
                    feat.decompose()[0],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                for feat in features
            ],
            dim=1,
        )

        x = self.feature_map_proj(x)

        return x

    def forward(
        self,
        samples: NestedTensor,
        exemplars: list,
        labels,
        targets: list | None = None,
        **kw,
    ):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        # encoder texts
        # Tokenizers return BatchEncoding inherits from UserDict
        # Functionally acts like a dictionary
        # .to sends internal data structures into specified device, returned is still
        # a BatchEncoding
        tokenized: BatchEncoding = self.tokenizer(
            captions, padding="longest", return_tensors="pt"
        ).to(samples.device)
        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            _cat_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, : self.max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, : self.max_text_len
            ]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(
            bert_output["last_hidden_state"]
        )  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(
                samples
            )  # ty: ignore[invalid-argument-type]

        features: list[NestedTensor]
        poss: list[torch.Tensor]
        features, layer0, poss = self.backbone(samples)
        combined_features = self.combine_features(features)

        # Get visual exemplar tokens.
        bs = len(exemplars)
        num_exemplars = exemplars[0].shape[0]
        if num_exemplars > 0:
            exemplar_tokens = (
                roi_align(
                    combined_features,
                    boxes=exemplars,
                    output_size=(1, 1),
                    spatial_scale=(1 / 8),
                    aligned=True,
                )
                .squeeze(-1)
                .squeeze(-1)
                .reshape(bs, num_exemplars, -1)
            )
        else:
            exemplar_tokens = None

        if exemplar_tokens is not None:
            text_dict = self.add_exemplar_tokens(
                tokenized, text_dict, exemplar_tokens, labels
            )

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        # pred_masks_per_dec_layer: n_dec, bs, nq, h, w
        (
            hs,
            reference,
            hs_enc,
            ref_enc,
            init_box_proposal,
            memory,
            spatial_shapes,
            tgt_undetach,
        ) = self.transformer(
            srcs,
            masks,
            input_query_bbox,
            poss,
            input_query_label,
            attn_mask,
            text_dict,
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        _mask_features, pred_masks_per_dec_layer, interm_masks = self.mask_head(
            hs,
            memory,
            spatial_shapes,
            backbone_layer_0=layer0,
            tgt_undetach=tgt_undetach,
            outputs_coord=outputs_coord_list,
        )

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
        }

        if pred_masks_per_dec_layer:
            out["pred_masks"] = pred_masks_per_dec_layer[-1]

        # Used to calculate losses
        bs, len_td = text_dict["text_token_mask"].shape
        out["text_mask"] = torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(
            samples.device
        )
        for b in range(bs):
            for j in range(len_td):
                if text_dict["text_token_mask"][b][j] == True:
                    out["text_mask"][b][j] = True

        # for intermediate outputs
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord_list, pred_masks_per_dec_layer
            )
        out["token"] = one_hot_token
        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            if interm_masks is not None:
                out["interm_outputs"]["pred_masks"] = interm_masks
            # out["interm_outputs_for_matching_pre"] = {
            #     "pred_logits": interm_class,
            #     "pred_boxes": init_box_proposal,
            # }

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_masks:
            return [
                {
                    "pred_logits": a,
                    "pred_boxes": b,
                    "pred_masks": c,
                }
                for a, b, c in zip(
                    outputs_class[:-1], outputs_coord[:-1], outputs_masks[:-1]
                )
            ]
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_dict,
        focal_alpha,
        focal_gamma,
        losses,
        # From MaskDINO
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes[:, :2], target_boxes[:, :2], reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
            losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        new_targets = outputs["one_hot"].to(pred_logits.device)
        text_mask = outputs["text_mask"]

        assert new_targets.dim() == 3
        assert pred_logits.dim() == 3  # batch x from x to

        bs, n, _ = pred_logits.shape
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        if text_mask is not None:
            # ODVG: each sample has different mask
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(
                outputs["text_mask"].shape[0], -1, outputs["text_mask"].shape[1]
            )
            pred_logits = torch.masked_select(pred_logits, text_mask)
            new_targets = torch.masked_select(new_targets, text_mask)

        new_targets = new_targets.float()
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, new_targets, reduction="none"
        )
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss

        total_num_pos = 0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos, 1.0)
        loss = loss.sum() / num_pos_avg_per_gpu

        losses = {"loss_ce": loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.token_sigmoid_binary_focal_loss,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_masks(self, outputs, targets, indices, num_masks, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        # def generate_gt_density(
        #     pts: torch.Tensor,
        #     shape: torch.Tensor | Sequence[int],
        #     s_factor: float = 8.0,
        #     normalize: bool = False,
        # ) -> torch.Tensor:
        #     """
        #     Generate per-point continuous GT Gaussian density maps on GPU.

        #     Args:
        #         pts (torch.Tensor[float32]): [N, 2] normalized coordinates (x, y) in range [0, 1].
        #             Can be passed directly from bounding box centers `boxes[:, :2]`.
        #         shape (tuple[int, int] | Sequence[int]): The (H, W) spatial resolution of the sampled canvas.
        #         s_factor (float): Divisor used to derive Gaussian standard deviation (sigma)
        #             from the 1st nearest neighbor distance.
        #         normalize (bool): If True, normalizes each map such that the 2D continuous
        #             integral equals 1. If False, peak amplitude at point center is 1.

        #     Returns:
        #         torch.Tensor[float32]: [N, H, W] Gaussian density maps for each GT point.
        #     """
        #     H, W = int(shape[0]), int(shape[1])
        #     N = pts.shape[0]

        #     if N == 0:
        #         return torch.zeros((0, H, W), dtype=torch.float32, device=pts.device)

        #     # 1. Denormalize coordinates: x -> [0, W], y -> [0, H]
        #     scale = torch.tensor([W, H], dtype=torch.float32, device=pts.device)
        #     pts_px = (
        #         pts[:, :2] * scale
        #     )  # [N, 2] -> col 0: x (pixels), col 1: y (pixels)

        #     x_center = pts_px[:, 0:1]  # [N, 1]
        #     y_center = pts_px[:, 1:2]  # [N, 1]

        #     # 2. Compute adaptive bandwidth (sigma) via nearest neighbor distance
        #     if N == 1:
        #         # Fallback for single object: scale relative to image average dimension
        #         sigma = (float(H + W) / 2.0) / (4.0 * s_factor)
        #     else:
        #         dists = torch.cdist(pts_px, pts_px, p=2.0)
        #         dists.fill_diagonal_(torch.inf)
        #         knn_dists, _ = torch.topk(dists, k=1, largest=False, dim=-1)
        #         sigma = (knn_dists.mean() / s_factor).clamp(min=1e-4).item()

        #     inv_two_var = 1.0 / (2.0 * (sigma**2))

        #     # 3. 1D Coordinate grids along height (Y) and width (X)
        #     # [1, H]
        #     y_grid = torch.arange(H, dtype=torch.float32, device=pts.device).unsqueeze(
        #         0
        #     )
        #     # [1, W]
        #     x_grid = torch.arange(W, dtype=torch.float32, device=pts.device).unsqueeze(
        #         0
        #     )

        #     # 4. Separable 1D Gaussian evaluations: O(N * (H + W))
        #     gy = torch.exp(-((y_grid - y_center) ** 2) * inv_two_var)  # [N, H]
        #     gx = torch.exp(-((x_grid - x_center) ** 2) * inv_two_var)  # [N, W]

        #     # 5. Outer product broadcasting: [N, H, 1] * [N, 1, W] -> [N, H, W]
        #     density = gy.unsqueeze(-1) * gx.unsqueeze(-2)

        #     # 6. Integral normalization
        #     if normalize:
        #         density = density / (2.0 * math.pi * (sigma**2))

        #     return density

        def generate_gt_density(
            pts: torch.Tensor,
            shape: torch.Tensor | Sequence[int],
            s_factor: float = 8.0,
            normalize: bool = False,
        ) -> torch.Tensor:
            """
            Generate per-point Gaussian maps on GPU.

            Args:
                pts:
                    [N, 2], normalized (x, y) coordinates in [0, 1].

                shape:
                    (H, W) of the target canvas.

                s_factor:
                    sigma = nearest-neighbor-distance / s_factor.

                normalize:
                    False:
                        peak-normalized Gaussian, peak ~= 1.

                    True:
                        continuous-integral normalized Gaussian.

            Returns:
                [N, H, W]
            """

            H, W = int(shape[0]), int(shape[1])
            N = pts.shape[0]

            if N == 0:
                return torch.zeros(
                    (0, H, W),
                    dtype=torch.float32,
                    device=pts.device,
                )

            # ---------------------------------------------------------
            # 1. Normalized coordinates -> pixel coordinates
            # ---------------------------------------------------------
            scale = torch.tensor(
                [W, H],
                dtype=torch.float32,
                device=pts.device,
            )

            pts_px = pts[:, :2] * scale
            # [N, 2]

            x_center = pts_px[:, 0:1]
            y_center = pts_px[:, 1:2]
            # [N, 1]

            # ---------------------------------------------------------
            # 2. Adaptive sigma for EACH point
            # ---------------------------------------------------------
            if N == 1:
                sigma = torch.tensor(
                    (H + W) / 2.0 / (4.0 * s_factor),
                    dtype=torch.float32,
                    device=pts.device,
                )

                # [1, 1]
                sigma = sigma.reshape(1, 1)

            else:
                dists = torch.cdist(
                    pts_px,
                    pts_px,
                    p=2.0,
                )
                # [N, N]

                dists.fill_diagonal_(torch.inf)

                knn_dists = dists.min(dim=-1).values
                # [N]

                sigma = (knn_dists / s_factor).clamp(min=1e-4)

                # IMPORTANT:
                # [N] -> [N, 1]
                sigma = sigma.unsqueeze(1)

            # ---------------------------------------------------------
            # 3. Gaussian coefficient
            # ---------------------------------------------------------
            inv_two_var = 1.0 / (2.0 * sigma.pow(2))
            # [N, 1]

            # ---------------------------------------------------------
            # 4. Coordinate grids
            # ---------------------------------------------------------
            y_grid = torch.arange(
                H,
                dtype=torch.float32,
                device=pts.device,
            ).unsqueeze(0)
            # [1, H]

            x_grid = torch.arange(
                W,
                dtype=torch.float32,
                device=pts.device,
            ).unsqueeze(0)
            # [1, W]

            # ---------------------------------------------------------
            # 5. Separable Gaussian
            # ---------------------------------------------------------
            gy = torch.exp(-((y_grid - y_center).pow(2)) * inv_two_var)
            # [N, H]

            gx = torch.exp(-((x_grid - x_center).pow(2)) * inv_two_var)
            # [N, W]

            # ---------------------------------------------------------
            # 6. Outer product
            # ---------------------------------------------------------
            density = gy.unsqueeze(-1) * gx.unsqueeze(-2)
            # [N, H, W]

            # ---------------------------------------------------------
            # 7. Optional integral normalization
            # ---------------------------------------------------------
            if normalize:
                # density = density / (
                #     2.0
                #     * math.pi
                #     * sigma.pow(2)
                # )
                density_sum = density.sum(
                    dim=(-2, -1),
                    keepdim=True,
                )

                density = density / density_sum.clamp(min=1e-8)

            return density

        def visualize_output_and_save(
            input_, output, save_path="", figsize=(20, 12), dots=None
        ):
            """
            dots: Nx2 numpy array for the ground truth locations of the dot annotation
                if dots is None, this information is not available
            """
            # 1. Extract batch dimensions [C, H_pad, W_pad] and unpadded dimensions (H_orig, W_orig)
            # image = targets[0]["samples"]
            # pt = targets[0]["boxes"]

            # image = F.interpolate(
            #     image.unsqueeze(0),
            #     scale_factor=0.25,
            #     mode="bilinear",
            #     align_corners=False,
            # ).squeeze(0)
            # # 2. Convert sample image directly to numpy format [H_pad, W_pad, C]
            # if image.dtype != torch.uint8:
            #     mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(
            #         3, 1, 1
            #     )
            #     std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
            #     image_denorm = (image * std + mean).clamp(0, 1)
            #     image_np = (
            #         (image_denorm.detach().cpu() * 255)
            #         .to(torch.uint8)
            #         .permute(1, 2, 0)
            #         .numpy()
            #     )
            # else:
            #     image_np = image.detach().cpu().permute(1, 2, 0).numpy()
            # format_for_plotting(denormalize(input_))
            output = output
            dots = dots

            # get the total count
            pred_cnt = output.sum().item()
            img1 = input_
            # output = format_for_plotting(output)

            fig = plt.figure(figsize=figsize)

            # display the input image
            ax = fig.add_subplot(2, 2, 1)
            ax.set_axis_off()
            ax.imshow(img1)
            if dots is not None:
                ax.scatter(dots[:, 0], dots[:, 1], c="red", edgecolors="blue")
                # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
                ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
            else:
                ax.set_title("Input image")

            ax = fig.add_subplot(2, 2, 2)
            ax.set_axis_off()
            ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

            img2 = (
                0.2989 * img1[:, :, 0] + 0.5870 * img1[:, :, 1] + 0.1140 * img1[:, :, 2]
            )
            ax.imshow(img2, cmap="gray")
            ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)

            # # display the density map
            ax = fig.add_subplot(2, 2, 3)
            ax.set_axis_off()
            ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
            ax.imshow(output)
            # plt.colorbar()

            # ax = fig.add_subplot(2, 2, 4)
            ax.set_axis_off()
            ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
            ret_fig = ax.imshow(output)
            fig.colorbar(ret_fig, ax=ax)
            fig.savefig("./output.png", bbox_inches="tight")
            # fig.show()
            plt.close()

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Gather matched predictions: [bs, 900, H, W] -> [N_matched, H, W]
        src_masks = outputs["pred_masks"][src_idx]

        if src_masks.numel() == 0 or num_masks == 0:
            return {
                "loss_mask": src_masks.sum() * 0.0,
                "loss_dice": torch.tensor(0.0, device=outputs["pred_masks"].device),
            }

        H_pad, W_pad = outputs["pred_masks"].shape[-2:]

        padded_densities = []
        for t in targets:
            pt = t["boxes"]
            # if len(pt) == 0:
            #     continue

            boxes_norm = pt[:, 1:] if pt.shape[-1] == 5 else pt
            pts = boxes_norm[:, :2]

            H_tgt, W_tgt = int(t["size"][0]) // 4, int(t["size"][1]) // 4
            gt_density = generate_gt_density(
                pts=pts,
                shape=(H_tgt, W_tgt),
                s_factor=8.0,
                normalize=False,
            )

            pad_w = max(0, W_pad - W_tgt)
            pad_h = max(0, H_pad - H_tgt)
            if pad_w > 0 or pad_h > 0:
                gt_density = F.pad(gt_density, (0, pad_w, 0, pad_h))
            padded_densities.append(gt_density[:, :H_pad, :W_pad])

        # Flatten and gather matched ground-truth instances: [N_matched, H, W]
        target_densities_flat = torch.cat(padded_densities, dim=0).to(src_masks.device)
        batch_idx, tgt_inst_idx = tgt_idx
        offsets = torch.tensor(
            [0] + [d.shape[0] for d in padded_densities], device=src_masks.device
        ).cumsum(0)
        flat_tgt_idx = offsets[batch_idx.to(src_masks.device)] + tgt_inst_idx.to(
            src_masks.device
        )
        target_densities = target_densities_flat[flat_tgt_idx]

        # Valid region mask to avoid loss on padded margins
        B = len(targets)
        valid_mask_batch = torch.zeros(
            (B, H_pad, W_pad), dtype=torch.bool, device=src_masks.device
        )
        for b, t in enumerate(targets):
            valid_mask_batch[b, : int(t["size"][0]) // 4, : int(t["size"][1]) // 4] = (
                True
            )
        valid_mask = valid_mask_batch[batch_idx.to(src_masks.device)]

        # Compute masked L2 loss
        pred_probs = src_masks.sigmoid()
        diff_sq = (pred_probs - target_densities) ** 2

        num_valid_pixels = valid_mask.sum().clamp(min=1.0)
        loss_mask = (diff_sq * valid_mask).sum() / num_valid_pixels

        return {
            "loss_mask": loss_mask,
            "loss_dice": torch.tensor(0.0, device=src_masks.device),
        }

    def forward(self, outputs, targets, cat_list, caption, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        device = next(iter(outputs.values())).device
        one_hot = torch.zeros(
            outputs["pred_logits"].size(), dtype=torch.int64
        )  # torch.Size([bs, 900, 256])
        token = outputs["token"]

        # print(outputs.keys(),outputs["pred_masks"].shape)
        label_map_list = []
        indices = []
        for j in range(len(cat_list)):  # bs
            label_map = []
            for i in range(len(cat_list[j])):
                label_id = torch.tensor([i])
                per_label = create_positive_map_exemplar(
                    token["input_ids"][j], label_id, [101, 102, 1012, 1029]
                )
                label_map.append(per_label)
            label_map = torch.stack(label_map, dim=0).squeeze(1)

            label_map_list.append(label_map)

        for j in range(len(cat_list)):  # bs
            for_match = {
                "pred_logits": outputs["pred_logits"][j].unsqueeze(0),
                "pred_boxes": outputs["pred_boxes"][j].unsqueeze(0),
            }
            if "pred_masks" in outputs:
                for_match["pred_masks"] = outputs["pred_masks"][j].unsqueeze(0)

            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)
        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)

        # import pdb; pdb.set_trace()
        tgt_ids = [v["labels"].cpu() for v in targets]
        # len(tgt_ids) == bs
        for i in range(len(indices)):
            tgt_ids[i] = tgt_ids[i][indices[i][1]]
            one_hot[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
        outputs["one_hot"] = one_hot
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = []
                for j in range(len(cat_list)):  # bs
                    aux_output_single = {
                        "pred_logits": aux_outputs["pred_logits"][j].unsqueeze(0),
                        "pred_boxes": aux_outputs["pred_boxes"][j].unsqueeze(0),
                    }
                    # if "pred_masks" in outputs:
                    #     aux_output_single["pred_masks"] = aux_outputs["pred_masks"][j].unsqueeze(0)
                    inds = self.matcher(
                        aux_output_single, [targets[j]], label_map_list[j]
                    )
                    indices.extend(inds)
                one_hot_aux = torch.zeros(
                    outputs["pred_logits"].size(), dtype=torch.int64
                )
                tgt_ids = [v["labels"].cpu() for v in targets]
                for i in range(len(indices)):
                    tgt_ids[i] = tgt_ids[i][indices[i][1]]
                    one_hot_aux[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(
                        torch.long
                    )
                aux_outputs["one_hot"] = one_hot_aux
                aux_outputs["text_mask"] = outputs["text_mask"]
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = []
            for j in range(len(cat_list)):  # bs
                interm_output_single = {
                    "pred_logits": interm_outputs["pred_logits"][j].unsqueeze(0),
                    "pred_boxes": interm_outputs["pred_boxes"][j].unsqueeze(0),
                }
                # if "pred_masks" in outputs:
                #     interm_output_single["pred_masks"] = interm_outputs["pred_masks"][j].unsqueeze(0)
                inds = self.matcher(
                    interm_output_single, [targets[j]], label_map_list[j]
                )
                indices.extend(inds)
            one_hot_aux = torch.zeros(outputs["pred_logits"].size(), dtype=torch.int64)
            tgt_ids = [v["labels"].cpu() for v in targets]
            for i in range(len(indices)):
                tgt_ids[i] = tgt_ids[i][indices[i][1]]
                one_hot_aux[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(
                    torch.long
                )
            interm_outputs["one_hot"] = one_hot_aux
            interm_outputs["text_mask"] = outputs["text_mask"]
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs
                )
                l_dict = {f"{k}_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(
        self,
        num_select=100,
        text_encoder_type="text_encoder_type",
        nms_iou_threshold=-1,
        use_coco_eval=False,
        args=None,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        if args.use_coco_eval:
            from pycocotools.coco import COCO

            coco = COCO(args.coco_val_path)
            category_dict = coco.loadCats(coco.getCatIds())
            cat_list = [item["name"] for item in category_dict]
        else:
            cat_list = args.label_list
        caption = " . ".join(cat_list) + " ."
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        label_list = torch.arange(len(cat_list))
        pos_map = create_positive_map(tokenized, label_list, cat_list, caption)

        self.nms_iou_threshold = nms_iou_threshold
        self.positive_map = pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(prob.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode="trunc")
        labels = topk_indexes % prob.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]

            results = [
                {"scores": s[i], "labels": l[i], "boxes": b[i]}
                for s, l, b, i in zip(scores, labels, boxes, item_indices)
            ]
        else:
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]
        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(
    args,
) -> tuple[GroundingDINO, SetCriterion, dict[str, PostProcess]]:
    device = torch.device(args.device)

    generate_mask = getattr(args, "generate_mask", False)

    # prepare weight dict
    weight_dict = {
        "loss_ce": args.cls_loss_coef,  # 5.0
        "loss_bbox": args.bbox_loss_coef,  # 1.0
        "loss_giou": args.giou_loss_coef,  # 0.0
        "loss_mask": 0,
        "loss_dice": 0,
    }
    weight_dict["loss_mask"] = args.mask_loss_coef
    weight_dict["loss_dice"] = args.dice_loss_coef
    # if generate_mask:

    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        # {k}_{0..4} for k in weight_dict
        for i in range(args.dec_layers - 1):  # dec_layers = 6
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in clean_weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    # two-stage (encoder-query) loss weights; criterion emits {k}_interm.
    # Only engaged with the mask branch: without it, interm losses stay
    # computed-but-dropped, preserving the original counting training dynamics.
    if generate_mask and args.two_stage_type != "no":
        _coeff_weight_dict = {
            "loss_ce": args.cls_loss_coef,
            "loss_bbox": (
                1.0
                if getattr(args, "no_interm_box_loss", False)
                else args.bbox_loss_coef
            ),
            "loss_giou": (
                1.0
                if getattr(args, "no_interm_box_loss", False)
                else args.giou_loss_coef
            ),
        }
        _coeff_weight_dict["loss_mask"] = args.mask_loss_coef
        _coeff_weight_dict["loss_dice"] = args.dice_loss_coef
        interm_weight_dict = {
            k + "_interm": v * getattr(args, "interm_loss_coef", 1.0)
            for k, v in _coeff_weight_dict.items()
        }
        weight_dict.update(interm_weight_dict)

        # print(weight_dict)
        # assert 1 == 0
    # Built Backbone according to config:
    # Joiner(
    #     SwinTransformer(
    #         pretrain_img_size=384,
    #         out_indices=[1,2,3],
    #         dilation=False,
    #         use_checkpoint=True,
    #         embed_dim=192,
    #         depths=[2, 2, 18, 2],
    #         num_heads=[6, 12, 24, 48],
    #         window_size=12
    #     ),
    #     PositionEmbeddingSineHW(128, 20, 20, True)
    # )
    backbone = build_backbone(args)
    args.backbone_layer0_channels = backbone[0].embed_dim
    # Built Transformer according to config
    # Transformer(
    #     d_model=args.hidden_dim,                                 # 256
    #     dropout=args.dropout,                                    # 0.0
    #     nhead=args.nheads,                                       # 8
    #     dim_feedforward=args.dim_feedforward,                    # 2048
    #     num_encoder_layers=args.enc_layers,                      # 6
    #     num_decoder_layers=args.dec_layers,                      # 6
    #     normalize_before=args.pre_norm,                          # False
    #     num_queries=args.num_queries,                            # 900
    #     return_intermediate_dec=True,                            # True
    #     query_dim=args.query_dim,                                # 4
    #     activation=args.transformer_activation,                  # ReLU
    #     num_patterns=args.num_patterns,                          # 0
    #     num_feature_levels=args.num_feature_levels,              # 4
    #     enc_n_points=args.enc_n_points,                          # 4
    #     dec_n_points=args.dec_n_points,                          # 4
    #     learnable_tgt_init=True,                                 # True
    #     # two stage                                              # ------Separator: Two Stage------
    #     two_stage_type=args.two_stage_type,                      # standard # ['no', 'standard', 'early']
    #     embed_init_tgt=args.embed_init_tgt,                      # True
    #     use_text_enhancer=args.use_text_enhancer,                # True
    #     use_fusion_layer=args.use_fusion_layer,                  # True
    #     use_checkpoint=args.use_checkpoint,                      # Trye
    #     use_transformer_ckpt=args.use_transformer_ckpt,          # True
    #     use_text_cross_attention=args.use_text_cross_attention,  # True
    #     text_dropout=args.text_dropout,                          # 0
    #     fusion_dropout=args.fusion_dropout,                      # 0
    #     fusion_droppath=args.fusion_droppath,                    # 0.1
    # )
    #
    # Built matcher according to config:
    # HungarianMatcher(
    #     cost_class=args.set_cost_class,  # 5.0
    #     cost_bbox=args.set_cost_bbox,    # 1.0
    #     cost_giou=args.set_cost_giou,    # 0.0
    #     focal_alpha=args.focal_alpha,    # 0.25
    # )
    return (
        GroundingDINO(
            backbone,
            build_transformer(args),
            build_maskhead(args),
            num_queries=args.num_queries,  # 900
            aux_loss=args.aux_loss,  # True
            iter_update=True,  # True
            query_dim=4,  # 4
            num_feature_levels=args.num_feature_levels,  # 4
            nheads=args.nheads,  # 8
            dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,  # True
            two_stage_type=args.two_stage_type,  # standard
            two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,  # False
            two_stage_class_embed_share=args.two_stage_class_embed_share,  # 900
            num_patterns=args.num_patterns,  # 0
            dn_number=0,  # 0
            dn_box_noise_scale=args.dn_box_noise_scale,  # 1.0
            dn_label_noise_ratio=args.dn_label_noise_ratio,  # 0.5
            dn_labelbook_size=args.dn_labelbook_size,  # 91
            text_encoder_type=args.text_encoder_type,  # bert-base-uncased
            sub_sentence_present=args.sub_sentence_present,  # True
            max_text_len=args.max_text_len,  # 256
        ),
        SetCriterion(
            matcher=build_matcher(args),
            weight_dict=weight_dict,  # loss weight coeffs
            focal_alpha=args.focal_alpha,  # 0.25
            focal_gamma=args.focal_gamma,  # 2.0
            losses=(
                ["labels", "boxes"]
                if not generate_mask
                else ["labels", "boxes", "masks"]
            ),
            num_points=getattr(args, "mask_num_points", 112 * 112),
            oversample_ratio=getattr(args, "mask_oversample_ratio", 3.0),
            importance_sample_ratio=getattr(args, "mask_importance_sample_ratio", 0.75),
        ).to(device),
        {
            "bbox": PostProcess(
                num_select=args.num_select,  # 900
                text_encoder_type=args.text_encoder_type,  # bert-base-uncased
                nms_iou_threshold=args.nms_iou_threshold,  # -1
                args=args,
            )
        },
    )


def create_positive_map(tokenized, tokens_positive, cat_list, caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, label in enumerate(tokens_positive):
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map


def create_positive_map_exemplar(input_ids, label, special_tokens):
    tokens_positive = torch.zeros(256, dtype=torch.float)
    count = -1
    for token_ind in range(len(input_ids)):
        input_id = input_ids[token_ind]
        if (input_id not in special_tokens) and (
            token_ind == 0 or (input_ids[token_ind - 1] in special_tokens)
        ):
            count += 1
        if count == label:
            ind_to_insert_ones = token_ind

            while input_ids[ind_to_insert_ones] not in special_tokens:
                tokens_positive[ind_to_insert_ones] = 1
                ind_to_insert_ones += 1
            break
    return tokens_positive
