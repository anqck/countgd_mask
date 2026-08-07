# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from groundingdino.util.misc import NestedTensor, is_main_process

from .position_encoding import (
    PositionEmbeddingLearned,
    PositionEmbeddingSineHW,
    build_position_encoding,
)
from .swin_transformer import SwinTransformer, build_swin_transformer


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly

        w = self.weight.reshape(1, -1, 1, 1)  # ty: ignore[call-non-callable]
        b = self.bias.reshape(1, -1, 1, 1)  # ty: ignore[call-non-callable]
        rv = self.running_var.reshape(1, -1, 1, 1)  # ty: ignore[call-non-callable]
        rm = self.running_mean.reshape(1, -1, 1, 1)  # ty: ignore[call-non-callable]
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int | list[int],
        return_interm_indices: list,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {f"layer{5 - len(return_interm_indices) + idx}": f"{layer_index}"}
            )

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

        if isinstance(self.num_channels, int):
            self.embed_dim = self.num_channels
        else:
            self.embed_dim = self.num_channels[0]

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[dict[str, NestedTensor], torch.Tensor]:
        xs = self.body(tensor_list.tensors)
        out: dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out, xs[0]


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=FrozenBatchNorm2d,
    ):
        if name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(),
                norm_layer=batch_norm,
            )
        else:
            raise NotImplementedError(
                f"Possible `name`s are "
                f"{['resnet18', 'resnet34', 'resnet50', 'resnet101']}"
            )
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ("resnet18", "resnet34"), (
            "Only resnet50 and resnet101 are available."
        )
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(
        self,
        backbone: Backbone | SwinTransformer,
        position_embedding: PositionEmbeddingLearned | PositionEmbeddingSineHW,
    ):
        super().__init__(backbone, position_embedding)

    def forward(
        self, tensor_list: NestedTensor
    ) -> tuple[list[NestedTensor], torch.Tensor, list[torch.Tensor]]:  # ty: ignore[invalid-method-override]
        backbone: Backbone | SwinTransformer = self[0]
        pos_emb: PositionEmbeddingLearned | PositionEmbeddingSineHW = self[1]
        xs: dict[int | str, NestedTensor]
        layer0: torch.Tensor
        xs, layer0 = backbone(tensor_list)
        out: list[NestedTensor] = []
        pos: list[torch.Tensor] = []
        for x in xs.values():
            out.append(x)
            # position encoding
            pos.append(pos_emb(x).to(x.tensors.dtype))

        return out, layer0, pos


def build_backbone(args) -> Joiner:
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = True
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=FrozenBatchNorm2d,
        )
        bb_num_channels = backbone.num_channels
    elif args.backbone in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )

        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    else:
        raise NotImplementedError(f"Unknown backbone {args.backbone}")

    assert len(bb_num_channels) == len(return_interm_indices), (
        f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"
    )
    assert isinstance(bb_num_channels, list), (
        f"bb_num_channels is expected to be a list but got {type(bb_num_channels)}"
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels  # ty: ignore[invalid-assignment]
    return model
