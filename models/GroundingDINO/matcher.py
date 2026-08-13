# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def point_sample(input, point_coords, **kwargs) -> torch.Tensor:
    # Input: input: [N, C, H, W]
    # Input: point_coords: [N, P, 2]
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        # Input: point_coords.unsqueeze(2)
        # [N, P, 2] -> [N, P, 1, 2]
        point_coords = point_coords.unsqueeze(2)

    # Input: 2.0 * point_coords - 1.0
    # [N, P, 1, 2]
    # F.grid_sample output:
    # [N, C, P, 1]
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)

    if add_dim:
        # Input: output.squeeze(3)
        # [N, C, P, 1] -> [N, C, P]
        output = output.squeeze(3)

    return output


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Input: inputs: [nq, P]
    # Input: targets: [T, P]
    inputs = inputs.sigmoid()  # [nq, P]
    inputs = inputs.flatten(1)  # [nq, P]

    # Input: einsum("nc,mc->nm", inputs, targets)
    # [nq, P] x [T, P] -> [nq, T]
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)

    # Input: inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    # [nq, 1] + [1, T] -> [nq, T]
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]

    loss = 1 - (numerator + 1) / (denominator + 1)  # [nq, T]
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Input: inputs: [nq, P]
    # Input: targets: [T, P]
    hw = inputs.shape[1]  # P

    # Input: binary_cross_entropy_with_logits(...)
    # pos: [nq, P]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    # neg: [nq, P]
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    # Input: einsum("nc,mc->nm", pos, targets)
    # [nq, P] x [T, P] -> [nq, T]
    # Input: einsum("nc,mc->nm", neg, 1 - targets)
    # [nq, P] x [T, P] -> [nq, T]
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    # [nq, T]
    return loss / hw


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)
batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        generate_mask: bool = False,
        focal_alpha=0.25,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        self.generate_mask = generate_mask
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets, label_map):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten to compute cost matrices in a batch
        out_prob: torch.Tensor = (
            outputs["pred_logits"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concat target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        sizes = [len(v["boxes"]) for v in targets]
        total_targets = sum(sizes)

        # Compute classification cost
        alpha = self.focal_alpha
        gamma = 2.0

        new_label_map = label_map[tgt_ids.cpu()]

        neg_cost_class: torch.Tensor = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class: torch.Tensor = (
            alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        )
        new_label_map: torch.Tensor = new_label_map.to(pos_cost_class.device)
        cost_bbox = torch.cdist(out_bbox[:, :2], tgt_bbox[:, :2], p=1)

        cost_class = []
        for idx_map in new_label_map:
            idx_map = idx_map / idx_map.sum()
            cost_class.append(pos_cost_class @ idx_map - neg_cost_class @ idx_map)
        if cost_class:
            cost_class = torch.stack(cost_class, dim=0).T
        else:
            cost_class = torch.zeros_like(cost_bbox)

        # Compute GIoU cost between boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Pre-allocate cost_mask and cost_dice with shape [bs, num_queries, total_targets]
        cost_mask = torch.zeros(
            (bs, num_queries, total_targets), device=out_bbox.device
        )
        cost_dice = torch.zeros(
            (bs, num_queries, total_targets), device=out_bbox.device
        )

        if self.generate_mask and total_targets > 0:
            tgt_idx = 0
            for b in range(bs):
                num_tgt = sizes[b]
                if num_tgt == 0:
                    continue

                out_mask = outputs["pred_masks"][b]
                tgt_mask = targets[b]["masks"].to(out_mask)

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with torch.autocast(enabled=False, device_type=out_mask.device.type):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    if out_mask.shape[0] == 0:
                        mask_l = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        dice_l = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        mask_l = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                        dice_l = batch_dice_loss_jit(out_mask, tgt_mask)

                cost_mask[b, :, tgt_idx : tgt_idx + num_tgt] = mask_l
                cost_dice[b, :, tgt_idx : tgt_idx + num_tgt] = dice_l
                tgt_idx += num_tgt

        # Flatten (bs, num_queries) -> (bs * num_queries)
        cost_mask = cost_mask.flatten(0, 1)
        cost_dice = cost_dice.flatten(0, 1)

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
            + self.cost_dice * cost_dice
            + self.cost_mask * cost_mask
        )
        C = C.view(bs, num_queries, -1).cpu()
        C[torch.isnan(C)] = 0.0
        C[torch.isinf(C)] = 0.0

        try:
            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]
        except:  # pylint: disable=bare-except
            print("warning: use SimpleMinsumMatcher")
            indices = []
            device = C.device
            for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
                weight_mat = c[i]
                idx_i = weight_mat.min(0)[1]
                idx_j = torch.arange(_size).to(device)
                indices.append((idx_i, idx_j))
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha=0.25,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        device = C.device
        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            weight_mat = c[i]
            idx_i = weight_mat.min(0)[1]
            idx_j = torch.arange(_size).to(device)
            indices.append((idx_i, idx_j))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(args):
    assert args.matcher_type in [
        "HungarianMatcher",
        "SimpleMinsumMatcher",
    ], f"Unknown args.matcher_type: {args.matcher_type}"
    if args.matcher_type == "HungarianMatcher":
        return HungarianMatcher(
            cost_class=args.set_cost_class,  # 5.0
            cost_bbox=args.set_cost_bbox,  # 1.0
            cost_giou=args.set_cost_giou,  # 0.0
            cost_mask=args.set_cost_mask,  # 1.0
            cost_dice=args.set_cost_dice,  # 1.0
            focal_alpha=args.focal_alpha,  # 0.25
        )
    elif args.matcher_type == "SimpleMinsumMatcher":
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
        )
    else:
        raise NotImplementedError(f"Unknown args.matcher_type: {args.matcher_type}")
