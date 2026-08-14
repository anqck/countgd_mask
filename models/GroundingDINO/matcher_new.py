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
    # print(f"{pos=}")
    # neg: [nq, P]
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    # print(f"{neg=}")

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

        # print(outputs.keys())
        # Inputs:
        # outputs["pred_logits"]: [bs, nq, C]
        # outputs["pred_boxes"]:   [bs, nq, 4]
        # outputs["pred_masks"]:   [bs, nq, H, W]
        #
        # targets: list of dicts, each with:
        #   labels: [T_b]
        #   boxes:  [T_b, 4]
        #   masks:  [T_b, H, W]
        #
        # label_map: [num_dataset_classes, C]

        bs, num_queries = outputs["pred_logits"].shape[:2]

        alpha = self.focal_alpha
        gamma = 2.0

        indices = []

        for b in range(bs):
            # Input: outputs["pred_logits"][b].sigmoid()
            # [nq, C] -> [nq, C]
            out_prob = outputs["pred_logits"][b].sigmoid()

            # Input: outputs["pred_boxes"][b]
            # [nq, 4]
            out_bbox = outputs["pred_boxes"][b]

            # [T]
            tgt_ids = targets[b]["labels"]

            # [T, 4]
            tgt_bbox = targets[b]["boxes"]

            num_tgt = tgt_ids.shape[0]

            # ---------------------------------------------------------------
            # Bounding-box L1 cost
            # ---------------------------------------------------------------
            # Input: out_bbox[:, :2]       -> [nq, 2]
            # Input: tgt_bbox[:, :2]       -> [T, 2]
            # torch.cdist(..., p=1) output -> [nq, T]
            cost_bbox = torch.cdist(out_bbox[:, :2], tgt_bbox[:, :2], p=1)

            # ---------------------------------------------------------------
            # GIoU cost
            # ---------------------------------------------------------------
            # Input: box_cxcywh_to_xyxy(out_bbox) -> [nq, 4]
            # Input: box_cxcywh_to_xyxy(tgt_bbox) -> [T, 4]
            # generalized_box_iou(...)            -> [nq, T]
            # Negative                            -> [nq, T]
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            # ---------------------------------------------------------------
            # Classification cost
            # ---------------------------------------------------------------
            # Input: out_prob                          -> [nq, C]
            # neg_cost_class                           -> [nq, C]
            neg_cost_class = (
                (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            )

            # pos_cost_class                           -> [nq, C]
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )

            if num_tgt == 0:
                # Input: torch.zeros_like(cost_bbox)
                # cost_bbox: [nq, 0] -> [nq, 0]
                cost_class = torch.zeros_like(cost_bbox)
            else:
                # Input: label_map[tgt_ids.cpu()]
                # [T] -> indexing -> [T, C]
                local_label_map = label_map[tgt_ids.cpu()].to(pos_cost_class.device)

                # Input: local_label_map.sum(dim=1, keepdim=True)
                # [T, C] -> [T, 1]
                # local_label_map / ... -> [T, C]
                local_label_map = local_label_map / local_label_map.sum(
                    dim=1, keepdim=True
                )

                # Input: pos_cost_class @ local_label_map.T
                # [nq, C] @ [C, T] -> [nq, T]
                # Input: neg_cost_class @ local_label_map.T
                # [nq, C] @ [C, T] -> [nq, T]
                # Difference -> [nq, T]
                cost_class = (
                    pos_cost_class @ local_label_map.T
                    - neg_cost_class @ local_label_map.T
                )

            # ---------------------------------------------------------------
            # Mask cost
            # ---------------------------------------------------------------
            # print(self.generate_mask , num_tgt)
            if self.generate_mask and num_tgt > 0:
                # Input: outputs["pred_masks"][b] -> [nq, H, W]
                out_mask = outputs["pred_masks"][b]

                # Input: targets[b]["masks"].to(out_mask) -> [T, H, W]
                tgt_mask = targets[b]["masks"].to(out_mask)


                # print(f"{tgt_mask.shape=}")
                # print(out_mask.shape, tgt_mask.shape)

                # Input: out_mask[:, None] -> [nq, 1, H, W]
                out_mask = out_mask[:, None]

                # Input: tgt_mask[:, None] -> [T, 1, H, W]
                tgt_mask = tgt_mask[:, None]

                # print(out_mask.shape, tgt_mask.shape)

                # Input: torch.rand(...) -> [1, P, 2]
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

                # Input: point_coords.repeat(tgt_mask.shape[0], 1, 1)
                # [1, P, 2] -> [T, P, 2]
                #
                # point_sample(tgt_mask, ...):
                #   input:  [T, 1, H, W]
                #   coords: [T, P, 2]
                #   return: [T, 1, P]
                #
                # .squeeze(1) -> [T, P]
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                # Input: point_coords.repeat(out_mask.shape[0], 1, 1)
                # [1, P, 2] -> [nq, P, 2]
                #
                # point_sample(out_mask, ...):
                #   input:  [nq, 1, H, W]
                #   coords: [nq, P, 2]
                #   return: [nq, 1, P]
                #
                # .squeeze(1) -> [nq, P]
                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with torch.autocast(enabled=False, device_type=out_mask.device.type):
                    # Input: out_mask.float() -> [nq, P]
                    out_mask = out_mask.float()

                    # Input: tgt_mask.float() -> [T, P]
                    tgt_mask = tgt_mask.float()

                    if out_mask.shape[0] == 0:
                        # batch_sigmoid_ce_loss: [nq, P] x [T, P] -> [nq, T]
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

                        # batch_dice_loss: [nq, P] x [T, P] -> [nq, T]
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        # JIT versions -> [nq, T]
                        cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                        cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

                # print(cost_mask)
                    # print("AAA", cost_dice)
                    # print(out_mask.mean())
                    # print(out_mask.std())
                    # print(out_mask.min())
                    # print(out_mask.max())
                    # assert 1 == 0

            else:
                # Input: torch.zeros_like(cost_bbox)
                # cost_bbox: [nq, T] -> [nq, T]
                cost_mask = torch.zeros_like(cost_bbox)

                # [nq, T]
                cost_dice = torch.zeros_like(cost_bbox)

            # assert 1 == 0

            # ---------------------------------------------------------------
            # Final matching cost
            # ---------------------------------------------------------------
            # All terms: [nq, T]

            # cost_mask = torch.nan_to_num(cost_mask, nan=0.0, posinf=0.0, neginf=0.0)
            # cost_dice = torch.nan_to_num(cost_dice, nan=0.0, posinf=0.0, neginf=0.0)
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
                + self.cost_dice * cost_dice
                + self.cost_mask * cost_mask
            )

            # print( (self.cost_bbox * cost_bbox).mean(), (self.cost_class * cost_class).mean()
            #     , (self.cost_giou * cost_giou).mean()
            #     , (self.cost_dice * cost_dice).mean()
            #     , (self.cost_mask * cost_mask).mean())
            # assert 1 == 2

            # [nq, T] on CPU
            C = C.cpu()

            C[torch.isnan(C)] = 0.0
            C[torch.isinf(C)] = 0.0

            # ---------------------------------------------------------------
            # Hungarian matching
            # ---------------------------------------------------------------
            if C.shape[1] == 0:
                idx_i = torch.empty(0, dtype=torch.int64)
                idx_j = torch.empty(0, dtype=torch.int64)
            else:
                try:
                    idx_i, idx_j = linear_sum_assignment(C)
                    idx_i = torch.as_tensor(idx_i, dtype=torch.int64)
                    idx_j = torch.as_tensor(idx_j, dtype=torch.int64)
                except Exception:  # pylint: disable=broad-except
                    print("warning: use SimpleMinsumMatcher")
                    # min along query dimension: [nq, T] -> [T]
                    idx_i = C.min(dim=0).indices
                    idx_j = torch.arange(C.shape[1], dtype=torch.int64)

            indices.append((idx_i, idx_j))

        return indices


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
            generate_mask=args.generate_mask,  # True
            num_points = args.mask_num_points
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
