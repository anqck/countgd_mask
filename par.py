import random

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math


from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# ----------------------------------------------------------------------
# Original matcher (exact copy of the first snippet)
# ----------------------------------------------------------------------
class OriginalMatcher:
    def __init__(
        self,
        cost_bbox,
        cost_class,
        cost_giou,
        cost_dice,
        cost_mask,
        focal_alpha,
        generate_mask,
        num_points,
    ):
        self.cost_bbox = cost_bbox
        self.cost_class = cost_class
        self.cost_giou = cost_giou
        self.cost_dice = cost_dice
        self.cost_mask = cost_mask
        self.focal_alpha = focal_alpha
        self.generate_mask = generate_mask
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, targets, label_map):
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

            def point_sample(input, point_coords, **kwargs):
                add_dim = False
                if point_coords.dim() == 3:
                    add_dim = True
                    point_coords = point_coords.unsqueeze(2)
                output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
                if add_dim:
                    output = output.squeeze(3)
                return output

            def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
                inputs = inputs.sigmoid()
                inputs = inputs.flatten(1)
                numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
                denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
                loss = 1 - (numerator + 1) / (denominator + 1)
                return loss

            def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
                hw = inputs.shape[1]

                pos = F.binary_cross_entropy_with_logits(
                    inputs, torch.ones_like(inputs), reduction="none"
                )
                neg = F.binary_cross_entropy_with_logits(
                    inputs, torch.zeros_like(inputs), reduction="none"
                )

                loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
                    "nc,mc->nm", neg, (1 - targets)
                )

                return loss / hw

            batch_dice_loss_jit = torch.jit.script(batch_dice_loss)
            batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)

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


# ----------------------------------------------------------------------
# Refactored matcher (loop-based, helpers extracted)
# ----------------------------------------------------------------------
class RefactoredMatcher:
    def __init__(
        self,
        cost_bbox,
        cost_class,
        cost_giou,
        cost_dice,
        cost_mask,
        focal_alpha,
        generate_mask,
        num_points,
    ):
        self.cost_bbox = cost_bbox
        self.cost_class = cost_class
        self.cost_giou = cost_giou
        self.cost_dice = cost_dice
        self.cost_mask = cost_mask
        self.focal_alpha = focal_alpha
        self.generate_mask = generate_mask
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, targets, label_map):
        print("Called new forward")

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
            if self.generate_mask and num_tgt > 0:
                print("Check")
                # Input: outputs["pred_masks"][b] -> [nq, H, W]
                out_mask = outputs["pred_masks"][b]

                # Input: targets[b]["masks"].to(out_mask) -> [T, H, W]
                tgt_mask = targets[b]["masks"].to(out_mask)

                # Input: out_mask[:, None] -> [nq, 1, H, W]
                out_mask = out_mask[:, None]

                # Input: tgt_mask[:, None] -> [T, 1, H, W]
                tgt_mask = tgt_mask[:, None]

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
            else:
                # Input: torch.zeros_like(cost_bbox)
                # cost_bbox: [nq, T] -> [nq, T]
                cost_mask = torch.zeros_like(cost_bbox)
                # [nq, T]
                cost_dice = torch.zeros_like(cost_bbox)

            # ---------------------------------------------------------------
            # Final matching cost
            # ---------------------------------------------------------------
            # All terms: [nq, T]
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
                + self.cost_dice * cost_dice
                + self.cost_mask * cost_mask
            )
            print(f"{cost_mask=}")
            print(f"{cost_bbox.shape=}, {cost_mask.shape=}")
            if (
                any(torch.isnan(cost_mask).flatten().tolist())
                or cost_mask.shape[1] == 0
            ):
                print(f"{num_tgt=}")
                print(f"{out_mask.shape=}")
                print(f"{tgt_mask.shape=}")

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


# ----------------------------------------------------------------------
# Global helper functions needed by RefactoredMatcher (must be defined)
# ----------------------------------------------------------------------
def point_sample(input, point_coords, **kwargs):
    # Input: input: [N, C, H, W]
    # Input: point_coords: [N, P, 2]
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        # [N, P, 1, 2]
        point_coords = point_coords.unsqueeze(2)

    # [N, C, P, 1]
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)

    if add_dim:
        # [N, C, P, 1] -> [N, C, P]
        output = output.squeeze(3)

    return output


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    # Input: inputs: [nq, P]
    # Input: targets: [T, P]
    inputs = inputs.sigmoid()  # [nq, P]
    inputs = inputs.flatten(1)  # [nq, P]

    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)  # [nq, T]
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]  # [nq, T]
    loss = 1 - (numerator + 1) / (denominator + 1)  # [nq, T]
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    # Input: inputs: [nq, P]
    # Input: targets: [T, P]
    hw = inputs.shape[1]  # P

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )  # [nq, P]
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )  # [nq, P]

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )  # [nq, T]
    return loss / hw


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)
batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


# ----------------------------------------------------------------------
# Test function
# ----------------------------------------------------------------------
def run_parity_test(seed=42, verbose=False):
    torch.manual_seed(seed)

    # Hyperparameters
    bs = 4
    num_queries = 8
    num_classes = 4  # model vocabulary size
    num_dataset_classes = 6  # dataset label space size
    H, W = 8, 8
    num_points = 125

    # Generate random label_map (rows sum > 0)
    label_map = torch.rand(num_dataset_classes, num_classes)
    label_map = label_map / label_map.sum(dim=1, keepdim=True)  # just to be safe
    # We don't need to normalize, original normalizes per target later

    # Generate targets for each image
    targets = []
    for _ in range(bs):
        # Random number of targets (including possibly zero)
        T = torch.randint(1, 4, (1,)).item()  # 1..3 targets
        labels = torch.randint(1, num_dataset_classes, (T,))
        boxes = torch.rand(T, 4) * 0.5 + 0.25  # cxcywh in [0.25, 0.75]
        # Make sure boxes are valid (width, height > 0)
        boxes[:, 2:] = boxes[:, 2:] * 0.5 + 0.1
        masks = torch.randint(1, 2, (T, H, W)).float()
        targets.append({"labels": labels, "boxes": boxes, "masks": masks})

    # Generate outputs
    pred_logits = torch.randn(bs, num_queries, num_classes)
    pred_boxes = torch.rand(bs, num_queries, 4) * 0.5 + 0.25  # cxcywh
    pred_boxes[:, :, 2:] = pred_boxes[:, :, 2:] * 0.5 + 0.1
    pred_masks = torch.randn(bs, num_queries, H, W)  # logits

    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
    }

    # Instantiate matchers with same hyperparameters
    common_kwargs = dict(
        cost_bbox=1.0,
        cost_class=2.0,
        cost_giou=3.0,
        cost_dice=4.0,
        cost_mask=5.0,
        focal_alpha=0.25,
        generate_mask=True,
        num_points=num_points,
    )
    matcher_orig = OriginalMatcher(**common_kwargs)
    matcher_ref = RefactoredMatcher(**common_kwargs)

    # Run original, resetting seed to ensure same random point_coords
    torch.manual_seed(seed)
    indices_orig = matcher_orig.forward(outputs, targets, label_map)

    # Run refactored, same seed
    torch.manual_seed(seed)
    indices_ref = matcher_ref.forward(outputs, targets, label_map)

    # Compare
    assert len(indices_orig) == len(indices_ref) == bs, "Batch size mismatch"
    for b in range(bs):
        i_orig, j_orig = indices_orig[b]
        i_ref, j_ref = indices_ref[b]
        if not (torch.equal(i_orig, i_ref) and torch.equal(j_orig, j_ref)):
            print(f"Seed {seed}, image {b}: mismatch!")
            print(f"  Original: ({i_orig.tolist()}, {j_orig.tolist()})")
            print(f"  Refactored: ({i_ref.tolist()}, {j_ref.tolist()})")
            return False
    if verbose:
        print(f"Seed {seed}: all indices match!")
    return True


if __name__ == "__main__":
    all_pass = True
    for seed in [random.randint(0, 999_999_999) for _ in range(10000)]:
        all_pass &= run_parity_test(seed=seed, verbose=True)
    if all_pass:
        print("\n✅ All parity tests passed!")
    else:
        print("\n❌ Some tests failed.")
