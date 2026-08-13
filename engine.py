# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F


import util.misc as utils
from datasets.cocogrounding_eval import CocoGroundingEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.utils import to_device

# from skimage.filters import threshold_otsu

FONT_CONFIG = {
    "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}


def get_xy_from_boxes(boxes, image):
    """
    Get box centers in image coordinates for a batch of xyxy boxes.
    """
    if len(boxes) == 0:
        return np.array([]), np.array([])

    w, h = image.size
    x = w * boxes[:, 0]
    y = h * boxes[:, 1]

    return x, y


def visualize_masks(
    pred_masks,
    image,
    output_path,
    boxes=None,
    gt_points=None,
    pred_count=None,
    gt_count=None,
    image_id=None,
    class_name=None,
):
    """
    Visualizes predicted binary masks overlaid on an input image along with predicted points,
    ground truth points, and text info using PIL, and saves to output_path.

    Parameters:
    - pred_masks: Tensor of shape (K, H_m, W_m) or a list of such tensors.
    - image: torch.Tensor of shape (3, H, W) or numpy array (H, W, 3) or PIL.Image.
    - output_path: str, filepath to save the visualization.
    - boxes: Tensor or ndarray of predicted box centers/boxes, shape (N, 4) or (N, 2).
    - gt_points: list, Tensor, or ndarray of ground truth points [[x, y], ...].
    - pred_count: int, predicted object count.
    - gt_count: int, ground truth object count.
    - image_id: str or int, identifier of the image.
    - class_name: str, class name or category label.
    """
    if isinstance(pred_masks, list):
        if len(pred_masks) == 0:
            pred_masks = None
        else:
            pred_masks = pred_masks[-1]

    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.detach().cpu()

    # Convert input image to PIL RGBA Image
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().float().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.max() > 1.0 or img_np.min() < 0.0:
            img_min, img_max = img_np.min(), img_np.max()
            if img_max > img_min:
                img_np = (img_np - img_min) / (img_max - img_min)
            else:
                img_np = np.clip(img_np, 0.0, 1.0)
        img_uint8 = (img_np * 255).astype(np.uint8)
        base_img = Image.fromarray(img_uint8).convert("RGBA")
    elif isinstance(image, np.ndarray):
        img_np = image
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        base_img = Image.fromarray(img_np.astype(np.uint8)).convert("RGBA")
    elif isinstance(image, Image.Image):
        base_img = image.convert("RGBA")
    else:
        return

    w_img, h_img = base_img.size

    # 1. Overlay predicted masks if present
    if pred_masks is not None and pred_masks.numel() > 0 and pred_masks.shape[0] > 0:
        if pred_masks.ndim == 3:
            masks_tensor = pred_masks.unsqueeze(0).float()
            if (masks_tensor.shape[-2], masks_tensor.shape[-1]) != (h_img, w_img):
                masks_tensor = F.interpolate(
                    masks_tensor,
                    size=(h_img, w_img),
                    mode="bilinear",
                    align_corners=False,
                )
            masks_tensor = masks_tensor.squeeze(0)

            prob_masks = (
                masks_tensor.sigmoid() if masks_tensor.min() < 0.0 else masks_tensor
            )
            binary_masks = (prob_masks > 0.8).numpy()

            num_masks = binary_masks.shape[0]
            print(f"Number of masks prediced {num_masks=}")
            palette = [
                (255, 59, 48),  # Red
                (52, 199, 89),  # Green
                (0, 122, 255),  # Blue
                (255, 149, 0),  # Orange
                (175, 82, 222),  # Purple
                (255, 204, 0),  # Yellow
                (85, 190, 240),  # Cyan
                (255, 45, 85),  # Pink
                (162, 132, 94),  # Brown
                (142, 142, 147),  # Gray
            ]

            for m_idx in range(num_masks):
                m = binary_masks[m_idx]
                if m.any():
                    color = palette[m_idx % len(palette)]
                    mask_layer_np = np.zeros((h_img, w_img, 4), dtype=np.uint8)
                    mask_layer_np[m, 0:3] = color
                    mask_layer_np[m, 3] = 128  # 50% alpha transparency
                    mask_layer_img = Image.fromarray(mask_layer_np, mode="RGBA")
                    comp_img = Image.alpha_composite(base_img, mask_layer_img)
                    base_img.close()
                    mask_layer_img.close()
                    base_img = comp_img

    # 2. Point and Text Visualisation (integrated from vis_v4.py)
    scale = max(w_img, h_img) / 1000.0
    r = max(1, int(4 * scale))

    font_size_text = max(12, int(14 * scale))
    font_size_markers = max(10, int(12 * scale))

    try:
        font_text = ImageFont.truetype(FONT_CONFIG["regular"], size=font_size_text)
        font_marker = ImageFont.truetype(FONT_CONFIG["bold"], size=font_size_markers)
    except IOError:
        font_text = ImageFont.load_default()
        font_marker = ImageFont.load_default()

    # (a) Lowest Pred Layer (Semi-transparent Red 'x' markers)
    if boxes is not None and len(boxes) > 0:
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.detach().cpu().numpy()
        else:
            boxes_np = np.array(boxes)

        if boxes_np.ndim == 2 and boxes_np.shape[0] > 0:
            x_pred, y_pred = get_xy_from_boxes(boxes_np, base_img)

            pred_layer = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
            draw_pred = ImageDraw.Draw(pred_layer)
            for x, y in zip(x_pred, y_pred):
                draw_pred.text(
                    (x, y), "x", fill=(255, 0, 0, 255), font=font_marker, anchor="mm"
                )
            comp_img = Image.alpha_composite(base_img, pred_layer)
            base_img.close()
            pred_layer.close()
            base_img = comp_img

    # (b) Lower GT Layer (Semi-transparent Blue circles)
    if gt_points is not None and len(gt_points) > 0:
        if isinstance(gt_points, torch.Tensor):
            gt_points_np = gt_points.detach().cpu().numpy()
        else:
            gt_points_np = np.array(gt_points)

        if gt_points_np.ndim == 2 and gt_points_np.shape[0] > 0:
            gt_pts = gt_points_np.copy()
            # If normalized coordinates (0..1), scale to pixel coordinates
            if (
                gt_pts[:, 0].max() <= 1.0
                and gt_pts[:, 1].max() <= 1.0
                and w_img > 1
                and h_img > 1
            ):
                gt_pts[:, 0] *= w_img
                gt_pts[:, 1] *= h_img

            gt_layer = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
            draw_gt = ImageDraw.Draw(gt_layer)
            for x, y in gt_pts:
                draw_gt.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 255, 128))
            comp_img = Image.alpha_composite(base_img, gt_layer)
            base_img.close()
            gt_layer.close()
            base_img = comp_img

    # (c) Highest Layer (Opaque Top-Left and Top-Right Info Text)
    if (
        image_id is not None
        or class_name is not None
        or pred_count is not None
        or gt_count is not None
    ):
        text_layer = Image.new("RGBA", (w_img, h_img), (0, 0, 0, 0))
        draw_text_layer = ImageDraw.Draw(text_layer)

        # Render Top Left Info
        img_id_str = str(image_id) if image_id is not None else ""
        cls_str = str(class_name) if class_name is not None else ""
        left_text = f"{img_id_str}|{cls_str}".strip()
        if left_text and left_text != "|":
            draw_text_layer.text(
                (int(0.02 * w_img), int(0.02 * h_img)),
                left_text,
                fill=(0, 0, 0, 255),
                font=font_text,
            )

        # Render Top Right Info
        if gt_count is not None:
            gt_text = f"GT:{gt_count}"
            gt_w = draw_text_layer.textlength(gt_text, font=font_text)
            draw_text_layer.text(
                (w_img - gt_w - int(0.02 * w_img), int(0.02 * h_img)),
                gt_text,
                fill=(0, 0, 255, 255),
                font=font_text,
            )

        if pred_count is not None:
            pred_text = f"Pred:{pred_count}"
            pred_w = draw_text_layer.textlength(pred_text, font=font_text)
            draw_text_layer.text(
                (
                    w_img - pred_w - int(0.02 * w_img),
                    int(0.02 * h_img) + font_size_text + 4,
                ),
                pred_text,
                fill=(255, 0, 0, 255),
                font=font_text,
            )

        comp_img = Image.alpha_composite(base_img, text_layer)
        base_img.close()
        text_layer.close()
        base_img = comp_img

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rgb_final = base_img.convert("RGB")
    rgb_final.save(output_path)
    base_img.close()
    rgb_final.close()


def make_interval_nested(df, intervals):
    """
    Iterates through flexible interval boundaries to group filenames by class.

    Parameters:
    - df: pd.DataFrame containing 'gt_cnt' column
    - intervals: List of tuples representing intervals, e.g., [(2, 5), (3,), (None, 4), (2, -1)].
    """
    for interval in intervals:
        # Extract boundaries supporting variable tuple lengths
        low = interval[0] if len(interval) > 0 else None
        high = interval[1] if len(interval) > 1 else None

        # Initialize an all-True Boolean mask matching the DataFrame index
        mask = pd.Series(True, index=df.index)

        # Apply lower bound constraint if present and valid
        if low is not None and low != -1:
            mask &= df["gt_cnt"] >= low

        # Apply upper bound constraint if present and valid
        if high is not None and high != -1:
            mask &= df["gt_cnt"] <= high

        # Generate the tracking label based on the active constraints
        is_low_bound = low is not None and low != -1
        is_high_bound = high is not None and high != -1

        if is_low_bound and is_high_bound:
            label = f"{low}-{high}"
        elif is_low_bound:
            label = f">={low}"
        elif is_high_bound:
            label = f"<={high}"
        else:
            label = "unbounded"

        # Filter the target DataFrame using the compiled mask
        yield label, df[mask]


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = f"Epoch: [{epoch}]"
    print_freq = 1
    print(len(data_loader))

    _cnt = 0

    for samples, targets in metric_logger.log_every(
        data_loader, print_freq, header, logger=logger
    ):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        exemplars = [t["exemplars"].to(device) for t in targets]
        labels_uncropped = [t["labels_uncropped"].to(device) for t in targets]
        shot_num = 0
        exemplars = [exemp[:shot_num] for exemp in exemplars]
        for exemp in exemplars:
            if exemp.shape[0] > 3:
                print(
                    "WARNING: Exemp shape greater than 3!!! Only 3 exemplars allowed during training"
                )
        targets = [
            {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
            for t in targets
        ]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, exemplars, labels_uncropped, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
            )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat


def plot_points(image, exemplars, size, points):
    h, w = (size[0], size[1])
    for point in points:
        in_exemp = (point[0] * w > exemplars[:, 0]) * (point[0] * w < exemplars[:, 2])
        in_exemp = (
            (in_exemp)
            * (point[1] * h > exemplars[:, 1])
            * (point[1] * h < exemplars[:, 3])
        )
        if in_exemp.sum() > 0:
            plt.plot(point[0] * w, point[1] * h, marker="v", color="red")
        else:
            plt.plot(point[0] * w, point[1] * h, marker="v", color="white")
    for exemp in exemplars:
        plt.gca().add_patch(
            Rectangle(
                (exemp[0], exemp[1]),
                exemp[2] - exemp[0],
                exemp[3] - exemp[1],
                edgecolor="red",
                facecolor="none",
                lw=4,
            )
        )
    plt.imshow(image)
    plt.savefig("sunglasses")


def tt_norm(pred_cnt, exemplars, size, points):
    e_cnt = 0
    h, w = (size[0], size[1])
    for point in points:
        in_exemp = (point[0] * w > exemplars[:, 0]) * (point[0] * w < exemplars[:, 2])
        in_exemp = (
            (in_exemp)
            * (point[1] * h > exemplars[:, 1])
            * (point[1] * h < exemplars[:, 3])
        )
        if in_exemp.sum() > 0:
            e_cnt += 1
    e_cnt = e_cnt / exemplars.shape[0]
    if e_cnt >= (5 / 3):
        # At least 2 of the exemplars contain 2 object instances.
        pred_cnt = pred_cnt / e_cnt
    return pred_cnt


def get_count_errs(
    samples,
    _exemplars,
    outputs,
    box_threshold,
    text_threshold,
    targets,
    tokenized_captions,
    _input_captions,
    counts=None,
    count_output_state_dict=None,
    save_masks: bool = False,
):
    # pylint: disable=consider-using-enumerate
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    if save_masks:
        masks = outputs.get("pred_masks", None)
    else:
        masks = None
    samples = samples.to_img_list()

    abs_errs = []
    for sample_ind in range(len(targets)):
        sample_logits = logits[sample_ind]
        sample_boxes = boxes[sample_ind]
        sample_masks = masks[sample_ind] if masks is not None else None

        end_idx = 0
        for token_ind in range(len(tokenized_captions["input_ids"][sample_ind])):
            idx = tokenized_captions["input_ids"][sample_ind][token_ind]
            if idx == 1012:
                end_idx = token_ind
                break

        box_mask = sample_logits.max(dim=-1).values > box_threshold
        sample_logits = sample_logits[box_mask, :]
        sample_boxes = sample_boxes[box_mask, :]
        if sample_masks is not None:
            sample_masks = sample_masks[box_mask]

        text_mask = (sample_logits[:, 1:end_idx] > text_threshold).sum(dim=-1) == (
            end_idx - 1
        )
        sample_logits = sample_logits[text_mask, :]
        sample_boxes = sample_boxes[text_mask, :]
        if sample_masks is not None:
            sample_masks = sample_masks[text_mask]

        gt_count = targets[sample_ind]["labels"].shape[0]
        pred_cnt = sample_logits.shape[0]

        if counts is not None:
            counts.append((pred_cnt, gt_count))

        if count_output_state_dict is not None:
            sample_scores = sample_logits.max(dim=-1).values
            count_info = torch.cat((sample_boxes, sample_scores.unsqueeze(-1)), dim=1)

            if "count_info" not in count_output_state_dict:
                count_output_state_dict["count_info"] = []
            if "image_ids" not in count_output_state_dict:
                count_output_state_dict["image_ids"] = []
            if "pred_cnt" not in count_output_state_dict:
                count_output_state_dict["pred_cnt"] = []
            if "gt_cnt" not in count_output_state_dict:
                count_output_state_dict["gt_cnt"] = []
            if "pred_masks" not in count_output_state_dict and save_masks:
                count_output_state_dict["pred_masks"] = []

            count_output_state_dict["count_info"].append(count_info.cpu())
            count_output_state_dict["image_ids"].append(
                int(targets[sample_ind]["image_id"].item())
            )
            count_output_state_dict["pred_cnt"].append(pred_cnt)
            count_output_state_dict["gt_cnt"].append(gt_count)
            if save_masks:
                if sample_masks is not None:
                    count_output_state_dict["pred_masks"].append(sample_masks.cpu())
                else:
                    count_output_state_dict["pred_masks"].append(
                        torch.zeros((gt_count, 300, 300))
                    )

        # print("Pred Count: " + str(pred_cnt) + ", GT Count: " + str(gt_count))
        abs_errs.append(np.abs(gt_count - pred_cnt))
    return abs_errs


def parse_results_and_dataset(result, _):
    rows = []
    print(*map(len, [result["image_ids"], result["pred_cnt"], result["gt_cnt"]]))
    for i, k, v in zip(result["image_ids"], result["pred_cnt"], result["gt_cnt"]):
        rows.append((i, k, v))

    return rows


@torch.no_grad()
def evaluate(
    model,
    _model_without_ddp,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir: str,
    wo_class_error=False,
    *,
    args,
    logger=None,
):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors)
    try:
        useCats = args.useCats
    except:  # noqa: E722
        useCats = True
    if not useCats:
        print(f"useCats: {useCats} !!!")

    coco_evaluator = None
    if base_ds is not None:
        coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)
    else:
        print("No COCO-format val annotations; skipping COCO evaluation (MAE only).")

    panoptic_evaluator = None
    if "panoptic" in postprocessors:
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    count_output_state_dict = {}

    if args.use_coco_eval:
        from pycocotools.coco import COCO

        coco = COCO(args.coco_val_path)
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item["name"] for item in category_dict]
    else:
        cat_list = args.val_label_list
    caption = " . ".join(cat_list) + " ."
    print("Input text prompt:", caption)

    abs_errs = []
    counts = []
    for samples, targets in metric_logger.log_every(
        data_loader, 10, header, logger=logger
    ):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        # exemplars = [t["exemplars"].to(device) for t in targets]
        _labels = [t["labels"].to(device) for t in targets]
        exemplars = [torch.tensor([]).to(device) for t in targets]
        _bs = samples.tensors.shape[0]
        input_captions = [cat_list[target["labels"][0]] + " ." for target in targets]
        # print("input_captions: " + str(input_captions))
        with torch.amp.autocast("cuda", enabled=args.amp):
            outputs = model(
                samples,
                exemplars,
                [torch.tensor([0]).to(device) for t in targets],
                captions=input_captions,
            )

        tokenized_captions = outputs["token"]
        abs_errs += get_count_errs(
            samples,
            exemplars,
            outputs,
            args.box_threshold,
            args.text_threshold,
            targets,
            tokenized_captions,
            input_captions,
            counts,
            count_output_state_dict,
            args.save_results,
        )
        counts[-1] = (targets[0]["image_id"].item(),) + counts[-1]

        if args.eval and args.save_results:
            img_list = samples.to_img_list()
            for sample_ind, tgt in enumerate(targets):
                img_id: int = tgt["image_id"].item()
                out_dir: str = (
                    output_dir or getattr(args, "output_dir", ".") or "."
                ) + "/masks-vis"
                out_filename = os.path.join(out_dir, f"mask_{int(img_id)}.png")
                batch_relative_idx = sample_ind - len(targets)
                sample_pred_masks = count_output_state_dict["pred_masks"][
                    batch_relative_idx
                ]
                sample_count_info = count_output_state_dict["count_info"][
                    batch_relative_idx
                ]
                sample_boxes = (
                    sample_count_info[:, :4]
                    if sample_count_info is not None and sample_count_info.numel() > 0
                    else None
                )
                sample_pred_cnt = count_output_state_dict["pred_cnt"][
                    batch_relative_idx
                ]
                sample_gt_cnt = count_output_state_dict["gt_cnt"][batch_relative_idx]

                class_name = tgt.get("class_name", "")
                if not class_name and "labels" in tgt and len(tgt["labels"]) > 0:
                    cat_idx = tgt["labels"][0].item()
                    if cat_idx < len(cat_list):
                        class_name = cat_list[cat_idx]

                gt_points = None
                if "points" in tgt:
                    gt_points = tgt["points"]
                elif "point" in tgt:
                    gt_points = tgt["point"]
                elif "boxes" in tgt:
                    gt_points = tgt["boxes"][:, :2]
                visualize_masks(
                    sample_pred_masks,
                    img_list[sample_ind],
                    out_filename,
                    boxes=sample_boxes,
                    gt_points=gt_points,
                    pred_count=sample_pred_cnt,
                    gt_count=sample_gt_cnt,
                    image_id=img_id,
                    class_name=class_name,
                )
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if "segm" in postprocessors:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for _, (tgt, res) in enumerate(zip(targets, results)):
                # pred vars:
                #     K: number of bbox pred
                #     score: Tensor(K),
                #     label: list(len: K),
                #     bbox: Tensor(K, 4)
                #     idx: list(len: K)
                # tgt: dict.
                # compare gt and res (after postprocess)
                gt_bbox = tgt["boxes"]
                gt_label = tgt["labels"]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res["boxes"]
                _res_prob = res["scores"]
                _res_label = res["labels"]
                res_info = torch.cat(
                    (_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1
                )

                if "gt_info" not in output_state_dict:
                    output_state_dict["gt_info"] = []
                output_state_dict["gt_info"].append(gt_info.cpu())

                if "res_info" not in output_state_dict:
                    output_state_dict["res_info"] = []
                output_state_dict["res_info"].append(res_info.cpu())
        _cnt += 1
        if args.debug and _cnt % 15 == 0:
            print("BREAK!" * 5)
            break
    count_mae = sum(abs_errs) / len(abs_errs)
    count_rmse = (np.array(abs_errs) ** 2).mean() ** (1 / 2)
    print("# of Images Tested: " + str(len(abs_errs)))
    print("MAE: " + str(count_mae) + ", RMSE: " + str(count_rmse))

    frame = pd.DataFrame(
        counts,
        columns=["image_id", "pred_cnt", "gt_cnt"],
    )
    target_intervals = [(1, 5), (6, 10), (11, 20), (21, 40), (41,)]
    headers = []
    values = []

    def calc_mae(gt, pred):
        return np.average(np.abs(np.array(gt) - np.array(pred)))

    def calc_rmse(gt, pred):
        return np.sum((np.array(pred) - np.array(gt)) ** 2 / len(gt)) ** 0.5

    for label, sub_df in make_interval_nested(frame, target_intervals):
        headers.append(label)
        print(f"Calculating MAE, RMSE {label}. {len(sub_df['gt_cnt'].values)} images")
        if len(sub_df) > 0:
            val_mae = calc_mae(sub_df["gt_cnt"].values, sub_df["pred_cnt"].values)
            val_rmse = calc_rmse(sub_df["gt_cnt"].values, sub_df["pred_cnt"].values)
            values.append((val_mae, val_rmse))
        else:
            values.append((0.0, 0.0))

    current_bins = []
    current_metrics = []
    for b, m in zip(headers, values):
        temp_bins = current_bins + [b]
        temp_metrics = current_metrics + [m]
        h_line = "".join(f"{x}\t\t" for x in temp_bins).rstrip("\t")
        m_line = "".join(f"{y[0]:.4f}\t{y[1]:.4f}\t" for y in temp_metrics).rstrip("\t")
        if len(current_bins) > 0 and (
            len(h_line.expandtabs(8)) > 80 or len(m_line.expandtabs(8)) > 80
        ):
            print("".join(f"{x}\t\t" for x in current_bins).rstrip("\t"))
            print(
                "".join(f"{y[0]:.4f}\t{y[1]:.4f}\t" for y in current_metrics).rstrip(
                    "\t"
                )
            )
            current_bins = [b]
            current_metrics = [m]
        else:
            current_bins = temp_bins
            current_metrics = temp_metrics
    if current_bins:
        print("".join(f"{x}\t\t" for x in current_bins).rstrip("\t"))
        print(
            "".join(f"{y[0]:.4f}\t{y[1]:.4f}\t" for y in current_metrics).rstrip("\t")
        )

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, "results-{}.pkl".format(utils.get_rank()))
        print(f"Saving res to {savepath}")
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if coco_evaluator is not None:
        if "bbox" in postprocessors:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]

    bins_result = {
        f"{h}_{suffix}": val
        for h, (mae, rmse) in zip(headers, values)
        for suffix, val in (("MAE", mae), ("RMSE", rmse))
    }

    return bins_result, count_mae, count_rmse, stats, coco_evaluator
