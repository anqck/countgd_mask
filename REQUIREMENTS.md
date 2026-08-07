# Instance Segmentation Head Requirements & Architectural Specification

This document details the functional and architectural requirements for attaching an instance segmentation mask branch to CountGD (built on GroundingDINO) by extracting components from MaskDINO.

---

## 1. Architectural & Functional Requirements

### 1.1 Non-Destructive Stage 0 Feature Tapping
- **Backbone Isolation**: Stage 0 (stride 4, 128 channels for Swin-B) features must be tapped from the Swin Transformer backbone without altering the primary DETR return path `return_interm_indices = [1, 2, 3]`.
- **Channel Stability**: The main return paths (`out`, `pos`) fed to `input_proj` and the Deformable Transformer Encoder must remain unchanged (`[256, 512, 1024]` channels for Swin-B) to preserve original DETR query mechanics.
- **Implementation**: Expose Stage 0 features through a secondary return element (`out, pos, layer0`) in [`Joiner.forward`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/backbone/backbone.py#L167) in [`backbone.py`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/backbone/backbone.py).

### 1.2 Transformer-Internal Mask Path
- **Encapsulation**: All mask processing logic—including 1/4-scale FPN feature fusion, query mask embedding (`mask_embed`), and mask prediction generation—must be encapsulated inside the [`Transformer`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/transformer.py#L40) class in [`transformer.py`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/transformer.py). Escalation into the top-level [`GroundingDINO`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/groundingdino.py#L77) model class must be strictly avoided.
- **FPN Feature Fusion**:
  - Unflatten the stride-8 encoder output `memory` for Level 0 (`H_0 * W_0` tokens) into a 2D feature map `(bs, d_model, H/8, W/8)`.
  - Project Stage 0 features using `layer0_proj` ($1 \times 1$ Conv + GroupNorm).
  - Interpolate stride-8 memory to 1/4 scale using bilinear interpolation and fuse with projected Stage 0 features.
  - Pass fused features through `fpn_output_conv` ($3 \times 3$ Conv + GroupNorm + ReLU) and `mask_features_proj` ($1 \times 1$ Conv) to produce `mask_features` ($256 \times H/4 \times W/4$).
- **Mask Prediction**:
  - Project decoder hidden states `hs` using a 3-layer `MLP` (`mask_embed`).
  - Generate mask predictions for each decoder layer via `torch.einsum("bqc,bchw->bqhw", mask_embed(hs_i), mask_features)`.

### 1.3 Surgical Mask Branch Ablation Toggle
- **Ablation Control**: Provide an explicit configuration option `enable_mask_branch` to surgically enable or disable the mask branch.
- **Bypass Behavior**: When `enable_mask_branch = False`:
  - Skip FPN feature fusion and mask prediction computation (`outputs_mask = None`).
  - Omit mask matching costs (`cost_mask`, `cost_dice`) in [`HungarianMatcher`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/matcher.py#L55).
  - Omit mask loss evaluation (`"masks"`) in [`SetCriterion`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/groundingdino.py#L645).

### 1.4 Loss Computation & Matching Costs
- **Point-Wise Mask Loss**: Implement mask Cross-Entropy (sigmoid BCE) and DICE losses in [`SetCriterion`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/groundingdino.py#L645) using point sampling (`point_sample`), adhering to the loss design in MaskDINO.
- **Initial Loss Coefficients**: Set initial loss coefficients to `1.0` for both mask CE (`loss_mask_coef`) and mask DICE (`dice_loss_coef`).
- **Hungarian Matcher**: Incorporate point-wise mask BCE (`batch_sigmoid_ce_loss`) and DICE (`batch_dice_loss`) matching costs into [`HungarianMatcher`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/models/GroundingDINO/matcher.py#L55) when mask targets are present.

### 1.5 Configuration File Management
- **Dataset Config Copy**: Copy [`cfg_fsc147_vit_b_odvg.py`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/config/cfg_fsc147_vit_b_odvg.py) to a new configuration file [`cfg_animal39.py`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/config/cfg_animal39.py) for the `Animal39` finetuning dataset.
- **Config Appending**: Append all new mask branch settings at the end of [`cfg_animal39.py`](file:///c:/Users/phanm/Projects/gitmanaged/CountGD/config/cfg_animal39.py):
  ```python
  enable_mask_branch = True
  loss_mask_coef = 1.0
  dice_loss_coef = 1.0
  cost_mask = 1.0
  cost_dice = 1.0
  ```

---

## 2. Verification Constraints

- **Execution Disclaimer**: Cannot execute training to verify correctness due to absence of local dataset files and pre-trained checkpoints.
- **Static Integrity**: Ensure code compiles cleanly via `python -m py_compile` without syntax or import errors.
