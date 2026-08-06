note:

- tap layer 0 (4-stride) from Swin backbone, project to transformer hidden dim (1-D Conv) (1)
- unflatten and FPN stride-8 `memory` to 1/4 scale (2)
- tap last decoder (`hs`) output, einsum with (1)+(2) (3)
- copy mask MLP over
- Run (3) over mask MLP
- Configure loss weight in config
- Copy loss functions to SetCriterion
- Request mask CE loss and DICE to be computed in Criterion/Config files
- Update final losses with mask CE and DICE for gradient to mask branch

references:

- mask forming: MaskDINO/maskdino/modeling/transformer_decoder

```
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output.transpose(0, 1),
                mask_features,
                self.training or (i == len(hs) - 1),
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
```