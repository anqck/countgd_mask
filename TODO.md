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
