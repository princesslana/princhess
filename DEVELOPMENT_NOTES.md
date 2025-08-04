# Princhess Development Notes

## Experimental History

### AdamW Optimizer Testing (July 2025)

**Context**: Switched from plain Adam to AdamW optimizer across different networks to evaluate impact on tournament ELO.

**Results**:
- **Value Network**: ELO gains achieved with AdamW (specific metrics TBD)
- **MG Policy Network**: ELO gains achieved with AdamW (specific metrics TBD)
- **EG Policy Network**: 
  - Weight decay 0.01: No ELO gains
  - Weight decay 0.001: Small positive Elo, failed to reach SPRT significance (Epoch 12)
  - Weight decay 0.0: Small negative Elo (-0.57 ± 7.91), trending toward SPRT rejection (Epoch 12)
  - **Weight Decay Conclusion**: Weight decay setting (0.0 vs 0.001) shows minimal impact on final ELO performance. Both produce marginal results around main's strength.
  - **Hybrid Learning Rate Schedule (0.00075→0.0005)**: Fixed-node test +6.95 Elo (86% LOS), LTC test -2.58 Elo (SPRT rejected). Consistent pattern of fixed-node improvements not translating to LTC gains.
  - **Soft Target Weight Testing**: 
    - 0.0 (no soft target): Training loss 2.314, fixed-node +2.43 Elo (64% LOS), LTC -16.45 Elo (severe overfitting)
    - 0.15 (increased regularization): Training loss 2.827, fixed-node +5.56 Elo (80% LOS), LTC +0.89 Elo (59% LOS, trending toward rejection)
  - **Consistent Pattern**: Multiple hyperparameter approaches (weight decay, learning rates, soft target weights) show training improvements and promising fixed-node results, but consistently fail to achieve meaningful LTC ELO gains.
- **Notable Finding**: EG policy epoch 12 net achieved higher ELO than epoch 13 net, despite epoch 13 showing better loss/accuracy metrics

### EG Policy Material Threshold Testing (July 2025)

**Context**: Testing different material count thresholds to define endgame phase for EG policy network training.

**Results**:
- **Material ≤ 6**: Tested, not effective (lower ELO)
- **Material ≤ 8**: Tested, achieved higher ELO than material ≤ 6

**Technical Notes**:
- Only two phase definitions have been experimentally tested
- Material count includes both sides' pieces

**Technical Notes**:
- AdamW differs from Adam by decoupling weight decay from gradient updates
- Weight decay acts as L2 regularization applied directly to weights
- Different networks show varying sensitivity to weight decay hyperparameter

**Strategic Implications**:
- Value and MG policy networks benefit from AdamW's regularization
- EG policy network may be more sensitive to overfitting or requires different regularization approach
- Loss/accuracy metrics may not correlate with ELO performance for EG policy training