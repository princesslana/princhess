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

### EG Policy Learning Rate Schedule Testing (August 2025)

**Context**: Testing EG policy with standard learning rate schedule: 0.001 → 0.0005 (single drop at 50% training).

**Training Results (13 epochs)**:
- Final loss: 2.6578
- Final accuracy: 40.84%

**Tournament Results**:
- **STC (25k nodes)**: +6.25 Elo ±12.95 (82.8% LOS, 1000 games)
- **LTC (40+0.4)**: -4.45 Elo ±9.11 (16.9% LOS, 1326 games)

**Network Analysis Data**:
- Piece attention utilization: bad_see_pawn 79.7%, good_see_pawn 83.3%, bad_see_queen 83.7%, king 70.3%
- Weight magnitude progression: E2 512k → E10 1.40M (bad_see_pawn_to_g8)
- Node attention variance: 3-5% range across attention heads

**Strategic Implications**:
- Value and MG policy networks benefit from AdamW's regularization
- EG policy network may be more sensitive to overfitting or requires different regularization approach
- Loss/accuracy metrics may not correlate with ELO performance for EG policy training

### Value Network King Bucketing Testing (August 2025)

**Context**: Testing consolidation of all endgame positions (≤6 major pieces) to king bucket 2 instead of using original KING_BUCKETS mapping.

**Results**:
- **Fixed-node test (25k nodes, 1000 games)**: -13.90 ±15.46 Elo (3.87% LOS)
- **Games**: 305 wins, 345 losses, 350 draws (48.00% points)

**Conclusion**: Endgame-aware king bucketing shows negative Elo. Reverted to original KING_BUCKETS logic.

### Value Network King Bucketing - f1/c1 Position Testing (August 2025)

**Context**: Testing moving f1 and c1 squares from king bucket 0 to king bucket 1 in KING_BUCKETS mapping.

**Training Results**:
- 9 epochs, 802M positions, LR schedule: 0.001 (epochs 1-6) → 0.0001 (epochs 7-9)
- Loss progression: 0.0696 → 0.0531 (23.6% reduction)
- Epoch 6: 0.0584 loss, Epoch 9: 0.0531 loss

**Tournament Results**:
- **Epoch 6 vs main**: -70.44 ±17.25 Elo (0.00% LOS, 1000 games)
- **Epoch 9 vs main**: -5.56 ±16.12 Elo (24.94% LOS, 1000 games)

**Conclusion**: Moving f1/c1 from bucket 0 to bucket 1 reduces performance. Original bucket mapping retained.