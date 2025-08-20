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

### Value Network Backpropagation Correctness & Weight Decay Testing (August 2025)

**Context**: Fixed SCreLU derivative bug (2.0 * x.sqrt() → 2.0 * x) and corrected backpropagation to use pre-activation values for derivative calculation. Original fix caused -17 ELO regression due to weight explosion in output layer.

**Problem**: Corrected backpropagation exposed weight explosion in NSTM output layer (weights grew from -247 to -1134 range).

**Solutions Tested**:
1. **Scaled Tanh Activation (tanh(x/100))**: Training completed, tournament testing showed ELO loss
2. **Gradient Clipping (max_norm=1.0)**: Ineffective (0% activation rate), problem was weight explosion not gradient explosion  
3. **Layer-Specific Weight Decay (0.01/0.10)**: Feature layers 0.01, Output layer 0.10
  - **Training**: 9 epochs, loss 0.0703 → 0.0534, controlled weight evolution
  - **Weight Analysis**: NSTM output weights [-323, 185] vs previous explosion to [-1134, 462]
  - **Fixed-node Test (25k nodes)**: -5.21 ±15.03 Elo (24.82% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -9.37 ±12.91 Elo (7.73% LOS, 816 games)

4. **Weight Decay Refinement (0.01/0.02)**: Reduced output layer weight decay to 0.02
  - **Training**: 9 epochs, loss 0.0700 → 0.0532
  - **Fixed-node Test (25k nodes)**: -6.60 ±15.15 Elo (19.64% LOS, 1000 games)  
  - **LTC Test (40+0.4s)**: -0.57 ±6.71 Elo (43.39% LOS, 3046 games)

5. **Weight Decay Further Refinement (0.01/0.03)**: Increased output layer weight decay to 0.03
  - **Training**: 9 epochs, loss 0.0701 → 0.0532
  - **Fixed-node Test (25k nodes)**: -8.69 ±15.69 Elo (13.87% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -12.23 ±14.51 Elo (4.90% LOS, 682 games)

6. **Layer-Specific Learning Rate (0.001/0.002)**: Feature layers 0.001, Output layer 0.002 with 0.02 weight decay
  - **Training**: 9 epochs, loss 0.0697 → 0.0532
  - **Weight Analysis**: STM output weights [-709, 1076], NSTM output weights [-682, 377], output bias 5043
  - **Fixed-node Test (25k nodes)**: -13.21 ±15.82 Elo (5.05% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -13.24 ±14.26 Elo (3.42% LOS, 604 games)

7. **Layer-Specific LR Schedule (0.5/0.7 drop points)**: Feature layers drop at 50% training, Output layers drop at 70% training
  - **Training**: 9 epochs, loss 0.0699 → 0.0532
  - **Fixed-node Test (25k nodes)**: -7.30 ±15.87 Elo (18.35% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -11.55 ±13.99 Elo (5.26% LOS, 692 games)

8. **Gentler LR Drop Factor (0.2)**: Uniform 0.7 drop point with 0.2 factor (0.001 → 0.0002)
  - **Training**: 9 epochs, loss 0.0593 → 0.0537
  - **Fixed-node Test (25k nodes)**: -14.25 ±15.35 Elo (3.41% LOS, 1000 games)

9. **Equal Weight Decay (0.02) + Gentler LR Drop (0.2)**: Both layers 0.02 weight decay with 0.2 factor
  - **Training**: 9 epochs, loss 0.0603 → 0.0541, gradient instability in final epochs
  - **Fixed-node Test (25k nodes)**: -9.73 ±15.68 Elo (11.16% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -3.37 ±9.24 Elo (23.75% LOS, 1652 games)

10. **Equal Weight Decay (0.02) + Original LR Drop (0.1)**: Both layers 0.02 weight decay with 0.1 factor
  - **Training**: 9 epochs, loss 0.0604 → 0.0534, stable gradients throughout
  - **Fixed-node Test (25k nodes)**: -4.17 ±15.89 Elo (30.34% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -2.52 ±8.57 Elo (28.18% LOS, 1928 games)

11. **Increased Hidden Size (352)**: Expanded from 320→352 hidden units
  - **Training**: 9 epochs, loss 0.0697 → 0.0528, stable training
  - **Performance Impact**: 20% slower (2.22M vs 2.78M nps)
  - **Fixed-node Test (25k nodes)**: +0.35 ±15.13 Elo (51.80% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -11.28 ±13.21 Elo (4.70% LOS, 678 games)
  - **Conclusion**: Performance cost outweighed capacity benefits

12. **Differential LR Drop Factors**: Feature layers 0.1 factor, Output layer 0.2 factor
  - **Training**: 9 epochs, loss 0.0592 → 0.0532, healthy gradients throughout
  - **Fixed-node Test (25k nodes)**: +3.82 ±14.94 Elo (69.21% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -21.74 ±18.72 Elo (1.12% LOS, 432 games)

13. **Conservative Output LR**: Output layer 0.0005 start, 0.2 drop factor
  - **Training**: 9 epochs, loss 0.0593 → 0.0533, stable throughout
  - **Fixed-node Test (25k nodes)**: +2.78 ±15.02 Elo (64.17% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -22.42 ±17.80 Elo (0.66% LOS, 388 games)

14. **Reduced Feature Weight Decay (0.005/0.02)**: Feature layers 0.005, Output layer 0.02
  - **Training**: 9 epochs, loss 0.0698 → 0.0536, stable gradients throughout
  - **Weight Analysis**: 36x avg bucket differentiation vs main's 1.73x, 8-9% zero weights vs main's 18%
  - **Fixed-node Test (25k nodes)**: -5.91 ±15.82 Elo (23.19% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -13.60 ±14.55 Elo (3.33% LOS, 588 games)

15. **Cosine Annealing LR Schedule (0.01/0.02)**: Feature layers 0.01, Output layer 0.02, cosine annealing 0.001→0.0001
  - **Training**: 9 epochs, loss 0.0699 → 0.0531, smooth gradients throughout, no optimization disruptions
  - **LR Schedule**: Smooth cosine decay vs previous harsh step drops, ended at min LR 0.0001
  - **Fixed-node Test (25k nodes)**: +10.77 ±14.96 Elo (92.12% LOS, 1000 games)
  - **LTC Test (40+0.4s)**: -13.41 ±15.15 Elo (4.12% LOS, 648 games)