# Short-horizon form — Elo-layer screen

Does blending a fast-adapting Elo ladder into the frozen one improve match-winner prediction? Scored at the Elo layer, where a grid point costs seconds. Everything downstream is a monotone function of this probability, so a blend that cannot win here cannot win there.


Fitted on all history up to the cutoff, **scored on TUNE only** (2024-01-01 .. 2025-06-30). TEST and HOLDOUT are both already spent and were not touched.


## Why

match_winner is the model's one badly calibrated family — ECE ~0.07 on two independent holdouts, against under 0.025 for every distribution family — and two different calibration maps failed to move it. A monotone map cannot fix it, so the defect is upstream. The rating being too slow is the leading candidate, and the fatigue experiment pointed the same way unprompted.


## Baseline

The current model is `weight = 0`: log-loss 0.64729, Brier 0.22821, ECE 0.01875.


## Grid, ranked by TUNE match-winner log-loss

|   k_fast |   weight |   logloss |    brier |       ece |
|---------:|---------:|----------:|---------:|----------:|
|        0 |     0    |  0.647293 | 0.228212 | 0.0187525 |
|       96 |     0.05 |  0.647599 | 0.22833  | 0.0201431 |
|      144 |     0.05 |  0.647785 | 0.228396 | 0.0205131 |
|      192 |     0.05 |  0.64792  | 0.22844  | 0.0192813 |
|       96 |     0.1  |  0.647962 | 0.228469 | 0.0220904 |
|      288 |     0.05 |  0.648162 | 0.228514 | 0.0197106 |
|       96 |     0.15 |  0.64838  | 0.22863  | 0.0223734 |
|      144 |     0.1  |  0.648463 | 0.228653 | 0.0216241 |
|       96 |     0.2  |  0.648853 | 0.228812 | 0.0228002 |
|      192 |     0.1  |  0.648918 | 0.228813 | 0.0200427 |
|      144 |     0.15 |  0.649324 | 0.228979 | 0.0215224 |
|      288 |     0.1  |  0.649926 | 0.229162 | 0.0237677 |
|       96 |     0.3  |  0.649962 | 0.229234 | 0.0225618 |
|      192 |     0.15 |  0.650276 | 0.229321 | 0.0245838 |
|      144 |     0.2  |  0.650363 | 0.22937  | 0.0236473 |


**Best: K_fast 0, weight 0.00, log-loss gain +0.00000.** Screen **FAILED**.

