# Stage 5 validation — cohort prior for thin-data players

Style vector: `ace_rate`, `df_rate`, `serve_pw`, `return_pw`, `clay_lean`, `hand_left`, `height`. Features and the means/stds used to standardize them are both computed as of the snapshot date (91-day snapshots), never once over the whole dataset.


Selected on TUNE thin-data matches: k = **5**, opponent-vs-cohort shrinkage **0**, thin threshold **1000** effective serve points, neighbour eligibility **3000**.


## TUNE, thin-data matches only

| prior | match-winner log-loss | total-games CRPS |
|---|---|---|
| **cohort (k=5)** | **0.63883** | **3.7113** |
| tour average | 0.63939 | 3.7191 |

n = 3,199 matches (3,199 with a scoreable games total).


## Grid

|   k |   opponent_vs_cohort_n0 |   logloss |    crps |    n |
|----:|------------------------:|----------:|--------:|-----:|
|   5 |                       0 |  0.638826 | 3.71127 | 3199 |
|  10 |                       0 |  0.638858 | 3.71147 | 3199 |
|  20 |                       0 |  0.638971 | 3.71299 | 3199 |
|  40 |                       0 |  0.639052 | 3.71347 | 3199 |
|  10 |                     200 |  0.640349 | 3.70255 | 3199 |
|   5 |                     200 |  0.640428 | 3.70251 | 3199 |
|  40 |                     200 |  0.640468 | 3.70459 | 3199 |
|  20 |                     200 |  0.640583 | 3.70413 | 3199 |


The opponent-vs-cohort adjustment — the well-sampled opponent's own record against the thin player's return-strength tier, shrunk toward their overall rate — is implemented and measured here. It consistently improves total-games CRPS and consistently costs match-winner log-loss, and selection is on log-loss, so it is switched off (`opponent_vs_cohort_n0 = 0`). That trade is visible in the grid above rather than buried.



## FIT-internal rolling origin (thin matches)

Log-loss gain over the tour-average prior, per fold: +0.00070, +0.00096, +0.00161, +0.00153, +0.00118


mean +0.00120, std 0.00038; TUNE gain +0.00056.


## Gate

Beats the tour-average prior on log-loss: **True** (0.63883 vs 0.63939). On total-games CRPS: **True** (3.7113 vs 3.7191). Gate **PASSED**.

