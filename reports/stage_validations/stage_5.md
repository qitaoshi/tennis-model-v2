# Stage 5 validation — cohort prior for thin-data players

Style vector: `ace_rate`, `df_rate`, `serve_pw`, `return_pw`, `clay_lean`, `hand_left`, `height`. Features and the means/stds used to standardize them are both computed as of the snapshot date (91-day snapshots), never once over the whole dataset.


Selected on TUNE thin-data matches: k = **10**, opponent-vs-cohort shrinkage **0**, thin threshold **1000** effective serve points, neighbour eligibility **3000**.


## TUNE, thin-data matches only

| prior | match-winner log-loss | total-games CRPS |
|---|---|---|
| **cohort (k=10)** | **0.63874** | **3.6203** |
| tour average | 0.63908 | 3.6273 |

n = 2,771 matches (2,771 with a scoreable games total).


## Grid

|   k |   opponent_vs_cohort_n0 |   logloss |    crps |    n |
|----:|------------------------:|----------:|--------:|-----:|
|  10 |                       0 |  0.63874  | 3.62034 | 2771 |
|   5 |                       0 |  0.638848 | 3.62013 | 2771 |
|  40 |                       0 |  0.638876 | 3.62342 | 2771 |
|  20 |                       0 |  0.638982 | 3.62287 | 2771 |
|   5 |                     200 |  0.639066 | 3.60418 | 2771 |
|  40 |                     200 |  0.639091 | 3.60728 | 2771 |
|  10 |                     200 |  0.6391   | 3.60425 | 2771 |
|  20 |                     200 |  0.63916  | 3.60661 | 2771 |


The opponent-vs-cohort adjustment — the well-sampled opponent's own record against the thin player's return-strength tier, shrunk toward their overall rate — is implemented and measured here. It consistently improves total-games CRPS and consistently costs match-winner log-loss, and selection is on log-loss, so it is switched off (`opponent_vs_cohort_n0 = 0`). That trade is visible in the grid above rather than buried.



## FIT-internal rolling origin (thin matches)

Log-loss gain over the tour-average prior, per fold: +0.00004, +0.00052, +0.00055, +0.00005, +0.00010


mean +0.00025, std 0.00026; TUNE gain +0.00034.


## Gate

Beats the tour-average prior on log-loss: **True** (0.63874 vs 0.63908). On total-games CRPS: **True** (3.6203 vs 3.6273). Gate **PASSED**.

