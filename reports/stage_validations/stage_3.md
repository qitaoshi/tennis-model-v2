# Stage 3 validation — surface Elo

Selected on TUNE (2024-01-01 .. 2024-12-31): K **32**, Challenger K x**1.0**, surface blend weight **0.3**, inactivity half-life **1095.0**, new-player level gap **0**, debut rank-seed scale **160**, cross-level offset **-30.5**.


Selected under the 2026-08-11 rule (grouped subgroup ECE, subject to TUNE log-loss <= incumbent + 0.0001), NOT the TUNE log-loss of MODEL_PROMPT.md:373 — see LOGLOSS_SLACK in scripts/fit_stage3.py. 100 of 2352 settings satisfy the constraint. Incumbent: grouped ECE 0.01470, log-loss 0.64848. Selected: grouped ECE 0.00903, log-loss 0.64740. The log-loss argmax would have been grouped ECE 0.01232 at log-loss 0.64506.


## TUNE performance

| model | log-loss | Brier |
|---|---|---|
| **surface Elo** | **0.64738** | **0.22829** |
| rankings baseline (logistic in log-rank difference) | 0.65156 | 0.22991 |

n = 8,785 matches.


## Decile calibration (TUNE)

Each match contributes both orientations, so a bin's observed rate is a real win rate rather than 1 by construction.

|   bin |    n |   predicted |   observed |         gap |
|------:|-----:|------------:|-----------:|------------:|
|     0 | 1757 |    0.235034 |   0.249289 |  0.0142544  |
|     1 | 1757 |    0.338692 |   0.343768 |  0.00507609 |
|     2 | 1757 |    0.395684 |   0.383608 | -0.0120756  |
|     3 | 1757 |    0.440729 |   0.446784 |  0.00605488 |
|     4 | 1756 |    0.480058 |   0.499431 |  0.0193726  |
|     5 | 1758 |    0.519919 |   0.500569 | -0.0193506  |
|     6 | 1757 |    0.559271 |   0.553216 | -0.00605488 |
|     7 | 1757 |    0.604316 |   0.616392 |  0.0120756  |
|     8 | 1757 |    0.661308 |   0.656232 | -0.00507609 |
|     9 | 1757 |    0.764966 |   0.750711 | -0.0142544  |


Max |gap| 0.0194 (criterion < 0.05), mean |gap| 0.0114 (criterion < 0.025).


## Cross-level consistency check

Matches where one player's rating is >=80% Challenger-built and the opponent's is not, both with at least 10 rated matches.


- n = 2,015

- predicted Challenger-side win rate 0.4148

- observed 0.3891

- gap -0.0257

- fitted level_offset -30.5 rating points (gap -0.0425 was 3.8 SE); gap after -0.0257


## FIT-internal rolling origin

Log-loss gain over the ranking baseline, per fold: +0.01131, +0.01370, +0.00920, +0.01332, -0.00553


mean +0.00840, std 0.00799; TUNE gain +0.00418, envelope 0.01598.


## Subgroup calibration at the selected setting

The defect this grid was widened to address is a subgroup one: matches with a debutant, and matches after a long layoff. Pooled ECE is not shown as a selection number because it rewards one group's bias cancelling another's.


- debut ECE 0.0763

- stale (70d+ layoff) ECE 0.0304

- known ECE 0.0096

- size-weighted mean 0.0129


## Grid (top 10 by TUNE log-loss)

|   k |   surface_weight |   inactivity_half_life |   k_chall_mult |   level_gap |   rank_seed_scale |   tune_logloss |   tune_brier |   ece_debut |   ece_stale |   ece_known |   grouped_ece |
|----:|-----------------:|-----------------------:|---------------:|------------:|------------------:|---------------:|-------------:|------------:|------------:|------------:|--------------:|
|  32 |              0.3 |                   1095 |           1    |         100 |               120 |       0.645057 |     0.227224 |   0.0961126 |   0.0336115 |  0.00811249 |     0.0123188 |
|  32 |              0.3 |                   1095 |           1    |         100 |                80 |       0.645096 |     0.227232 |   0.132745  |   0.0390648 |  0.00965137 |     0.0150219 |
|  48 |              0.3 |                   1095 |           0.75 |         100 |               120 |       0.64567  |     0.227453 |   0.0831655 |   0.0245207 |  0.0144394  |     0.0169174 |
|  48 |              0.3 |                   1095 |           0.75 |         100 |                80 |       0.645793 |     0.227513 |   0.121112  |   0.0377321 |  0.0140616  |     0.0185754 |
|  32 |              0.3 |                   1095 |           1    |         100 |               160 |       0.645873 |     0.227598 |   0.0621767 |   0.0267604 |  0.00922284 |     0.011934  |
|  32 |              0.3 |                   1095 |           0.75 |         100 |                80 |       0.645922 |     0.22762  |   0.125742  |   0.0497322 |  0.00715669 |     0.0134991 |
|  32 |              0.5 |                   1095 |           1    |         100 |                80 |       0.645978 |     0.227617 |   0.134234  |   0.0408534 |  0.00731912 |     0.0131196 |
|  32 |              0.3 |                   1095 |           1    |         100 |                40 |       0.645999 |     0.227654 |   0.104498  |   0.0503658 |  0.00823405 |     0.014004  |
|  32 |              0.3 |                   1095 |           0.75 |         100 |               120 |       0.646075 |     0.227714 |   0.0975927 |   0.0315495 |  0.0107692  |     0.0145601 |
|  32 |              0.5 |                   1095 |           1    |         100 |               120 |       0.646122 |     0.227688 |   0.110944  |   0.0333069 |  0.0122949  |     0.0163887 |


## Gate

Calibration tracks observed win rates: **True**. Brier beats the rankings baseline: **True**. Gate **PASSED**.

