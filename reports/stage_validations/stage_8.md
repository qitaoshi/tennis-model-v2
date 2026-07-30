# Stage 8 validation — isotonic recalibration

Maps fitted on FIT+TUNE predictions against outcomes and frozen to `data/processed/calibration_maps.pkl`. **This report is the one and only use of the pre-cutoff TEST set**; no map is fitted or refitted on it.


TEST window: 2023-07-03 .. 2023-11-27, 3,799 matches.


The first TEST window (2022-07-01 .. 2023-06-30) was spent by this stage's first run and is recorded burned in `constants.py`. This is its replacement, carved by moving the TUNE/TEST boundary forward. The holdout cutoff did not move.


Corrections in force: tiebreak inflation -0.0200, split sigma 0.080, provenance scheme `pooled`.


## Map eligibility, decided on FIT+TUNE only

A family gets a map only if one helps on a held-out slice of FIT+TUNE. Deciding this from TEST results would be selection on the test set.

| family            |   n_check |   ece_before |   ece_after |   brier_before |   brier_after | eligible   |
|:------------------|----------:|-------------:|------------:|---------------:|--------------:|:-----------|
| match_winner      |      7994 |   0.011737   |  0.0101888  |      0.222508  |     0.222482  | True       |
| set_score         |     16504 |   0.00980829 |  0.00841145 |      0.169726  |     0.169783  | False      |
| totals_under_high |      7994 |   0.0286947  |  0.0154361  |      0.210238  |     0.209669  | True       |
| totals_under_low  |      7994 |   0.00978259 |  0.0119452  |      0.0996429 |     0.0998841 | False      |
| totals_under_mid  |     11991 |   0.0160006  |  0.0150017  |      0.238155  |     0.238393  | False      |


## Calibration on the pre-cutoff TEST set

Families without a map pass through unchanged and are shown with identical before/after figures.

| family            |     n |   ece_before |   ece_after |   brier_before |   brier_after |   logloss_before |   logloss_after | improved   |
|:------------------|------:|-------------:|------------:|---------------:|--------------:|-----------------:|----------------:|:-----------|
| match_winner      |  7598 |    0.0182076 |   0.0169459 |       0.239806 |      0.240125 |         0.672186 |        0.672992 | False      |
| set_score         | 15688 |    0.0102167 |   0.0102167 |       0.175773 |      0.175773 |         0.532881 |        0.532881 | True       |
| totals_under_high |  7598 |    0.0349372 |   0.0109963 |       0.208946 |      0.207809 |         0.60852  |        0.605911 | True       |
| totals_under_low  |  7598 |    0.0216751 |   0.0216751 |       0.113492 |      0.113492 |         0.376066 |        0.376066 | True       |
| totals_under_mid  | 11397 |    0.0256417 |   0.0256417 |       0.239358 |      0.239358 |         0.671535 |        0.671535 | True       |


## Reliability after calibration


### match_winner

| bin | n | predicted | observed | gap |
|---|---|---|---|---|
| 0 | 540 | 0.2516 | 0.2926 | +0.0410 |
| 1 | 886 | 0.3654 | 0.3792 | +0.0139 |
| 2 | 647 | 0.4272 | 0.4529 | +0.0257 |
| 3 | 733 | 0.4638 | 0.4720 | +0.0082 |
| 4 | 710 | 0.4714 | 0.4817 | +0.0103 |
| 5 | 750 | 0.5030 | 0.4960 | -0.0070 |
| 6 | 526 | 0.5343 | 0.5304 | -0.0039 |
| 7 | 1,090 | 0.5427 | 0.5330 | -0.0097 |
| 8 | 927 | 0.6154 | 0.5912 | -0.0242 |
| 9 | 789 | 0.7200 | 0.6895 | -0.0305 |

### totals_under_high

| bin | n | predicted | observed | gap |
|---|---|---|---|---|
| 0 | 363 | 0.6493 | 0.6639 | +0.0146 |
| 1 | 961 | 0.6584 | 0.6576 | -0.0007 |
| 3 | 1,551 | 0.6714 | 0.6731 | +0.0017 |
| 4 | 756 | 0.6821 | 0.6561 | -0.0260 |
| 5 | 378 | 0.6974 | 0.7354 | +0.0381 |
| 6 | 643 | 0.7264 | 0.7387 | +0.0123 |
| 7 | 919 | 0.7441 | 0.7399 | -0.0042 |
| 8 | 525 | 0.7487 | 0.7124 | -0.0363 |
| 9 | 1,502 | 0.7530 | 0.7463 | -0.0066 |


## Gate

Every family improved on both ECE and Brier: **False**. Gate **FAILED**.

