# Match-context (fatigue) adjustment

Rest days, seven-day workload and entry route, applied as a logit-space shift to each side's serve-point probability. Coefficients fitted on FIT, gate decided on TUNE. **TEST and HOLDOUT were not touched.**


## Why

Stages 2-6 describe ability. None of them describe the state a player walks on court in. These are the cheapest features that do, and they come from columns already in the vendor files that nothing read.


## Fitted coefficients (FIT)


| feature | coefficient |
|---|---:|
| rest_days (deficit) | -0.0116 |
| load7 | +0.0090 |
| is_qualifier | -0.0218 |


Positive means the condition costs serve points.


## Gate (TUNE)


| metric | baseline | adjusted | improved |
|---|---:|---:|---|
| serve-points-won MAE | 0.05964 | 0.05965 | False |
| match-winner log-loss | 0.66202 | 0.66225 | False |


n = 12,766 TUNE matches. Gate **FAILED** — both metrics must improve.


## Per-feature, applied alone

So that a null result names which feature was null rather than condemning the whole idea.

| feature      |   coefficient |   tune_spw_mae |   tune_logloss |     mae_gain |   logloss_gain |
|:-------------|--------------:|---------------:|---------------:|-------------:|---------------:|
| rest_days    |   -0.0116471  |      0.0596594 |       0.661655 | -1.72719e-05 |    0.000361078 |
| load7        |    0.00902247 |      0.0596697 |       0.662192 | -2.75558e-05 |   -0.000175871 |
| is_qualifier |   -0.0218065  |      0.0596366 |       0.66248  |  5.59283e-06 |   -0.000464599 |


## Caveat

TUNE is 2024-01-01 .. 2025-06-30, read once by the original backtest at aggregate level. Not a pristine selection set.

