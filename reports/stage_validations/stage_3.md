# Stage 3 validation — surface Elo

Selected on TUNE (2021-01-01 .. 2022-06-30): K **48**, Challenger K x**0.75**, surface blend weight **0.3**, inactivity half-life **1095.0**, new-player level gap **100**, cross-level offset **+0.0**.


## TUNE performance

| model | log-loss | Brier |
|---|---|---|
| **surface Elo** | **0.63977** | **0.22483** |
| rankings baseline (logistic in log-rank difference) | 0.65107 | 0.22993 |

n = 11,225 matches.


## Decile calibration (TUNE)

Each match contributes both orientations, so a bin's observed rate is a real win rate rather than 1 by construction.

|   bin |    n |   predicted |   observed |          gap |
|------:|-----:|------------:|-----------:|-------------:|
|     0 | 2245 |    0.214973 |   0.21559  |  0.000617598 |
|     1 | 2245 |    0.326922 |   0.343875 |  0.0169535   |
|     2 | 2245 |    0.389924 |   0.379955 | -0.00996837  |
|     3 | 2245 |    0.437815 |   0.443207 |  0.0053921   |
|     4 | 2244 |    0.480009 |   0.48975  |  0.00974187  |
|     5 | 2246 |    0.519974 |   0.51024  | -0.0097332   |
|     6 | 2245 |    0.562185 |   0.556793 | -0.0053921   |
|     7 | 2245 |    0.610076 |   0.620045 |  0.00996837  |
|     8 | 2245 |    0.673078 |   0.656125 | -0.0169535   |
|     9 | 2245 |    0.785027 |   0.78441  | -0.000617598 |


Max |gap| 0.0170 (criterion < 0.05), mean |gap| 0.0085 (criterion < 0.025).


## Cross-level consistency check

Matches where one player's rating is >=80% Challenger-built and the opponent's is not, both with at least 10 rated matches.


- n = 2,883

- predicted Challenger-side win rate 0.4243

- observed 0.4131

- gap -0.0112

- gap -0.0112 is 1.2 SE — within noise, no offset fitted


## FIT-internal rolling origin

Log-loss gain over the ranking baseline, per fold: +0.00452, +0.00682, +0.01107, +0.01037, +0.00884


mean +0.00833, std 0.00268; TUNE gain +0.01130, envelope 0.00536.


## Grid (top 10 by TUNE log-loss)

|   k |   surface_weight |   inactivity_half_life |   k_chall_mult |   level_gap |   tune_logloss |   tune_brier |
|----:|-----------------:|-----------------------:|---------------:|------------:|---------------:|-------------:|
|  48 |              0.3 |                   1095 |           0.75 |         100 |       0.639766 |     0.224828 |
|  32 |              0.3 |                   1095 |           1    |         100 |       0.639851 |     0.224841 |
|  48 |              0.3 |                   1095 |           1    |         100 |       0.640676 |     0.225166 |
|  48 |              0.5 |                   1095 |           0.75 |         100 |       0.640803 |     0.225296 |
|  32 |              0.3 |                   1095 |           0.75 |         100 |       0.640865 |     0.225301 |
|  32 |              0.5 |                   1095 |           1    |         100 |       0.641002 |     0.225379 |
|  32 |              0.3 |                    nan |           1    |           0 |       0.641508 |     0.225502 |
|  48 |              0.3 |                   1095 |           0.75 |           0 |       0.641679 |     0.225724 |
|  48 |              0.5 |                   1095 |           1    |         100 |       0.64179  |     0.225654 |
|  32 |              0.3 |                    nan |           0.75 |           0 |       0.641836 |     0.225746 |


## Gate

Calibration tracks observed win rates: **True**. Brier beats the rankings baseline: **True**. Gate **PASSED**.

