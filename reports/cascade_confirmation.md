# Did the match_winner cascade help, end to end?

`confirm_elo_refit.md` measured the new Elo with Stage 4, Stage 7 and the calibration map still fitted against the OLD Elo, and the Elo-layer gain did not survive. That was a lower bound by construction. This is the same measurement after all three were refitted.


Incumbent: Stage 3 half-life 1095 / seed scale 0, and every downstream stage as shipped before the cascade. Cascade: Stage 3 half-life 540 / seed scale 160, Stage 4 and Stage 7 refit against it, calibration map refit on top of those.


TUNE, 6,000 sampled matches, same rows in both arms. Nothing fitted here; TEST and HOLDOUT untouched.


## Uncalibrated


| family | n | ECE before | ECE after | Brier before | Brier after | log-loss before | log-loss after |
|---|---:|---:|---:|---:|---:|---:|---:|
| games_handicap | 84,000 | 0.02294 | 0.01842 | 0.14562 | 0.14168 | 0.43174 | 0.42260 |
| match_winner | 12,000 | 0.01815 | 0.02349 | 0.23554 | 0.22785 | 0.66286 | 0.64676 |
| set_score | 24,762 | 0.00982 | 0.00832 | 0.17375 | 0.17107 | 0.52763 | 0.52107 |
| totals_under_high | 12,000 | 0.03333 | 0.02408 | 0.20837 | 0.21052 | 0.60718 | 0.61172 |
| totals_under_low | 12,000 | 0.01079 | 0.00773 | 0.10326 | 0.09435 | 0.34860 | 0.32255 |
| totals_under_mid | 18,000 | 0.02195 | 0.01018 | 0.23876 | 0.23645 | 0.67027 | 0.66535 |

## Calibrated


| family | n | ECE before | ECE after | Brier before | Brier after | log-loss before | log-loss after |
|---|---:|---:|---:|---:|---:|---:|---:|
| games_handicap | 84,000 | 0.02932 | 0.02293 | 0.14560 | 0.14189 | 0.42524 | 0.41848 |
| match_winner | 12,000 | 0.01745 | 0.02426 | 0.23564 | 0.22801 | 0.66310 | 0.64720 |
| set_score | 24,762 | 0.01131 | 0.01047 | 0.17378 | 0.17114 | 0.52766 | 0.52126 |
| totals_under_high | 12,000 | 0.01063 | 0.01085 | 0.20732 | 0.20995 | 0.60475 | 0.61040 |
| totals_under_low | 12,000 | 0.00639 | 0.00863 | 0.10320 | 0.09435 | 0.34819 | 0.32257 |
| totals_under_mid | 18,000 | 0.01015 | 0.00590 | 0.23836 | 0.23633 | 0.66944 | 0.66511 |


## match_winner, the family this was for


Calibrated ECE 0.01745 -> 0.02426 (-0.00681), Brier 0.23564 -> 0.22801 (+0.00764), log-loss 0.66310 -> 0.64720 (+0.01590).


## Caveat

TUNE-selected and TUNE-confirmed. Stage 4, Stage 7 and the calibration map were all selected on this same window, so these numbers are optimistic for the cascade arm and not for the incumbent one. No untouched window remains to settle it: TEST went to the calibration refit and HOLDOUT to the 2026 backtest.

