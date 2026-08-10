# Calibration refit — TEST evaluation

Map: `blended`, fitting window `8`, selected on TUNE and refit on FIT+TUNE. **This is the one and only use of the post-re-split TEST window.** Nothing is fitted on it.


TEST: 2025-07-01 .. 2025-12-31, 4,133 matches.


## Before and after

| family            |     n |   ece_before |   ece_after |   brier_before |   brier_after |   logloss_before |   logloss_after |
|:------------------|------:|-------------:|------------:|---------------:|--------------:|-----------------:|----------------:|
| match_winner      |  8266 |    0.0217206 |   0.0180846 |       0.239461 |      0.239379 |         0.671358 |        0.671143 |
| set_score         | 16762 |    0.0121924 |   0.0100581 |       0.177251 |      0.177193 |         0.536165 |        0.535965 |
| totals_under_high |  8266 |    0.0419738 |   0.0213422 |       0.206342 |      0.204757 |         0.603091 |        0.599476 |
| totals_under_low  |  8266 |    0.0181279 |   0.0159892 |       0.106439 |      0.106293 |         0.357469 |        0.356708 |
| totals_under_mid  | 12399 |    0.0453978 |   0.0335632 |       0.241317 |      0.240432 |         0.675692 |        0.673894 |


Every family improved on ECE: **True**.


## Caveat

This TEST window was read once by the original backtest at aggregate level. No parameter was fitted against it then; it is not pristine.

