# Calibration refit — TEST evaluation

Map: `platt`, fitting window `5`, selected on TUNE and refit on FIT+TUNE. **This is the one and only use of the post-re-split TEST window.** Nothing is fitted on it.


TEST: 2025-01-01 .. 2025-06-30, 4,316 matches.


## Before and after

| family            |     n |   ece_before |   ece_after |   brier_before |   brier_after |   logloss_before |   logloss_after |
|:------------------|------:|-------------:|------------:|---------------:|--------------:|-----------------:|----------------:|
| games_handicap    | 60424 |   0.0210604  |  0.0210604  |      0.142334  |     0.142334  |         0.424296 |        0.424296 |
| match_winner      |  8632 |   0.0191659  |  0.0191659  |      0.230174  |     0.230174  |         0.651683 |        0.651683 |
| set_score         | 17978 |   0.00782464 |  0.00782464 |      0.170595  |     0.170595  |         0.520046 |        0.520046 |
| totals_under_high |  8632 |   0.0328403  |  0.0115702  |      0.208012  |     0.206965  |         0.606436 |        0.60401  |
| totals_under_low  |  8632 |   0.00646325 |  0.0121351  |      0.0932486 |     0.0933823 |         0.318191 |        0.318894 |
| totals_under_mid  | 12948 |   0.0107642  |  0.00610557 |      0.236806  |     0.236663  |         0.666152 |        0.665853 |


Every family improved on ECE: **False**.


## Caveat

This TEST window was read once by the original backtest at aggregate level. No parameter was fitted against it then; it is not pristine.

