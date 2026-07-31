# Final backtest — HOLDOUT


Holdout window: 2024-01-01 onward. 22,634 scoreable matches (completed, in scope, clean score, usable serve stats).


## By market family

| family | n | Brier | log-loss | ECE |
|---|---|---|---|---|
| handicap | 181,072 | 0.1676 | 0.5037 | 0.0088 |
| match_winner | 45,268 | 0.2255 | 0.6413 | 0.0693 |
| set_betting | 92,914 | 0.1711 | 0.5207 | 0.0065 |
| tiebreak | 22,634 | 0.2182 | 0.6273 | 0.0091 |
| totals | 158,438 | 0.1885 | 0.5523 | 0.0081 |

## By tour level

| level | n | Brier | log-loss | ECE |
|---|---|---|---|---|
| ATP 250 | 48,224 | 0.1787 | 0.5298 | 0.0079 |
| ATP 500 | 27,126 | 0.1701 | 0.5073 | 0.0109 |
| Challenger | 352,946 | 0.1852 | 0.5477 | 0.0038 |
| Grand Slam | 28,536 | 0.1773 | 0.5272 | 0.0142 |
| Masters 1000 | 40,326 | 0.1745 | 0.5201 | 0.0058 |
| Tour (other) | 3,168 | 0.1800 | 0.5337 | 0.0256 |

## By format rule and provenance

| final-set rule | provenance | n | Brier | ECE |
|---|---|---|---|---|
| tiebreak | documented | 28,536 | 0.1773 | 0.0142 |
| tiebreak | inferred | 471,790 | 0.1827 | 0.0022 |

## Ablations

| configuration | Brier | log-loss | ECE | games CRPS |
|---|---|---|---|---|
| full | 0.1824 | 0.5403 | 0.0018 | 3.3800 |
| no_cohort | 0.1825 | 0.5405 | 0.0017 | 3.3809 |
| no_venue | 0.1824 | 0.5404 | 0.0021 | 3.3812 |
| no_corrections | 0.1929 | 0.5692 | 0.0403 | 3.5976 |
| no_calibration | 0.1824 | 0.5403 | 0.0054 | 3.3800 |
