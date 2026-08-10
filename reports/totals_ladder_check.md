# Is the backtest's totals ECE flattered by its line choices?

The same question had a bad answer for handicaps: `backtest.py` scores those at fixed margins where half the selections are nearly certain, giving ECE 0.0098 against 0.0624 on real market lines. Totals ECE 0.0078 is the headline 'this model is good' figure in the holdout report, so it is worth the same scrutiny.


Measured on TUNE. Nothing fitted; TEST and HOLDOUT untouched.


## Per rung of the ladder

`backtest.py` pools all seven of these into a single totals number. A rung whose base rate is near 0 or 1 is close to a free bet and contributes almost nothing to calibration error.

|   offset |    n |   mean_p |   base_rate |        ece |     brier |
|---------:|-----:|---------:|------------:|-----------:|----------:|
|     -6.5 | 6000 | 0.059157 |   0.0641667 | 0.00708602 | 0.0576626 |
|     -4.5 | 6000 | 0.175943 |   0.184833  | 0.0119142  | 0.148866  |
|     -2.5 | 6000 | 0.334609 |   0.350333  | 0.0161644  | 0.226428  |
|     -0.5 | 6000 | 0.466099 |   0.489     | 0.0267108  | 0.249937  |
|      1.5 | 6000 | 0.574268 |   0.6015    | 0.0276219  | 0.239909  |
|      3.5 | 6000 | 0.633816 |   0.665     | 0.0311842  | 0.223436  |
|      5.5 | 6000 | 0.704697 |   0.740167  | 0.03547    | 0.193296  |


Pooled ECE over the whole ladder: **0.02092**.


## What lines actually exist


Market total-games lines on TUNE: median 24.5, p5 18.5, p95 41.5. Typical model median total 22.0, so real lines sit roughly -3.5 to +19.5 games from it.


| scope | ECE | n |
|---|---:|---:|
| whole ladder (what the backtest reports) | 0.02092 | 42,000 |
| offsets inside the market's p5-p95 | 0.02650 | 30,000 |
| central rungs only, abs(offset) <= 2.5 | 0.02195 | 18,000 |


**Materially flattered by the tails: False.**

