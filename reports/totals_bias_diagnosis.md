# Where the total-games over-prediction comes from

`totals_ladder_check.md` found the model expects more games than matches produce, at all seven rungs of the ladder with no exceptions. Total games is a function of the serve probabilities and of the engine that turns them into a distribution, and those need opposite fixes — so this measures both on the same matches.


TUNE only. Nothing fitted.


## The two candidates


| quantity | predicted | actual | bias |
|---|---:|---:|---:|
| total games | 24.416 | 23.988 | +0.428 |
| total games, Stage 7 off | 26.206 | 23.988 | +2.218 |
| serve points won | 0.62342 | 0.62200 | +0.00142 |


Shifting `pa`/`pb` down by the measured serve bias leaves a games bias of +0.546, against +0.579 before. If most of the games bias survives that shift, the serve rates are not the cause and the engine's iid point assumption is.


## By segment

Positive means the model predicts too high.

| cut     | value        |    n |   games_bias |     spw_bias |
|:--------|:-------------|-----:|-------------:|-------------:|
| level   | ATP 250      |  665 |   -0.0326868 | -0.00981818  |
| level   | ATP 500      |  343 |   -0.769024  | -0.0081248   |
| level   | Challenger   | 4054 |    0.663195  |  0.00450795  |
| level   | Grand Slam   |  381 |   -0.206235  | -0.00672254  |
| level   | Masters 1000 |  501 |    0.337703  |  0.00257153  |
| surface | Clay         | 2821 |    0.525011  |  0.00527597  |
| surface | Grass        |  378 |   -1.41358   | -0.0212547   |
| surface | Hard         | 2801 |    0.579691  |  0.000600219 |
| best_of | 3            | 5619 |    0.471438  |  0.00197397  |
| best_of | 5            |  381 |   -0.206235  | -0.00672254  |


## Verdict

**THE ENGINE / IID ASSUMPTION — fix in Stage 7.**

