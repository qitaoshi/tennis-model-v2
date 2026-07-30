# Stage 2 validation — player serve rates

Selected on TUNE (2021-01-01 .. 2022-06-30): half-life **730d**, surface pooling **300** serve points, shrinkage n0 **500** serve points.


Opponent adjustment converged inside 50 sweeps at tolerance 1e-06; the tour-level average serve-win% per period is the identifiability anchor.


## Next-match serve points won (TUNE)

| model | MAE | per-point log-loss | match-winner log-loss |
|---|---|---|---|
| **full rate model** | **0.06595** | **0.66261** | **0.67106** |
| career_average | 0.06834 | 0.66332 | 0.71706 |
| last_10 | 0.07004 | 0.66429 | 0.74711 |

n = 21,244 player-matches.


## FIT-internal rolling origin (selected setting)

Gain over the career-average baseline, per-point log-loss, over 5 sequential FIT windows: +0.00111, +0.00107, +0.00098, +0.00102, +0.00124


mean +0.00108, fold-to-fold std 0.00010. TUNE gain +0.00071 (envelope at 2.0x std: 0.00020).


## Grid (top 10 by TUNE per-point log-loss)

|   half_life_days |   surface_pool |   shrink_n0 |       mae |   logloss |     n |
|-----------------:|---------------:|------------:|----------:|----------:|------:|
|              730 |            300 |         500 | 0.0659535 |  0.662608 | 21244 |
|              730 |            100 |         200 | 0.0659337 |  0.662615 | 21244 |
|              730 |            300 |         200 | 0.0658952 |  0.662617 | 21244 |
|              365 |            300 |         200 | 0.0658202 |  0.662625 | 21244 |
|              365 |            100 |         200 | 0.0658673 |  0.662628 | 21244 |
|              730 |           1000 |         500 | 0.0659531 |  0.662632 | 21244 |
|              365 |            300 |         500 | 0.0659473 |  0.662633 | 21244 |
|              730 |            100 |         500 | 0.0660564 |  0.662642 | 21244 |
|              365 |           1000 |         500 | 0.0659092 |  0.662646 | 21244 |
|              730 |              0 |         200 | 0.0660793 |  0.662667 | 21244 |


## Gate

MAE beats both baselines: **True**. Match-winner log-loss beats both baselines: **True**. Gate **PASSED**.

