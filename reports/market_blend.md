# Blending the model with the market

One weight `w` on the model, applied in log-odds space, fitted by log-loss. `w = 1` is the model alone, `w = 0` is the market alone.


**Scope: TUNE only (2024-01-01 .. 2025-06-30), ATP main tour, 1,103 matches.** TEST and HOLDOUT were not read. `w` is a selected hyperparameter and is logged to `tune_ledger.json`.


**Fitted `w` = 0.0000** — the weight the data puts on the model, the remaining 1.0000 on the market. Free-form coefficients, fitted without the convexity constraint and without an intercept: model -0.2828, market 1.3191. The control — one scale on the market's log-odds with the model absent — fits 1.1618.


## Scores


| forecaster                                   |   brier |   logloss |     ece |
|:---------------------------------------------|--------:|----------:|--------:|
| market (median of books)                     | 0.18354 |   0.54380 | 0.02970 |
| model (calibrated)                           | 0.21498 |   0.61667 | 0.01011 |
| blend, w=0.000 (in-sample)                   | 0.18354 |   0.54380 | 0.02925 |
| blend (5-fold out-of-fold)                   | 0.18354 |   0.54380 | 0.02925 |
| free stack (in-sample)                       | 0.18226 |   0.54017 | 0.01529 |
| free stack (5-fold out-of-fold)              | 0.18285 |   0.54186 | 0.02116 |
| market scaled x1.162, no model (in-sample)   | 0.18311 |   0.54177 | 0.02258 |
| market scaled, no model (5-fold out-of-fold) | 0.18320 |   0.54209 | 0.02060 |


## Reading this


The in-sample blend rows cannot lose to their own inputs — the endpoints are inside the search space — so they are reported for completeness, not as evidence. The out-of-fold rows are the claim: the weight is fitted on four folds and scored on the fifth, five times over, folded on matches so that a match's two orientations never straddle a fold.


Against the market alone, out-of-fold: log-loss -0.00000, Brier -0.00000. Positive means the blend is better.


Per-fold `w`: 0.000, 0.000, 0.000, 0.000, 0.000. A weight that moves a lot between folds is a weight fitted on noise; a stable one is a real division of labour between the two forecasts.


## What this does not settle


Whether a projection that reads the bookmaker's price still counts as an accurate *match projection* under the 2026-08-08 goal. A blend with a low `w` is mostly the market repeated back, and it cannot price a match no bookmaker has quoted. That is a judgment call for the human. This report settles only how much accuracy is on the table.


The usual caveat on the odds source applies: these are OddsPortal 2024-2025 quotes, several books, de-vigged. The paper-trading path now reads PointsBet alone, which is one book and not a consensus.

