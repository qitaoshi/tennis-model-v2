# Flat-stake ROI on total games

A **benchmark, not the objective**. The model can be the best available forecaster and still lose money, because the bookmaker's margin sits between being right and getting paid. This says how far the totals edge is from clearing that margin.


**Scope: TUNE only (2024-01-01 .. 2025-06-30), ATP main tour.** TEST and HOLDOUT are spent and were not read.


## Method

- Flat one unit per bet; no staking plan, which would amplify an edge that has to exist first.
- A bet is placed when the model's probability beats the probability implied by the **raw** decimal odds — the price actually on offer, vig included. Comparing against the de-vigged number instead counts bets as positive-edge when the real price is not, which is how paper edges get manufactured.
- Push returns the stake and stays in the denominator.
- Bootstrap resamples **whole matches**, because one match's ladder rungs are a correlated cluster and treating them as independent would shrink the interval to fiction.


## Results

| book                   |   min_edge |   n_bets |   n_matches |   roi_pct |   ci_lo_pct |   ci_hi_pct |   hit_rate |   mean_odds |
|:-----------------------|-----------:|---------:|------------:|----------:|------------:|------------:|-----------:|------------:|
| bet365                 |       0    |      966 |         729 | -5.00725  |   -11.5427  |     2.44664 |   0.517598 |     1.83898 |
| bet365                 |       0.02 |      783 |         581 | -3.05236  |   -11.205   |     4.64074 |   0.527458 |     1.84106 |
| bet365                 |       0.05 |      338 |         249 | -0.112426 |   -12.8959  |    12.033   |   0.544379 |     1.84101 |
| bet365                 |       0.1  |      158 |          99 |  3.79114  |   -15.498   |    22.5381  |   0.563291 |     1.84076 |
| ALL BOOKS (best price) |       0    |     9690 |        1090 | -3.40671  |    -8.72923 |     1.77941 |   0.511558 |     2.19686 |
| ALL BOOKS (best price) |       0.02 |     7195 |         995 | -2.91272  |    -9.59415 |     3.76769 |   0.507158 |     2.17067 |
| ALL BOOKS (best price) |       0.05 |     4032 |         720 | -0.666667 |    -8.97052 |     7.72035 |   0.503224 |     2.18553 |
| ALL BOOKS (best price) |       0.1  |     1475 |         310 |  7.26576  |    -6.80235 |    21.4184  |   0.518644 |     2.2375  |


**Any configuration whose confidence interval clears zero: False.**


## Reading this

`ALL BOOKS (best price)` assumes you always found the best quote of the seven books in the file. That is the most generous honest assumption available and still not achievable in practice — it ignores limits, closing lines and the accounts being restricted.


Raising the edge threshold trades sample for selectivity. If ROI does not improve as the threshold rises, the model's edge estimate carries no ordering information, which matters more than the headline number: it means the model cannot tell its good bets from its bad ones.


## Betting every match, no edge filter

One bet per match on the primary line, taking whichever side the model prefers, regardless of whether it beats the price. One bet per match, so these observations are independent — the ladder-correlation caveat above does not apply here.

`book_margin_pct` is the bookmaker's actual overround on that line, and `vs_margin_pp` is ROI plus margin: **how much better than blind betting the model's side selection is**. Betting blind at these prices loses the margin by construction, so zero on that column means no signal and positive means real information, even while losing money.

| book   |   n_matches |   roi_pct |   ci_lo_pct |   ci_hi_pct |   hit_rate |   mean_odds |   book_margin_pct |   vs_margin_pp |
|:-------|------------:|----------:|------------:|------------:|-----------:|------------:|------------------:|---------------:|
| bet365 |        1086 |  -9.27072 |   -14.9846  |    -3.82592 |   0.494475 |     1.83598 |           7.93343 |       -1.33729 |
| best   |        1095 |  -2.27123 |    -8.13527 |     3.70336 |   0.50137  |     1.95228 |           1.55861 |       -0.71262 |

