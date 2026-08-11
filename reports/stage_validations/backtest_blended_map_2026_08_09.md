# Final backtest — HOLDOUT

> **STALE as of 2026-08-11 — describes a calibration map that no longer ships.**
> This run used the `blended`/8 maps (five families, no `games_handicap`). The
> shipped map since commit `4564a71` (2026-08-10) is `platt`/8 over six families,
> and it has **never been evaluated on TEST**
> (`fitted_params.json` → `calibration_refit.test_evaluated: false`).
> The numbers below are unaltered and remain valid for the map they were produced
> with; they do not describe current behaviour. See `PROGRESS.json` →
> `calibration_platt_2026_08_10_ships_unevaluated`.

Holdout window: 2026-01-01 onward. 5,735 scoreable matches (completed, in scope, clean score, usable serve stats).


## By market family

| family | n | Brier | log-loss | ECE |
|---|---|---|---|---|
| handicap | 45,880 | 0.1676 | 0.5037 | 0.0098 |
| match_winner | 11,470 | 0.2240 | 0.6377 | 0.0665 |
| set_betting | 23,404 | 0.1722 | 0.5231 | 0.0078 |
| tiebreak | 5,735 | 0.2224 | 0.6359 | 0.0236 |
| totals | 40,145 | 0.1902 | 0.5563 | 0.0078 |

## By tour level

| level | n | Brier | log-loss | ECE |
|---|---|---|---|---|
| ATP 250 | 9,328 | 0.1780 | 0.5277 | 0.0130 |
| ATP 500 | 7,414 | 0.1742 | 0.5184 | 0.0094 |
| Challenger | 94,468 | 0.1850 | 0.5469 | 0.0041 |
| Grand Slam | 5,568 | 0.1836 | 0.5427 | 0.0176 |
| Masters 1000 | 9,306 | 0.1769 | 0.5250 | 0.0124 |
| Tour (other) | 550 | 0.1919 | 0.5590 | 0.0618 |

## By format rule and provenance

| final-set rule | provenance | n | Brier | ECE |
|---|---|---|---|---|
| tiebreak | documented | 5,568 | 0.1836 | 0.0176 |
| tiebreak | inferred | 121,066 | 0.1832 | 0.0034 |

## Ablations

| configuration | Brier | log-loss | ECE | games CRPS |
|---|---|---|---|---|
| full | 0.1832 | 0.5421 | 0.0034 | 3.4413 |
| no_cohort | 0.1832 | 0.5422 | 0.0030 | 3.4419 |
| no_venue | 0.1833 | 0.5424 | 0.0035 | 3.4421 |
| no_corrections | 0.1933 | 0.5694 | 0.0363 | 3.6339 |
| no_calibration | 0.1832 | 0.5421 | 0.0054 | 3.4413 |
