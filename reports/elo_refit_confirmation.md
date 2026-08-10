# Does the Elo refit survive the full pipeline?

`scripts/fit_unknown_players.py` selected inactivity half-life 540 and rank seed scale 160 at the Elo layer. Stage 4's blend weights were fitted against the OLD Elo and Stage 7's corrections against the old blend, so an Elo-layer gain can be diluted or undone downstream. This runs the frozen pipeline over TUNE both ways.


Probabilities are compared **uncalibrated** — the calibration map was fitted against the incumbent Elo, so applying it would measure the stale map as much as the change. Nothing is fitted here. TEST and HOLDOUT are not touched.


## match_winner on TUNE, through the full pipeline


| setting | n | ECE | Brier | log-loss |
|---|---:|---:|---:|---:|
| incumbent (hl 1095, seed 0) | 12,000 | 0.01815 | 0.23554 | 0.66286 |
| selected (hl 540, seed 160) | 12,000 | 0.02074 | 0.23561 | 0.66301 |


Delta: ECE -0.00259, log-loss -0.00016, Brier -0.00007.


**End-to-end gain survives: False.**


## If this ships

Stage 4's blend weights and Stage 7's corrections were both selected against the incumbent Elo and would need refitting, and the calibration map would need refitting after that. This confirmation measures the change with all of them held at their old values, so it is a LOWER bound on what a full refit would give — and also a reminder that shipping this is a cascade, not a one-line edit.


## Caveat

No untouched window remains to confirm this on: TEST went to the calibration refit and HOLDOUT to the 2026 backtest. TUNE-selected, TUNE-confirmed, and nothing stronger until new data accrues.

