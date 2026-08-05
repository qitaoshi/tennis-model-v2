# Calibration ablation — match_winner CLV/ROI, with vs without the Stage 8 map

Measurement only: no parameter fitted or selected. Same HOLDOUT odds join as
`clv_backtest.py` (6,448 matched rows), scored twice: once through
the shipped isotonic map (`model_p_calibrated`), once with the model's raw
Elo/serve-blend probability (`model_p_raw`). Bet side is fixed by the raw
model in both variants, so this isolates the calibration map's effect on the
probability estimate from its effect on bet selection.

## Why this was run

`reports/stage_validations/stage_8.md` shows the match_winner isotonic map
made Brier *worse* on its own pre-cutoff TEST window (0.23981 -> 0.24013,
`improved: False`); it passed only on ECE, and shipped riding on the human
accept decision for `totals_under_high`. HOLDOUT ECE for match_winner
(0.0693, `reports/stage_validations/backtest.md`) is 4x worse than the
0.0169 the map showed right after fitting, consistent with a map that
doesn't generalize past the window it was tuned on.

## Result

| variant | n | mean CLV | CLV 95% CI | positive CLV | mean edge | flat-stake ROI |
|---|---:|---:|---|---:|---:|---|
| calibrated (shipped) | 6,448 | +4.98% | +4.42% to +5.56% | 54.1% | +1.31% | -3.26% (CI -6.10% to -0.39%), n=3,488 |
| raw (uncalibrated) | 6,448 | +2.89% | +2.33% to +3.46% | 46.4% | -0.19% | -3.24% (CI -6.44% to -0.07%), n=2,992 |

## Reading this

If raw ROI's interval clears zero (or is materially less negative than
calibrated), the Stage 8 match_winner map is actively hurting, not helping,
and the fix is to drop it from `fitted_params.json["stage_8"]["families"]`
rather than fit a new one. If both are similarly negative, the map is not
the explanation and the CLV+/ROI- gap has some other cause (staking,
selection threshold, or a real absence of edge net of vig).
