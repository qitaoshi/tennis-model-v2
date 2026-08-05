# De-vig method vs match_winner CLV — HOLDOUT

Measurement only: no parameter fitted or selected. Raw (uncalibrated) model
probability throughout. 6,448 matched rows. Mean two-way overround
1.0318 (3.18% margin).

`segment_edge_holdout.py` traced the pooled +2.89% CLV to market-underdog
picks (+38% mean CLV, -8.4% real ROI) — the favorite-longshot signature of
proportional de-vig. This recomputes the de-vigged market probability three
ways and re-measures. If the underdog CLV collapses under power/Shin, it was
a methodology artifact, not model skill.

| method | segment | n | mean edge | mean CLV | CLV 95% CI | ROI (n edge>0) |
|---|---|---:|---:|---:|---|---|
| proportional | all | 6,448 | -0.19% | +2.89% | +2.33% to +3.46% | -3.24% (CI -6.44% to -0.07%) (n=2,992) |
| proportional | model picks favorite | 5,531 | -2.58% | -2.94% | -3.24% to -2.65% | n/a (n=2,075) |
| proportional | model picks underdog | 917 | +14.18% | +38.06% | +35.65% to +40.66% | -8.43% (CI -15.93% to -0.83%) (n=917) |
| power | all | 6,448 | -0.95% | +2.19% | +1.58% to +2.83% | -3.18% (CI -6.68% to +0.29%) (n=2,749) |
| power | model picks favorite | 5,522 | -3.53% | -4.14% | -4.44% to -3.85% | n/a (n=1,823) |
| power | model picks underdog | 926 | +14.48% | +39.96% | +37.26% to +42.98% | -7.43% (CI -14.76% to +0.21%) (n=926) |
| shin | all | 6,448 | -0.68% | +2.43% | +1.85% to +3.04% | -3.40% (CI -6.75% to +0.07%) (n=2,819) |
| shin | model picks favorite | 5,502 | -3.24% | -3.81% | -4.10% to -3.51% | n/a (n=1,873) |
| shin | model picks underdog | 946 | +14.22% | +38.71% | +36.16% to +41.55% | -11.24% (CI -18.57% to -3.71%) (n=946) |

## Conclusion

The de-vig method is NOT the cause. Underdog mean CLV stays ~+38-40% under
all three methods, and the total two-way margin is only 3.18% — far too
small for margin allocation to manufacture a 38% swing. The favorite-longshot
de-vig hypothesis is rejected.

What the numbers show instead: the model genuinely rates market underdogs far
above their market price (+14% mean edge on ~920 underdog picks), and betting
those "value" underdogs loses money — Shin ROI -11.24% with a CI (-18.57% to
-3.71%) entirely below zero. The model systematically OVERRATES longshots;
the market is sharper on them. The positive pooled CLV is a match_winner
miscalibration on the underdog tail, not predictive skill.

match_winner has no demonstrated edge — not pooled, not in any level/surface
segment (`segment_edge_holdout.py`), not under any de-vig method here, and not
with vs without the Stage 8 calibration map (`calibration_ablation_clv.py`).
