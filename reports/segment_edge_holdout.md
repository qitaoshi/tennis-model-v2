# Segment edge check — match_winner, raw probability, HOLDOUT

Measurement only: no parameter fitted or selected. Uses the raw
(uncalibrated) model probability throughout, since the calibration ablation
(`reports/calibration_ablation_clv.md`) already showed the isotonic map
doesn't explain the CLV+/ROI- gap. 6,448 matched rows.

Overall: mean edge -0.19%, mean CLV +2.89%
(CI +2.33% to +3.46%), ROI -3.24% (CI -6.44% to -0.07%).

## By tour level

| segment | n | mean edge | mean CLV | CLV 95% CI | ROI (n edge>0) |
|---|---:|---:|---:|---|---|
| ATP 250 | 2,011 | +1.18% | +4.86% | +4.00% to +5.77% | -3.50% (CI -8.96% to +2.12%) (n=1,050) |
| Masters 1000 | 1,915 | -0.36% | +1.94% | +1.13% to +2.79% | -0.50% (CI -6.45% to +5.43%) (n=869) |
| ATP 500 | 1,262 | +0.27% | +3.45% | +2.32% to +4.59% | -5.60% (CI -12.36% to +1.12%) (n=625) |
| Grand Slam | 1,192 | -2.75% | +0.58% | -1.18% to +2.62% | n/a (n=408) |
| Tour (other) | 68 | +0.25% | +1.60% | -1.42% to +4.74% | n/a (n=40) |

## By surface

| segment | n | mean edge | mean CLV | CLV 95% CI | ROI (n edge>0) |
|---|---:|---:|---:|---|---|
| Hard | 3,631 | -0.21% | +2.37% | +1.76% to +2.99% | -2.38% (CI -6.55% to +1.80%) (n=1,690) |
| Clay | 1,986 | -0.38% | +2.40% | +1.55% to +3.30% | -3.17% (CI -8.90% to +2.55%) (n=919) |
| Grass | 831 | +0.35% | +6.33% | +3.85% to +9.12% | -7.23% (CI -16.78% to +2.37%) (n=383) |

## By market agreement

| segment | n | mean edge | mean CLV | CLV 95% CI | ROI (n edge>0) |
|---|---:|---:|---:|---|---|
| model picks market favorite | 5,531 | -2.58% | -2.94% | -3.24% to -2.65% | n/a (n=2,075) |
| model picks market underdog | 917 | +14.18% | +38.06% | +35.65% to +40.66% | -8.43% (CI -15.93% to -0.83%) (n=917) |

## Reading this

Any segment whose ROI interval clears zero has real, isolable edge even if
the pooled average doesn't. A segment with a positive mean edge but n_edge>0
too small for a ROI estimate ("n/a") is a candidate to revisit once more
holdout matches accrue, not a demonstrated edge yet.
