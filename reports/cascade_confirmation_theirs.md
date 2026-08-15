# Did the match_winner cascade help, end to end?

This scores the cascade shipped on `claude/tennis-calibration-validation-la21r5`
against the model as it stood before it, on that branch's code and that
branch's split. Both arms are scored identically; only the parameters differ.

**Incumbent:** Stage 3 half-life 1095 / no rank seed, K 48, level gap 100, and
every downstream stage as shipped before the cascade. Calibration maps
blended/8, covering all six families.

**Cascade:** Stage 3 K 32, level gap 0, cross-level offset -30.5, rank seed
scale 160, **half-life left at 1095**; Stage 4 blend 0.90, Stage 5 cohort k 10,
Stage 7 refit; calibration maps platt/5, covering only the three totals
families -- no map beat identity on match_winner, set_score or games_handicap,
so those three pass through uncalibrated.

TUNE (2024-01-01 .. 2024-12-31 on this branch's split), 6,000 sampled matches, same rows in both arms. Nothing fitted here; TEST and HOLDOUT untouched.


## Uncalibrated


| family | n | ECE before | ECE after | Brier before | Brier after | log-loss before | log-loss after |
|---|---:|---:|---:|---:|---:|---:|---:|
| games_handicap | 84,000 | 0.02208 | 0.01721 | 0.14739 | 0.14320 | 0.43563 | 0.42604 |
| match_winner | 12,000 | 0.02588 | 0.01992 | 0.23949 | 0.23126 | 0.67133 | 0.65446 |
| set_score | 24,672 | 0.01162 | 0.01089 | 0.17522 | 0.17238 | 0.53145 | 0.52455 |
| totals_under_high | 12,000 | 0.04121 | 0.03154 | 0.20631 | 0.20856 | 0.60294 | 0.60763 |
| totals_under_low | 12,000 | 0.01942 | 0.01021 | 0.11123 | 0.10087 | 0.37172 | 0.34159 |
| totals_under_mid | 18,000 | 0.02840 | 0.01479 | 0.23940 | 0.23744 | 0.67164 | 0.66747 |

## Calibrated


| family | n | ECE before | ECE after | Brier before | Brier after | log-loss before | log-loss after |
|---|---:|---:|---:|---:|---:|---:|---:|
| games_handicap | 84,000 | 0.03216 | 0.01721 | 0.14765 | 0.14320 | 0.42976 | 0.42604 |
| match_winner | 12,000 | 0.02412 | 0.01992 | 0.23965 | 0.23126 | 0.67170 | 0.65446 |
| set_score | 24,672 | 0.01195 | 0.01089 | 0.17525 | 0.17238 | 0.53151 | 0.52455 |
| totals_under_high | 12,000 | 0.01654 | 0.01095 | 0.20486 | 0.20757 | 0.59958 | 0.60537 |
| totals_under_low | 12,000 | 0.01428 | 0.00677 | 0.11104 | 0.10077 | 0.37058 | 0.34098 |
| totals_under_mid | 18,000 | 0.01662 | 0.00746 | 0.23885 | 0.23722 | 0.67050 | 0.66702 |


## match_winner, the family this was for


Calibrated ECE 0.02412 -> 0.01992 (+0.00420), Brier 0.23965 -> 0.23126 (+0.00839), log-loss 0.67170 -> 0.65446 (+0.01724).


## Caveat

TUNE-selected and TUNE-confirmed. Every stage in the cascade arm was selected
on this window and the incumbent arm was not, so the comparison is optimistic
for the cascade by construction. It is still the only head-to-head that exists:
the branch shipped the cascade without ever measuring it against the model it
replaced.

Both map sets saw 2024 during fitting, so the calibrated columns flatter both
arms. The uncalibrated columns are the cleaner read of the cascade itself, and
they show the same result.

Not comparable to `cascade_confirmation.md`: that run used a different split
(TUNE through 2025-06) and a different cascade, so its incumbent baseline
differs. Compare arms within a run, never across the two.

