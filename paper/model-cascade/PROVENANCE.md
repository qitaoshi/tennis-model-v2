# The cascade variant

These two files are the model shipped on
`claude/tennis-calibration-validation-la21r5`, copied verbatim from that
branch. They are here so the forward paper run can price the same fixtures
under both models on the same day. Nothing here is fitted, and nothing here
is read by any evaluation script — only `scripts/paper_trade.py --variant
cascade`.

What differs from the incumbent:

| stage | incumbent | cascade |
|---|---|---|
| 3 Elo | K 48, level gap 100, no rank seed | K 32, level gap 0, cross-level offset -30.5, rank seed 160 |
| 4 blend | w 0.80 / 0.80 / 0.85 / 0.90 | w 0.90 / 0.85 / 0.95 / 0.90 |
| 5 cohort | k 5 | k 10 |
| 7 corrections | refit against the old Elo | refit against the new one |
| calibration | blended / 8yr, all six families | platt / 5yr, totals only |

Both keep `inactivity_half_life` at 1095. A variant that also shortened it to
540 was measured and discarded — it regressed match_winner ECE, because
shortening the half-life is global shrinkage rather than anything the model
learned. See `reports/cascade_confirmation.md` (that variant) and
`reports/cascade_confirmation_theirs.md` (this one).

The cascade has never been evaluated out of sample. Its TEST read on that
branch compared map against no-map, not cascade against incumbent, and three
of the six families carry no map at all so their columns are identical there.
This forward run is the first out-of-sample read it will get.
