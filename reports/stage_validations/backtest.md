# Final backtest — HOLDOUT


Calibration map in force: `platt`, 6 families, pickle sha256 `588cf1efeb66e0bc`. `calibration_refit.test_evaluated` in fitted_params.json is `False` — see PROGRESS.json → calibration_platt_2026_08_10_ships_unevaluated.


Holdout window: 2026-01-01 onward. 5,735 scoreable matches (completed, in scope, clean score, usable serve stats).


## How to read these numbers

**A pooled figure across market families is not a score of anything.**
The families have different base rates and different numbers of
selections per match — seven totals rungs and eight handicap lines
against one match winner — so a pooled Brier is a weighted average
whose weights are an artefact of how the ladder was enumerated, and it
moves when the ladder changes even if no forecast does. Every table
below is therefore reported per family. The one pooled row that
remains is marked and carries an interval.

**Intervals are cluster bootstraps over matches, not rows.** One match
supplies every selection in the row count, all driven by the same two
serve rates. Row-level resampling would understate the spread by
roughly the square root of the selections per match.

**`n` is selections; `matches` is the real sample size.**


## By market family

| family | n | matches | Brier | Brier 95% CI | log-loss | ECE | ECE 95% CI |
|---|---|---|---|---|---|---|---|
| handicap | 45,880 | 5,735 | 0.1676 | [0.1652, 0.1699] | 0.5037 | 0.0098 | [0.0081, 0.0150] |
| match_winner | 5,735 | 5,735 | 0.2238 | [0.2202, 0.2271] | 0.6374 | 0.0161 | [0.0145, 0.0329] |
| set_betting | 23,404 | 5,735 | 0.1721 | [0.1703, 0.1736] | 0.5229 | 0.0073 | [0.0054, 0.0138] |
| tiebreak | 5,735 | 5,735 | 0.2224 | [0.2185, 0.2265] | 0.6359 | 0.0236 | [0.0185, 0.0384] |
| totals | 40,145 | 5,735 | 0.1902 | [0.1883, 0.1920] | 0.5564 | 0.0080 | [0.0053, 0.0167] |

**Pooled across families (kept for continuity with earlier reports, and not a meaningful score):** Brier 0.1813 [0.1798, 0.1827], log-loss 0.5375, ECE 0.0037 [0.0030, 0.0069] on 120,899 selections from 5,735 matches.


Total-games CRPS: 3.4413 [3.3933, 3.4930] (one value per match, so no clustering issue).


## By market family and tour level

| family | level | n | matches | Brier | ECE |
|---|---|---|---|---|---|
| handicap | ATP 250 | 3,392 | 424 | 0.1522 | 0.0192 |
| handicap | ATP 500 | 2,696 | 337 | 0.1511 | 0.0121 |
| handicap | Challenger | 34,352 | 4,294 | 0.1711 | 0.0123 |
| handicap | Grand Slam | 1,856 | 232 | 0.1736 | 0.0280 |
| handicap | Masters 1000 | 3,384 | 423 | 0.1575 | 0.0140 |
| handicap | Tour (other) | 200 | 25 | 0.1676 | 0.0637 |
| match_winner | ATP 250 | 424 | 424 | 0.2287 | 0.0459 |
| match_winner | ATP 500 | 337 | 337 | 0.2152 | 0.0460 |
| match_winner | Challenger | 4,294 | 4,294 | 0.2263 | 0.0154 |
| match_winner | Grand Slam | 232 | 232 | 0.1974 | 0.0525 |
| match_winner | Masters 1000 | 423 | 423 | 0.2104 | 0.0614 |
| match_winner | Tour (other) | 25 | 25 | 0.3027 | 0.2293 |
| set_betting | ATP 250 | 1,696 | 424 | 0.1771 | 0.0259 |
| set_betting | ATP 500 | 1,348 | 337 | 0.1687 | 0.0221 |
| set_betting | Challenger | 17,176 | 4,294 | 0.1756 | 0.0082 |
| set_betting | Grand Slam | 1,392 | 232 | 0.1267 | 0.0248 |
| set_betting | Masters 1000 | 1,692 | 423 | 0.1691 | 0.0203 |
| set_betting | Tour (other) | 100 | 25 | 0.2066 | 0.1398 |
| tiebreak | ATP 250 | 424 | 424 | 0.2336 | 0.0592 |
| tiebreak | ATP 500 | 337 | 337 | 0.2445 | 0.0646 |
| tiebreak | Challenger | 4,294 | 4,294 | 0.2180 | 0.0269 |
| tiebreak | Grand Slam | 232 | 232 | 0.2424 | 0.0975 |
| tiebreak | Masters 1000 | 423 | 423 | 0.2266 | 0.0605 |
| tiebreak | Tour (other) | 25 | 25 | 0.2359 | 0.2418 |
| totals | ATP 250 | 2,968 | 424 | 0.1858 | 0.0265 |
| totals | ATP 500 | 2,359 | 337 | 0.1823 | 0.0132 |
| totals | Challenger | 30,058 | 4,294 | 0.1895 | 0.0050 |
| totals | Grand Slam | 1,624 | 232 | 0.2315 | 0.0401 |
| totals | Masters 1000 | 2,961 | 423 | 0.1870 | 0.0183 |
| totals | Tour (other) | 175 | 25 | 0.1741 | 0.0752 |

## By market family, format rule and provenance

| family | final-set rule | provenance | n | matches | Brier | ECE |
|---|---|---|---|---|---|---|
| handicap | tiebreak | documented | 1,856 | 232 | 0.1736 | 0.0280 |
| handicap | tiebreak | inferred | 44,024 | 5,503 | 0.1673 | 0.0104 |
| match_winner | tiebreak | documented | 232 | 232 | 0.1974 | 0.0525 |
| match_winner | tiebreak | inferred | 5,503 | 5,503 | 0.2249 | 0.0153 |
| set_betting | tiebreak | documented | 1,392 | 232 | 0.1267 | 0.0248 |
| set_betting | tiebreak | inferred | 22,012 | 5,503 | 0.1750 | 0.0059 |
| tiebreak | tiebreak | documented | 232 | 232 | 0.2424 | 0.0975 |
| tiebreak | tiebreak | inferred | 5,503 | 5,503 | 0.2215 | 0.0239 |
| totals | tiebreak | documented | 1,624 | 232 | 0.2315 | 0.0401 |
| totals | tiebreak | inferred | 38,521 | 5,503 | 0.1885 | 0.0069 |

## Ablations, by family

Brier / ECE per configuration. Compare down a column, never across families.

| configuration | handicap | match_winner | set_betting | tiebreak | totals | games CRPS |
|---|---|---|---|---|---|---|
| full | 0.1676 / 0.0098 | 0.2238 / 0.0161 | 0.1721 / 0.0073 | 0.2224 / 0.0236 | 0.1902 / 0.0080 | 3.4413 |
| no_cohort | 0.1677 / 0.0101 | 0.2239 / 0.0206 | 0.1721 / 0.0070 | 0.2223 / 0.0216 | 0.1903 / 0.0085 | 3.4419 |
| no_venue | 0.1676 / 0.0101 | 0.2239 / 0.0185 | 0.1721 / 0.0057 | 0.2226 / 0.0198 | 0.1905 / 0.0074 | 3.4421 |
| no_corrections | 0.1697 / 0.0322 | 0.2238 / 0.0162 | 0.1754 / 0.0227 | 0.2255 / 0.0608 | 0.2161 / 0.0970 | 3.6339 |
| no_calibration | 0.1676 / 0.0098 | 0.2239 / 0.0175 | 0.1721 / 0.0084 | 0.2224 / 0.0236 | 0.1902 / 0.0067 | 3.4413 |
