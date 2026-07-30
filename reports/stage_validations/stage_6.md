# Stage 6 validation — venue court-speed index

Per-(venue, surface) serve-dominance multiplier relative to the same-surface tour average, applied to the LEVEL of (pa + pb) and never to the split. Keyed on the stable `tourney_code`, so sponsor and city renames do not split a venue's history and a surface change does not merge two different courts.


Selected on TUNE: shrinkage **20,000** serve points, indoor flag in the key **False**.


## TUNE total-games CRPS

| venue group | no venue index | with venue index |
|---|---|---|
| above-median history (>= 144 matches) | 3.8418 | **3.8404** |
| unmeasured (first edition) | 3.5880 | 3.5880 |
| all TUNE matches | 3.6815 | 3.6805 |

Match-winner log-loss is unchanged by construction (0.63446 -> 0.63451); the multiplier moves the level, and the level barely moves who wins.


## Provenance

99.2% of matches are at a venue with prior same-surface history. The rest receive exactly the surface average and are flagged `measured=False`, which price.py carries into its metadata (ground rule 7).


## Grid

|   shrink_n0 | use_indoor   |    crps |   crps_well_measured |   crps_unmeasured |   logloss |
|------------:|:-------------|--------:|---------------------:|------------------:|----------:|
|       20000 | False        | 3.6805  |              3.84037 |             3.588 |  0.634507 |
|       20000 | True         | 3.6805  |              3.84037 |             3.588 |  0.634507 |
|        5000 | False        | 3.68005 |              3.84045 |             3.588 |  0.634524 |
|        5000 | True         | 3.68005 |              3.84045 |             3.588 |  0.634524 |
|       50000 | False        | 3.68089 |              3.84079 |             3.588 |  0.634447 |
|       50000 | True         | 3.68089 |              3.84079 |             3.588 |  0.634447 |
|      150000 | False        | 3.68112 |              3.84109 |             3.588 |  0.634488 |
|      150000 | True         | 3.68112 |              3.84109 |             3.588 |  0.634488 |


## FIT-internal rolling origin (well-measured venues)

CRPS gain over the no-venue baseline, per fold: +0.00468, -0.00194, +0.00307, +0.00081, +0.00085


mean +0.00149, std 0.00252; TUNE gain +0.00144.


## Gate

Improves at venues with above-median history: **True**. No degradation at unmeasured venues: **True**. Gate **PASSED**.

