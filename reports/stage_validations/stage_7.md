# Stage 7 validation — iid failure-mode corrections

Measured on TUNE, conditioned on the format rule in force at match time, excluding retirements, walkovers and `score_string_suspect` matches.


Selected provenance scheme: **pooled** (inferred weight 1.0). Tiebreak inflation **-0.0600**, level sigma **0.000**.


## (a) Tiebreak occurrence

| | observed | predicted | gap |
|---|---|---|---|
| before | 0.3445 | 0.4264 | -0.0819 |
| after | 0.3445 | 0.3957 | **-0.0512** |

### Inflation sweep (selected scheme)

| inflation | tiebreak gap |
|---|---|
| -0.0600 | -0.05125 |
| -0.0500 | -0.05592 |
| -0.0400 | -0.06077 |
| -0.0300 | -0.06580 |
| -0.0200 | -0.07101 |
| -0.0100 | -0.07639 |
| 0.0000 | -0.08195 |
| 0.0100 | -0.08766 |

## (b) Total games — PIT and coverage

| | PIT deviation from uniform | central-80% coverage |
|---|---|---|
| before | 0.02860 | 0.7332 |
| after | **0.02810** | **0.7312** |

### Sigma sweep (selected scheme, at the chosen inflation)

| level sigma | PIT deviation | coverage80 |
|---|---|---|
| 0.000 | 0.02810 | 0.7312 |
| 0.010 | 0.02810 | 0.7313 |
| 0.020 | 0.02813 | 0.7317 |
| 0.030 | 0.02820 | 0.7322 |
| 0.050 | 0.02870 | 0.7325 |
| 0.080 | 0.02997 | 0.7357 |

## Provenance-weighting schemes compared (ground rule 4)

| scheme | inflation | sigma | tiebreak gap | PIT deviation |
|---|---|---|---|---|
| documented_only | -0.0600 | 0.030 | -0.10764 | 0.04547 |
| pooled **(selected)** | -0.0600 | 0.000 | -0.05125 | 0.02810 |
| downweight_inferred | -0.0600 | 0.000 | -0.05529 | 0.02810 |

## Separation check

Neither correction may degrade the other's calibration.


- tiebreak gap with the tiebreak correction alone: -0.05125; with both: -0.05125

- PIT deviation with the variance correction alone: 0.02860; with both: 0.02810


## FIT-internal rolling origin

Tiebreak-gap improvement per fold: +0.03075, +0.03198, +0.03235, +0.03229, +0.03264


PIT-deviation improvement per fold: +0.00024, +0.00033, +0.00100, +0.00057, +0.00132


## Gate

Tiebreak calibration acceptable (|gap| < 0.01 and improved): **False**. Games PIT/coverage acceptable (improved and coverage in (0.75, 0.85)): **False**. Neither correction degraded the other: **True**. Gate **FAILED**.

