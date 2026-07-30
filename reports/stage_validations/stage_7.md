# Stage 7 validation — iid failure-mode corrections

Measured on TUNE, conditioned on the format rule in force at match time, excluding retirements, walkovers and `score_string_suspect` matches.


Selected provenance scheme: **pooled** (inferred weight 1.0). Tiebreak inflation **-0.0200**, split sigma **0.080**.


## (a) Tiebreak occurrence

| | observed | predicted | gap |
|---|---|---|---|
| before | 0.3445 | 0.4262 | -0.0817 |
| after | 0.3445 | 0.3451 | **-0.0006** |

### Inflation sweep (selected scheme, at the fitted split sigma)

| inflation | tiebreak gap |
|---|---|
| -0.0300 | +0.00384 |
| -0.0200 | -0.00058 |
| -0.0100 | -0.00511 |
| 0.0000 | -0.00982 |
| 0.0100 | -0.01468 |

## (b) Total games — PIT and coverage

| | PIT deviation from uniform | central-80% coverage |
|---|---|---|
| before | 0.02873 | 0.7325 |
| after | **0.00703** | **0.8070** |

### Split-sigma sweep (selected scheme)

| split sigma | PIT deviation | coverage80 |
|---|---|---|
| 0.000 | 0.02873 | 0.7325 |
| 0.040 | 0.02010 | 0.7608 |
| 0.060 | 0.01267 | 0.7840 |
| 0.080 | 0.00697 | 0.8075 |
| 0.100 | 0.01000 | 0.8287 |
| 0.120 | 0.01757 | 0.8423 |

## Provenance-weighting schemes compared (ground rule 4)

| scheme | inflation | sigma | tiebreak gap | PIT deviation |
|---|---|---|---|---|
| documented_only | -0.0300 | 0.080 | -0.02288 | 0.01623 |
| pooled **(selected)** | -0.0200 | 0.080 | -0.00058 | 0.00703 |
| downweight_inferred | -0.0300 | 0.080 | +0.00185 | 0.00710 |

## Separation check

Neither correction may degrade the other's calibration.


- tiebreak gap with the tiebreak correction alone: -0.07081; with both: -0.00058

- PIT deviation with the variance correction alone: 0.00697; with both: 0.00703


## FIT-internal rolling origin

Tiebreak-gap improvement per fold: +0.06051, +0.08105, +0.08086, +0.08172, +0.08108


PIT-deviation improvement per fold: +0.01990, +0.01961, +0.02352, +0.01619, +0.00858


## Gate

Tiebreak calibration acceptable (|gap| < 0.01 and improved): **True**. Games PIT/coverage acceptable (improved and coverage in (0.75, 0.85)): **True**. Neither correction degraded the other: **True**. Gate **PASSED**.

