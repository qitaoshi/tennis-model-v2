# Stage 7 validation — iid failure-mode corrections

Measured on TUNE, conditioned on the format rule in force at match time, excluding retirements, walkovers and `score_string_suspect` matches.


Selected provenance scheme: **pooled** (inferred weight 1.0). Tiebreak inflation **-0.0200**, split sigma **0.080**.


## (a) Tiebreak occurrence

| | observed | predicted | gap |
|---|---|---|---|
| before | 0.3487 | 0.4295 | -0.0808 |
| after | 0.3487 | 0.3486 | **+0.0001** |

### Inflation sweep (selected scheme, at the fitted split sigma)

| inflation | tiebreak gap |
|---|---|
| -0.0300 | +0.00461 |
| -0.0200 | +0.00007 |
| -0.0100 | -0.00457 |
| 0.0000 | -0.00941 |
| 0.0100 | -0.01438 |

## (b) Total games — PIT and coverage

| | PIT deviation from uniform | central-80% coverage |
|---|---|---|
| before | 0.02837 | 0.7448 |
| after | **0.00873** | **0.8053** |

### Split-sigma sweep (selected scheme)

| split sigma | PIT deviation | coverage80 |
|---|---|---|
| 0.000 | 0.02837 | 0.7448 |
| 0.040 | 0.02057 | 0.7690 |
| 0.060 | 0.01380 | 0.7875 |
| 0.080 | 0.00860 | 0.8058 |
| 0.100 | 0.00963 | 0.8252 |
| 0.120 | 0.01890 | 0.8402 |

## Provenance-weighting schemes compared (ground rule 4)

| scheme | inflation | sigma | tiebreak gap | PIT deviation |
|---|---|---|---|---|
| documented_only | -0.0300 | 0.080 | -0.01160 | 0.01857 |
| pooled **(selected)** | -0.0200 | 0.080 | +0.00007 | 0.00873 |
| downweight_inferred | -0.0200 | 0.080 | -0.00085 | 0.00887 |

## Separation check

Neither correction may degrade the other's calibration.


- tiebreak gap with the tiebreak correction alone: -0.06966; with both: +0.00007

- PIT deviation with the variance correction alone: 0.00860; with both: 0.00873


## FIT-internal rolling origin

Tiebreak-gap improvement per fold: +0.08163, +0.08023, +0.06938, +0.05523, +0.06091


PIT-deviation improvement per fold: +0.02033, +0.02010, +0.01864, +0.01699, +0.02101


## Gate

Tiebreak calibration acceptable (|gap| < 0.01 and improved): **True**. Games PIT/coverage acceptable (improved and coverage in (0.75, 0.85)): **True**. Neither correction degraded the other: **True**. Gate **PASSED**.

