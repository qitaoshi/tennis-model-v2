# Stage 4 validation — combining serve rates and Elo

Level (pa + pb) comes from the Stage 2 rate model; the split is `(1-w) * serve_gap + w * elo_gap`, with the Elo opinion converted to a serve gap by inverting the Stage 1 engine on a 26x81 (level, gap) grid.


Selected on TUNE: overall w **0.90** (w fitted on FIT alone would be 0.85), thin threshold **1000** effective serve points.


## Blend weight by sample-size bucket

| bucket | TUNE matches | w (TUNE) | w (FIT) |
|---|---|---|---|
| both well sampled | 5,679 | 0.85 | 0.85 |
| one thin | 2,188 | 0.95 | 0.85 |
| both thin | 583 | 0.90 | 0.90 |

## TUNE match-winner log-loss

| blend | log-loss |
|---|---|
| **fitted w per bucket** | **0.64288** |
| fitted w overall (0.90) | 0.64303 |
| w = 0 (serve rates only) | 0.68838 |
| w = 1 (Elo only) | 0.64341 |

n = 8,450 TUNE matches.


## Elo vs serve disagreement (metadata, not absorbed into the blend)

Percentage points of match win probability, all matches in the panel:


- mean +1.27pp, median +1.04pp

- absolute: mean 11.67pp, p50 9.11, p90 24.83, p99 47.74, max 88.43

- share above 10pp: 46.1%; above 20pp: 16.7%


This is carried per match into price.py's metadata alongside each side's effective n, for downstream stake and confidence decisions.


## FIT-internal rolling origin

Log-loss gain over w=1, per fold: +0.00027, +0.00176, +0.00111, +0.00216, +0.00141


mean +0.00134, std 0.00072; TUNE gain +0.00053, envelope 0.00143.


## Gate

Fitted w beats w=0 (0.64288 < 0.68838) and w=1 (0.64288 < 0.64341). Gate **PASSED**.

