# Stage 4 validation — combining serve rates and Elo

Level (pa + pb) comes from the Stage 2 rate model; the split is `(1-w) * serve_gap + w * elo_gap`, with the Elo opinion converted to a serve gap by inverting the Stage 1 engine on a 26x81 (level, gap) grid.


Selected on TUNE: overall w **0.80** (w fitted on FIT alone would be 0.85), thin threshold **1000** effective serve points.


## Blend weight by sample-size bucket

| bucket | TUNE matches | w (TUNE) | w (FIT) |
|---|---|---|---|
| both well sampled | 7,423 | 0.80 | 0.80 |
| one thin | 2,594 | 0.85 | 0.80 |
| both thin | 605 | 0.90 | 0.95 |

## TUNE match-winner log-loss

| blend | log-loss |
|---|---|
| **fitted w per bucket** | **0.63470** |
| fitted w overall (0.80) | 0.63475 |
| w = 0 (serve rates only) | 0.67107 |
| w = 1 (Elo only) | 0.63655 |

n = 10,622 TUNE matches.


## Elo vs serve disagreement (metadata, not absorbed into the blend)

Percentage points of match win probability, all matches in the panel:


- mean +0.81pp, median +0.99pp

- absolute: mean 11.64pp, p50 9.43, p90 24.50, p99 43.24, max 76.16

- share above 10pp: 47.5%; above 20pp: 16.9%


This is carried per match into price.py's metadata alongside each side's effective n, for downstream stake and confidence decisions.


## FIT-internal rolling origin

Log-loss gain over w=1, per fold: +0.00122, +0.00165, +0.00360, +0.00277, +0.00292


mean +0.00243, std 0.00097; TUNE gain +0.00185, envelope 0.00195.


## Gate

Fitted w beats w=0 (0.63470 < 0.67107) and w=1 (0.63470 < 0.63655). Gate **PASSED**.

