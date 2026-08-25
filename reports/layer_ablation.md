# Layer ablation: which parts of the cascade earn their place

One layer removed at a time, everything else held at its shipped value, rescored on the same matches. A layer that contributes shows up as its removal making the model worse.


**Scope: TUNE only (2024-01-01 .. 2025-06-30), 12,766 matches, both tours.** TEST and HOLDOUT were not read. Nothing is fitted or selected — every arm reuses `fitted_params.json` unchanged — so there is no tune-ledger entry.


**Scores are uncalibrated.** The shipped calibration map was fitted on top of the shipped arm, so applying it would flatter that arm and penalise the others for a mismatch that is the map's, not theirs. ECE is consequently poor across the board and is shown for shape rather than for ranking.


## Results


| arm                       |     n |   brier |   logloss |     ece |   crps_games |     acc |   brier_vs_shipped |   logloss_vs_shipped |   crps_vs_shipped |   n_paired |   brier_delta |    ci_lo |    ci_hi | distinguishable   |
|:--------------------------|------:|--------:|----------:|--------:|-------------:|--------:|-------------------:|---------------------:|------------------:|-----------:|--------------:|---------:|---------:|:------------------|
| shipped (all layers)      | 12766 | 0.23515 |   0.66198 | 0.01461 |      3.40436 | 0.58570 |            0.00000 |              0.00000 |           0.00000 |      12766 |       0.00000 |  0.00000 |  0.00000 | False             |
| Elo only (split from Elo) | 12766 | 0.23480 |   0.66129 | 0.01244 |      3.40649 | 0.58366 |           -0.00035 |             -0.00069 |           0.00213 |      12766 |      -0.00035 | -0.00087 |  0.00018 | False             |
| serve rates only (no Elo) | 12766 | 0.26403 |   0.75778 | 0.11609 |      3.47220 | 0.55656 |            0.02888 |              0.09579 |           0.06784 |      12766 |       0.02888 |  0.02638 |  0.03150 | True              |
| no Stage 5 cohort prior   | 12766 | 0.23482 |   0.66131 | 0.01580 |      3.40009 | 0.58585 |           -0.00033 |             -0.00067 |          -0.00427 |      12766 |      -0.00033 | -0.00044 | -0.00021 | True              |
| no Stage 6 venue          | 12766 | 0.23514 |   0.66196 | 0.01367 |      3.40524 | 0.58687 |           -0.00001 |             -0.00002 |           0.00088 |      12766 |      -0.00001 | -0.00007 |  0.00005 | False             |
| no Stage 7 corrections    | 12766 | 0.23516 |   0.66202 | 0.01075 |      3.66075 | 0.58914 |            0.00002 |              0.00003 |           0.25639 |      12766 |       0.00002 | -0.00003 |  0.00006 | False             |


`*_vs_shipped` is the arm minus the shipped model. **Positive means removing that layer made things worse, i.e. the layer is earning its place.** Negative means the model is better without it.


## Verdict


These layers are measurably not earning their place — removing them improved Brier by more than the bootstrap interval:

* **no Stage 5 cohort prior** — Brier -0.00033 (95% CI -0.00044 to -0.00021)


These arms are indistinguishable from the shipped model — their confidence intervals straddle zero, so the honest statement is "no measurable difference", not "dead weight":

* **Elo only (split from Elo)** — Brier -0.00035 (95% CI -0.00087 to +0.00018)
* **no Stage 6 venue** — Brier -0.00001 (95% CI -0.00007 to +0.00005)
* **no Stage 7 corrections** — Brier +0.00002 (95% CI -0.00003 to +0.00006)


`distinguishable` is a paired bootstrap over matches, 2,000 resamples, both orientations of a match kept together. Pairing is what makes it usable at this scale: the arms share almost all their variance, so an unpaired interval would be far too wide to say anything.


## Reading the Elo-only and rates-only arms


The level — how serve-dominated a match is — always comes from the Stage 2 rate model, because Elo produces a win probability and no level at all. So *Elo only* means Elo owns the split while the rate model still sets the level; it is not a rate-model-free arm. That asymmetry is in the architecture, not in this experiment.


CRPS is the total-games score and depends mostly on the level, so it should barely move between the split arms. If it moves a lot, the split is leaking into the level and that is a bug worth chasing.

## What acting on this would require

Nothing here is a decision. Switching Stage 5 off is a change to the shipped
model, and it would be selected against TUNE — so it needs a `tune_ledger.json`
entry per ground rule 2, and a human sign-off, before it ships. This report is
the measurement that would justify asking, not the approval.

Two caveats belong next to the Stage 5 result. The effect is real but small:
0.00033 of a Brier point, on a model that sits 0.03 behind the bookmakers.
Deleting the layer buys a simpler pipeline far more than it buys accuracy. And
Stage 5 was originally selected on this same TUNE window, so TUNE has now been
asked about that layer twice, in opposite directions.
