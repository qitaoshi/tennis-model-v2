# The nocohort variant

The incumbent model with Stage 5 — the cohort prior for thinly-sampled
players — switched off, and nothing else changed. `fitted_params.json` here is
the repository's file with `stage_5.enabled` flipped to `false`; every other
stage keeps its shipped value.

## Why

`reports/layer_ablation.md` removed one cascade layer at a time on TUNE and
rescored. Stage 5 was the only layer whose removal produced an improvement the
paired bootstrap could distinguish from zero:

| arm | Brier vs shipped | 95% CI | distinguishable |
|---|---:|---|---|
| no Stage 5 cohort prior | -0.00033 | -0.00044 to -0.00021 | yes |
| Elo only (split from Elo) | -0.00035 | -0.00087 to +0.00018 | no |
| no Stage 6 venue | -0.00001 | -0.00007 to +0.00005 | no |
| no Stage 7 corrections | +0.00002 | -0.00003 to +0.00006 | no |

Total-games CRPS also improved, 3.4044 to 3.4001.

## Why it is a variant and not a change to the incumbent

Run 3 of the forward test started 2026-08-19 and runs fourteen days. Editing
`fitted_params.json` in place would move the primary variant's fingerprint and
`check_unchanged` would refuse to write another row — correctly, because days
1-6 would then measure a model that no longer exists. So this follows the
pattern the cascade variant already established: a third model priced against
the same fixtures and the same quotes, with its own ledger, leaving both
running experiments intact.

## The gap to be honest about

**The calibration map here is the incumbent's, copied unchanged.** It was
fitted on top of a pipeline that includes Stage 5, so it is slightly
mismatched to this arm. The ablation deliberately scored uncalibrated for
exactly this reason — a map is fitted to one arm and flatters it. Refitting
the map against this arm is a separate job and a separate tune-ledger entry;
until it happens, this variant's calibrated numbers carry that caveat and its
uncalibrated ones do not.

The effect size is small in absolute terms: 0.00033 of a Brier point against a
model that trails the bookmakers by 0.03. The case for this change is a
simpler pipeline with one less layer, not a materially better forecast.
