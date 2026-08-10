# Unknown and stale players — the match_winner defect

Selected on TUNE at the Elo layer. **TEST and HOLDOUT were not touched**; both were already spent before this work began.


## The diagnosis this acts on

From `reports/match_winner_diagnosis.md`, Elo-layer ECE on all pre-holdout data:

| group | n | ECE |
|---|---|---|
| both players known | 110,809 | 0.0025 |
| match has a debutant | 2,948 | 0.0800 |
| layoff 70-140 days | 5,043 | 0.0314 |
| layoff 140+ days | 5,276 | 0.0550 |


The model picks winners very well between players it knows and is overconfident about players it does not. That is why no calibration map fixed match_winner, and why a faster-adapting ladder made it worse: K=48 is right for active players.


## Incumbent

half-life 1095, flat debut seed: grouped ECE 0.02412, pooled 0.01875, log-loss 0.64729, known 0.0214, debut 0.1268, stale 0.0311.


## The metric, and why it is not pooled ECE

The first run of this grid selected on pooled ECE and picked a setting whose pooled score (0.0062) was better than every one of its own subgroups (known 0.0108, debut 0.0933, stale 0.0471). That is one group's overprediction cancelling another's underprediction — arithmetic, not accuracy. The metric here is the size-weighted mean of the subgroup ECEs, which can only fall when a group genuinely improves. Pooled ECE is still reported, as a diagnostic.


## Grid, ranked by size-weighted group ECE

|   half_life |   seed_scale |   grouped_ece |   pooled_ece |   tune_logloss |   ece_known |   ece_debut |   ece_stale |
|------------:|-------------:|--------------:|-------------:|---------------:|------------:|------------:|------------:|
|         270 |          200 |     0.0112006 |   0.00738734 |       0.65006  |  0.00766095 |   0.0643031 |   0.0367786 |
|         270 |          240 |     0.011485  |   0.00705926 |       0.650463 |  0.00796696 |   0.0598614 |   0.0378212 |
|         180 |          240 |     0.0119692 |   0.0105907  |       0.653406 |  0.008429   |   0.0575088 |   0.0391243 |
|         270 |          160 |     0.0122256 |   0.00883466 |       0.649715 |  0.00833937 |   0.0708899 |   0.0402332 |
|         270 |          120 |     0.0125168 |   0.0110898  |       0.649842 |  0.00772843 |   0.102986  |   0.0432501 |
|         540 |          160 |     0.0135006 |   0.00821005 |       0.646541 |  0.0112474  |   0.0683043 |   0.025422  |
|         270 |            0 |     0.0139822 |   0.0104867  |       0.653759 |  0.00886103 |   0.120272  |   0.0448716 |
|         180 |          200 |     0.0140921 |   0.0119719  |       0.65307  |  0.0104066  |   0.0618151 |   0.0422959 |
|         270 |           80 |     0.0145064 |   0.0139725  |       0.650535 |  0.00858333 |   0.134041  |   0.0509381 |
|         270 |           40 |     0.0145221 |   0.013054   |       0.651829 |  0.00877548 |   0.120913  |   0.0518587 |
|         540 |           80 |     0.0146222 |   0.00771373 |       0.646782 |  0.0104443  |   0.10335   |   0.0394045 |
|         540 |          200 |     0.0146721 |   0.011643   |       0.64715  |  0.0130528  |   0.0723139 |   0.0194488 |


**Best: half-life 540, seed scale 160. Grouped-ECE gain +0.01062.** Known group not harmed: True. 10 of 35 settings satisfy the log-loss constraint. **PASSED**



### Selection rule

Maximise the fall in size-weighted group ECE **subject to** log-loss not worsening. The constraint filters the candidates before the argmax rather than vetoing the winner after it — applied the second way it rejects the entire grid whenever the unconstrained optimum happens to be a sharpness-destroying setting, and discards the settings that satisfy both. That is an implementation detail that changed the verdict on this grid, so it is written down.


## What this does and does not establish

The debut group has only 239 TUNE matches and its ECE swings 0.0327 to 0.1634 across the grid with no orderly response to the seed scale. **`rank_seed_scale` is not resolvable on this window.** Whatever value comes out of the grid is a report, not a finding, and it should not be shipped on this evidence alone.


The half-life effect is on firmer ground — the known group carries roughly 12,000 TUNE matches — but note what it is. Log-loss barely moves while ECE moves a lot, and shortening the half-life pulls every rating toward the reference between matches. That is global shrinkage: better calibration, not better discrimination. It is a real improvement to a real defect, and it is not the model learning anything new about who wins.


Note also that shortening the half-life makes the STALE group worse, not better, which is the opposite of the hypothesis this script was written to test.


## Caveat

There is no untouched window left to confirm this on. TEST went to the calibration refit and HOLDOUT to the 2026 backtest, both before this was scoped. This is a TUNE-selected result and nothing more until new season data accrues.

