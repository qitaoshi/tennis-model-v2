# Is point-level momentum big enough to change the engine?

`model/engine.py` is exact DP over score states with **constant**
per-point win probabilities. It is Markov in the scoreboard but has no
memory: what happened on the previous point cannot change the next one.
Stage 7's `split_sigma` mixes over (pa, pb), which makes points
exchangeable but not sequentially dependent — it buys static
heterogeneity, not momentum.

The model over-predicts total games by +0.428 on TUNE (mean 24.0).
Stage 7's corrections already remove 80% of a +2.218 raw bias and
serve-rate bias explains ~6% of what is left. The iid point assumption is
the remaining suspect. This measures the sequential part and prices it in
games. Nothing is fitted and no evaluation window is consumed.

## What is and is not being separated

Psychological momentum and genuine score-state tactical play (a server
plays 40-0 differently from 0-40) are **not** decomposed here, and the
data could not decompose them anyway. That is deliberate: both are
history dependence the engine cannot express, both would need the same
fix — augmenting the state — and both are therefore in scope together.
Read every number below as *total history dependence*, an upper bound on
the psychological part.

## Data

Match Charting Project, men's singles, `charting-m-points-2010s.csv` and
`charting-m-points-2020s.csv`, retrieved 2026-08-09 from the `master`
branch. CC BY-NC-SA 4.0 — attribution, non-commercial, share-alike. Raw
CSVs land in `data/raw/mcp/` and are gitignored; `data/raw/mcp/NOTICE`
records the provenance.

The source named in the original task, `JeffSackmann/tennis_slam_pointbypoint`,
returns HTTP 404 — that account is down to a single public repository. MCP
is the surviving Sackmann point-by-point source and is the better one for
this question regardless, because it contains best-of-three matches and the
slam repo did not.

After dropping matches on or after `HOLDOUT_CUTOFF` (2026-01-01):
**5,384 matches, 876,428 points**, 2010-01-08 to 2025-12-21.
`C.assert_no_holdout` is called on the surviving dates, so the boundary is
a runtime error rather than a comment.

A tiebreak game is identified as the one game both players serve in, not
by a 6-6 test. The two rules agree on 99.977% of games; the exceptions are
NextGen Finals short sets, where the serve-alternation rule is the correct
one. Tiebreaks are excluded from every statistic.

At the point level a further **0.8%** of units are
dropped because replaying the scoring rule over their point sequence does
not reproduce their observed game count — a retirement or a charting gap
leaves an unfinishable game, which desynchronises the parse for every game
after it. Since the null re-derives boundaries by replay (below), the
observed statistic has to be computed on the same parse or the two are not
comparable.

## Method

### Confound 1 — between-player-match heterogeneity

A player who gets broken is on average the weaker player in that match, so
pooling across matches manufactures a positive correlation that is really
skill heterogeneity. Without this control the answer is guaranteed positive
and meaningless. Every statistic is therefore computed **within** a
(match, server) unit — a player-match fixed effect — and only then
aggregated across units.

### Confound 2 — Miller & Sanjurjo small-sample bias

Within a finite sequence whose win count is fixed, the naive
P(win | prev win) − P(win | prev loss) is biased **downward**. The bias is
exactly −1/(n−1) and does not depend on the win count; this script verifies
that by enumerating every arrangement for n=10. At the game level a player
serves about ten games, so the bias is about −11pp — far larger than the
effect being hunted, and in the direction that would hide it. The null is
therefore obtained by permuting within each unit, never assumed to be zero.

### Why the point-level null replays the scoring rule

At the point level the null must re-derive game boundaries from each
permuted sequence rather than reuse the observed ones. A four-point game is
`WWWW` or `LLLL` by construction, so shuffling inside observed game lengths
would manufacture exactly the correlation being measured. Permutation is on
the player's whole service-point sequence, then the scoring rule (first to
four, win by two) is replayed to cut it into games.

### Aggregation, standard error, p-value

Units are combined with the Mantel-Haenszel risk difference,
`sum(a_u) / sum(b_u)` with `a_u = (n11·N0 − n01·N1)/T` and `b_u = N1·N0/T`.
A unit with no post-win or no post-loss point contributes zero to both and
drops out on its own — there is no inclusion rule that could depend on the
observed outcome, which is how the heterogeneity confound would otherwise
get back in. The standard error is a sandwich clustered on **player**, since
the same player appears in many charted matches. The p-value is the
permutation p-value, exact under within-unit exchangeability.

`raw` below is the uncorrected within-unit difference, `null` the
permutation mean, `delta` the debiased effect `raw − null`.

## Result — point level

P(win point on serve | won the previous point of the same service game)
minus P(win | lost it).

| cut       |   units |     raw |    null |   delta |     se |     lo |     hi |      p |
|:----------|--------:|--------:|--------:|--------:|-------:|-------:|-------:|-------:|
| all       |   10685 | -0.0049 | -0.0101 |  0.0053 | 0.0013 | 0.0028 | 0.0078 | 0.0099 |
| best-of-3 |    8219 | -0.0059 | -0.0117 |  0.0058 | 0.0016 | 0.0027 | 0.0090 | 0.0099 |
| best-of-5 |    2464 | -0.0027 | -0.0073 |  0.0045 | 0.0022 | 0.0002 | 0.0089 | 0.0495 |
| hard      |    6796 | -0.0061 | -0.0104 |  0.0043 | 0.0017 | 0.0009 | 0.0077 | 0.0099 |
| clay      |    2837 | -0.0035 | -0.0104 |  0.0070 | 0.0023 | 0.0025 | 0.0115 | 0.0099 |
| grass     |    1050 | -0.0012 | -0.0090 |  0.0078 | 0.0036 | 0.0007 | 0.0149 | 0.0495 |
| 2010s     |    4430 | -0.0024 | -0.0103 |  0.0079 | 0.0023 | 0.0034 | 0.0123 | 0.0099 |
| 2020s     |    6255 | -0.0065 | -0.0100 |  0.0035 | 0.0016 | 0.0004 | 0.0066 | 0.0297 |

Note the sign of `raw`. Uncorrected, the within-unit difference is
**negative** in every cut — winning the previous point appears to *hurt*.
That is the Miller & Sanjurjo artifact, not a finding, and it is what a
player-match fixed effect on its own would have reported. The debiased
effect is positive, small and stable across surface and format: serving
players win about half a point in a hundred more often after winning the
previous point of the same game.

## Result — game level

P(hold | held the previous service game) minus P(hold | was broken).

| cut       |   units |     raw |    null |   delta |     se |      lo |     hi |      p |
|:----------|--------:|--------:|--------:|--------:|-------:|--------:|-------:|-------:|
| all       |   10735 | -0.0797 | -0.0842 |  0.0045 | 0.0031 | -0.0016 | 0.0106 | 0.1329 |
| best-of-3 |    8237 | -0.0974 | -0.0985 |  0.0010 | 0.0036 | -0.0060 | 0.0081 | 0.7812 |
| best-of-5 |    2496 | -0.0468 | -0.0584 |  0.0116 | 0.0054 |  0.0009 | 0.0222 | 0.0230 |
| hard      |    6841 | -0.0837 | -0.0855 |  0.0017 | 0.0039 | -0.0059 | 0.0093 | 0.6693 |
| clay      |    2841 | -0.0782 | -0.0861 |  0.0078 | 0.0063 | -0.0045 | 0.0202 | 0.1718 |
| grass     |    1051 | -0.0582 | -0.0710 |  0.0127 | 0.0098 | -0.0066 | 0.0320 | 0.1638 |
| 2010s     |    4452 | -0.0879 | -0.0864 | -0.0015 | 0.0044 | -0.0101 | 0.0071 | 0.7253 |
| 2020s     |    6283 | -0.0741 | -0.0827 |  0.0087 | 0.0044 |  0.0001 | 0.0172 | 0.0280 |

The bias dominates completely here. A player serves only ~10-12 games, so
the null sits near -8 to -10pp and the raw numbers are almost exactly the
bias with nothing left over. On **best-of-three — the format the model
actually prices — the debiased effect is indistinguishable from zero**.

Best-of-five does show a real effect. That divergence is worth recording:
had this been run on the Grand Slam point-by-point data the task originally
specified, it would have found game-level momentum and generalised it to a
challenger holdout that is 94% best-of-three, where it is not there. Having
best-of-three matches in the sample is what makes the answer usable, and it
is the reason the substituted source is the better one.

The era splits point in opposite directions at the two levels (point-level
momentum falls 2010s to 2020s, game-level rises), so they are read as noise
rather than drift.

## Sizing in engine units

Both effects are re-expressed so the **marginal** serve-win rate is
unchanged: `P(W|prev W) = p + d(1−p)`, `P(W|prev L) = p − d·p`, which is the
unique split leaving the two-state chain stationary at `p`. Without that
constraint the exercise would measure a level shift, not dependence.

The point-level effect resets at each game start, so service games stay
independent and the effect reaches total games only through P(hold). It is
therefore priced exactly: an exact DP over (points, points, previous
outcome) gives the Markov hold rate, and the iid point rate reproducing that
hold rate is fed to the engine's closed form. The game-level effect does not
collapse that way and is simulated at game granularity.

Sized on the **best-of-3** row of the tables above, because the model's
holdout is 94% challengers and those are all best-of-three.

| spec           | rates         | scenario                   |   mean_games |   d_mean |   sd_games |   d_sd |   pct_of_bias |
|:---------------|:--------------|:---------------------------|-------------:|---------:|-----------:|-------:|--------------:|
| best-of-3 tour | 0.640 / 0.640 | iid baseline (exact)       |       25.683 |    0.000 |      5.892 |  0.000 |         0.000 |
| best-of-3 tour | 0.640 / 0.640 | point momentum, estimate   |       25.672 |   -0.011 |      5.891 | -0.002 |        -2.654 |
| best-of-3 tour | 0.640 / 0.640 | point momentum, 95% CI top |       25.665 |   -0.017 |      5.890 | -0.003 |        -4.085 |
| best-of-3 tour | 0.640 / 0.640 | game momentum, estimate    |       25.680 |   -0.003 |      5.893 |  0.000 |        -0.752 |
| best-of-3 tour | 0.640 / 0.640 | game momentum, 95% CI top  |       25.644 |   -0.039 |      5.894 |  0.002 |        -9.206 |
| best-of-3 tour | 0.640 / 0.640 | both, 95% CI top           |       25.630 |   -0.053 |      5.894 |  0.002 |       -12.283 |
| best-of-3 tour | 0.660 / 0.620 | iid baseline (exact)       |       25.112 |    0.000 |      5.980 |  0.000 |         0.000 |
| best-of-3 tour | 0.660 / 0.620 | point momentum, estimate   |       25.104 |   -0.008 |      5.978 | -0.002 |        -1.829 |
| best-of-3 tour | 0.660 / 0.620 | point momentum, 95% CI top |       25.100 |   -0.012 |      5.977 | -0.003 |        -2.815 |
| best-of-3 tour | 0.660 / 0.620 | game momentum, estimate    |       25.107 |   -0.005 |      5.978 | -0.002 |        -1.093 |
| best-of-3 tour | 0.660 / 0.620 | game momentum, 95% CI top  |       25.084 |   -0.028 |      5.983 |  0.002 |        -6.582 |
| best-of-3 tour | 0.660 / 0.620 | both, 95% CI top           |       25.074 |   -0.038 |      5.978 | -0.002 |        -8.934 |
| best-of-5 slam | 0.640 / 0.640 | iid baseline (exact)       |       42.377 |    0.000 |      8.833 |  0.000 |         0.000 |
| best-of-5 slam | 0.640 / 0.640 | point momentum, estimate   |       42.358 |   -0.019 |      8.830 | -0.003 |        -4.379 |
| best-of-5 slam | 0.640 / 0.640 | point momentum, 95% CI top |       42.348 |   -0.029 |      8.828 | -0.004 |        -6.741 |
| best-of-5 slam | 0.640 / 0.640 | game momentum, estimate    |       42.373 |   -0.004 |      8.832 | -0.001 |        -0.903 |
| best-of-5 slam | 0.640 / 0.640 | game momentum, 95% CI top  |       42.319 |   -0.058 |      8.841 |  0.008 |       -13.541 |
| best-of-5 slam | 0.640 / 0.640 | both, 95% CI top           |       42.298 |   -0.079 |      8.837 |  0.004 |       -18.485 |
| best-of-5 slam | 0.660 / 0.620 | iid baseline (exact)       |       41.017 |    0.000 |      9.069 |  0.000 |         0.000 |
| best-of-5 slam | 0.660 / 0.620 | point momentum, estimate   |       41.006 |   -0.010 |      9.065 | -0.004 |        -2.392 |
| best-of-5 slam | 0.660 / 0.620 | point momentum, 95% CI top |       41.001 |   -0.016 |      9.063 | -0.006 |        -3.682 |
| best-of-5 slam | 0.660 / 0.620 | game momentum, estimate    |       41.008 |   -0.008 |      9.066 | -0.003 |        -1.932 |
| best-of-5 slam | 0.660 / 0.620 | game momentum, 95% CI top  |       40.969 |   -0.048 |      9.067 | -0.002 |       -11.165 |
| best-of-5 slam | 0.660 / 0.620 | both, 95% CI top           |       40.965 |   -0.051 |      9.063 | -0.006 |       -12.024 |

### Reading that table

**Direction.** The combined effect is -0.053 games — it shortens matches, which is the direction that would help, since streakier holds make sets more lopsided and therefore shorter, and the
model over-predicts. So this is a real contribution to the residual rather
than a contradiction of it. It is simply 12% of the 0.428 games that need explaining, at the *top* of the
confidence interval, with both effects stacked.

**Noise floor.** The game-momentum and combined rows are simulated at 2M
matches, so their MC standard error is about 0.004 games. Point-momentum
rows are exact. Any entry under ~0.01 games should be read as zero.

**Why the point-level effect is worth so little.** A 0.7pp shift in the
point win rate conditional on the previous point changes P(hold) by only a
fraction of that — the dependence partly cancels within the game, because a
server who is more likely to follow a won point with another is equally more
likely to follow a lost point with another. Marginal-preserving dependence
moves the tails of the game-length distribution, not its centre, and total
games is driven by the centre.

## Checks

- E_null[D] = -1/(n-1) exactly, verified by enumerating all C(10,k) arrangements for k=2..8
- p_hold_markov(p, 0) == engine.p_hold(p) to 1e-12 for p in 0.55/0.60/0.64/0.70
- sim_games at d=0 (bo3): 25.6985 vs engine 25.6829, within 1.7 MC SE
- sim_games at d=0 (bo5): 42.3936 vs engine 42.3768, within 1.2 MC SE
- marginal hold rate under d=0.05: 0.8129 vs 0.8126 (momentum reallocates, does not shift)

## What this does not establish

- **MCP is not a random sample.** It is crowdsourced and skews to notable
  matches and top players. The model's holdout is 94% challengers. Slam and
  tour-level tennis is what is measured here; challenger tennis is what it
  is being extrapolated to. The best-of-3 cut narrows that gap but does not
  close it — those are still main-tour best-of-three matches.
- **Conditioning on unit length conditions on a stopping time.** How many
  service games a player gets depends on how the match went, so
  exchangeability given (n, k) is an approximation, not an identity.
- **Retirements are only handled at the point level**, via the replay-parse
  filter. At the game level a truncated final service game still enters as a
  break.
- Momentum and score-state effects are measured jointly, per the section
  above.
- The measured game-level effect may already contain point-level carry-over
  across the changeover, so the combined row is an upper bound, not a sum
  of disjoint effects.

**Verdict: SMALL. Even at the top of the 95% CI the combined effect moves expected total games by 0.053 on a best-of-three, against the 0.428 game residual it would have to explain — 12% of it, with both effects stacked. Keep the mixture; retune split_sigma to 0.09 as already scoped; close this line of work.**

