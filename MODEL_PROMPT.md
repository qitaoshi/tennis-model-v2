# PROMPT: Build a Tennis Prop Pricing Model from Scratch (v4)

You are building a complete tennis match and props pricing model in Python.
The repo already contains raw match data from TML-Database (schema inspired
by, but not identical to, Jeff Sackmann's tennis_atp — verify actual column
names before writing ingestion code; do not guess; TML deviates in places
such as retirement flags and some ID fields, and these are common silent-bug
sources). The data is licensed CC BY-NC-SA: non-commercial use, attribution
required. (Note for the human, not the code: if the eventual use is
comparing fair prices against market lines for wagering at scale or a
commercial product, the license question needs separate thought — the model
cannot resolve that.)

The model takes two players and outputs fair probabilities and fair decimal
prices for every standard tennis market: match winner, set betting (exact
set scores), total games (full ladder of lines), game handicaps, tiebreak
occurrence, and per-player games. It computes probabilities in closed form
wherever possible; Monte Carlo is only used for quantities the closed form
cannot express.

SCOPE: this is a PRE-MATCH pricer only. Live/in-play pricing (conditional
on current score state) is a different engine and is explicitly out of
scope. Fatigue/scheduling features (days since last match, travel,
back-to-back deciders) are also an explicit non-goal for this build —
they have documented predictive value and may be added later, but nothing
here should silently assume they exist.

## Non-negotiable ground rules

1. FOUR-WAY DATA SPLIT, DEFINED FIRST. Before writing any model code,
   define in constants.py:
   - FIT set: earliest data up to a fit-boundary date. All parameter
     fitting happens here.
   - TUNE set: a window after the fit boundary. All stage-level
     hyperparameter selection (half-life, n0, K, w, k, venue shrinkage,
     correction magnitudes) is evaluated here. This set WILL be reused
     across Stages 2–7.
   - PRE-CUTOFF TEST set: a window after the tune boundary and before
     the final holdout cutoff. Touched exactly once, by Stage 8's
     calibration check. Never used for any selection decision.
   - HOLDOUT: everything after the final cutoff (e.g. 2024-01-01+).
     Touched only by the final backtest, after all components are
     frozen. **THE HOLDOUT CUTOFF DATE, ONCE SET IN constants.py, IS
     PERMANENTLY FIXED.** It never moves, including after a Stage 8
     failure. The only boundary that can move during iteration is the
     tune/test boundary (see ground rule 3 and Stage 8's failure
     protocol below) — never confuse "carve a replacement TEST window"
     with "push the holdout cutoff back to buy more room." If that
     distinction is ever ambiguous in a PR or a report, treat it as a
     blocking issue, not a judgment call.

2. TUNE-SET REUSE IS SEQUENTIAL, NOT INDEPENDENT — TRACE IT, AND KEEP
   THE TRACE READABLE. Stages 2–7 are fit in order, and each later
   stage is selected against TUNE using a pipeline whose earlier stages
   were ALSO already selected against that same TUNE window. No single
   stage looks like it's overfitting in isolation, but the composition
   can still overfit TUNE as a whole — a purely sequential greedy-fit
   risk that the four-way split contains (the PRE-CUTOFF TEST gate will
   catch that it happened) but does not by itself make diagnosable
   (it won't tell you which stage caused it). To make it diagnosable:
   - Every stage's validation script appends a record to
     `reports/tune_ledger.json`: stage name, `run_id` (timestamp or git
     commit hash), the metric(s) used for selection, the TUNE-set
     value, the FIT-internal rolling-origin value for the same metric
     (i.e. the stability of the gain *within* FIT, before TUNE is ever
     consulted), AND a `frozen: true/false` flag.
   - **FROZEN FLAG.** A stage's ledger entry is marked `frozen: true`
     only when that stage has passed its validation gate AND is not
     going to be re-fit before the next downstream stage consumes it.
     During normal development a stage will be rebuilt and re-tuned
     many times (`frozen: false` entries accumulate) — that history is
     kept for the developer's own reference but is NOT what gets
     diagnosed later. The ledger reader used in the Stage 8 failure
     protocol filters to `frozen: true` entries only, i.e. the actual
     lineage currently feeding `price.py`, not the full dev history.
     Setting `frozen: true` is a one-way action per stage-version: if a
     "frozen" stage is later changed, its old entry is superseded by a
     new one and the old one stays in the ledger for audit purposes but
     is excluded from the active lineage view.
   - **OVERFIT-SIGNAL THRESHOLD, DEFINED NOW, NOT AT DIAGNOSIS TIME.**
     "TUNE gain is large relative to FIT-internal gain" is not eyeballed
     after the fact — it is computed the same way for every stage: use
     the fold-to-fold spread (e.g. standard deviation) of the
     FIT-internal rolling-origin metric across its folds as the noise
     envelope for that stage, and flag the stage if its realized
     TUNE-set gain exceeds that envelope by a stated multiple (e.g.
     TUNE gain > 2× the FIT-internal fold-to-fold std of the gain).
     This multiple is itself a constant, set once in `constants.py`,
     not re-decided per incident. The point is that two people reading
     the ledger after a Stage 8 failure must reach the same conclusion
     about which stage is implicated — a qualitative "looks large" read
     is exactly the kind of post-hoc storytelling this mechanism exists
     to prevent.

3. STAGE GATES. Build in the stage order below. Each stage's validation
   is evaluated on the TUNE set (except Stage 8, which uses the
   PRE-CUTOFF TEST set) and logged to the ledger per rule 2. Do not
   build a stage on top of a previous stage that has not passed its
   validation.

4. NO MAGIC CONSTANTS, NO UNRESOLVED "CONSIDER... OR" DECISIONS. Any
   constant that affects predictions must be either derived from theory
   or selected on FIT+TUNE data, with the selection code kept in the
   repo and the result written to fitted_params.json — never hardcoded
   in module bodies. This applies equally to modeling *choices*, not
   just numeric constants: if a stage's spec below poses a design
   decision as an open consideration (e.g. how to weight `documented`
   vs `inferred` provenance in Stage 7), that decision must be resolved
   by the same FIT/TUNE selection discipline as any other hyperparameter
   before the stage is marked passed — it does not ship as a runtime
   "consider doing X or Y" ambiguity.

5. AS-OF-DATE DISCIPLINE EVERYWHERE, INCLUDING DERIVED STATISTICS USED
   FOR STANDARDIZATION. Every quantity computed from match history
   (serve rates, Elo, cohort style vectors, venue multipliers) must be
   computable "as of" a date using only matches strictly before that
   date. This is restated per-stage below, because lookahead
   reintroduced through a secondary code path (e.g. cohort features
   built from full-career stats, OR summary statistics like feature
   means/stds computed once over the full dataset and then reused to
   standardize every player regardless of pricing date) is the most
   common way this discipline silently fails. As-of-date discipline
   covers not just the raw per-player features but any aggregate
   statistic derived from the broader population and used to transform
   them (e.g. z-scoring against a tour-wide mean/std).

6. RETIREMENT/WALKOVER/DEFAULT HANDLING IS EXPLICIT, NOT IMPLIED.
   Stage 0 detects them; every downstream stage states what it does
   with them. Defaults: EXCLUDED from serve-rate fitting (Stage 2 —
   stats are contaminated by whatever caused the retirement); EXCLUDED
   from total-games and tiebreak-frequency measurement (Stages 1 and 7
   — truncated distributions would otherwise be fitted as genuine iid
   deviation); INCLUDED in Elo updates (Stage 3) as win/loss only if
   TML marks them as completed-enough results — decide and document in
   Stage 0. Walkovers never update anything.

7. MEASURED VS. INFERRED PROVENANCE IS TRACKED, NOT JUST FOR VENUES.
   Any derived fact whose confidence varies by data tier — venue
   multipliers (Stage 6, already specified) AND format rules
   (rules.py, see below) — carries an explicit provenance flag through
   to price.py's output metadata. A confident-looking number built on
   a shaky inference must not look identical, downstream, to one built
   on a citable source.

8. SCORE-STRING SUSPECT FLAG PROPAGATES TO EVERY CONSUMER, NOT JUST THE
   MODULE THAT FOUND IT. A garbled score string (typo, walkover
   artifact, retirement recorded mid-tiebreak) can corrupt more than
   the module that first notices the anomaly. Stage 0 sets a single
   `score_string_suspect` boolean per match (populated in part from
   `rules.py`'s discrepancy scan, and from any other Stage 0 parsing
   failures), and it lives on the canonical match record — not only in
   `reports/rules_discrepancies.csv`. Every downstream stage that
   consumes set scores or game-level detail from that row (Stage 1
   validation targets, Stage 2 serve-rate computation, Stage 7's
   tiebreak-frequency and variance measurement) checks this flag and
   excludes or separately reports on flagged matches, rather than each
   stage independently rediscovering (or failing to discover) the same
   data quality problem.

9. DETERMINISM. Same inputs → same outputs. Seed all Monte Carlo. The
   closed-form engine must be exactly reproducible.

10. EVERY MODULE GETS A __main__ demo printing a small worked example.

## Repository layout to produce

```
model/
  constants.py        # split boundaries, seeds, paths, overfit-signal
                       # multiple (ground rule 2) — nothing fitted
  rules.py            # format/tiebreak rules lookup by tournament+year,
                       # with provenance
  data_audit.py       # Stage 0 — also sets score_string_suspect flag
  engine.py           # Stage 1 — closed-form scoring math
  player_rates.py     # Stage 2 — serve/return rates from data
  elo.py              # Stage 3 — surface Elo
  combine.py          # Stage 4 — rates + ratings blend
  cohort.py           # Stage 5 — similar-player prior for thin data
  venue.py            # Stage 6 — court speed index
  corrections.py      # Stage 7 — iid failure-mode corrections
  recalibrate.py      # Stage 8 — isotonic calibration
  simulate.py         # Monte Carlo for non-closed-form quantities
  price.py            # top-level: two player IDs in → all fair prices out
tests/
  test_engine.py
  test_rules.py
  test_rates.py
  ...
reports/
  stage_validations/   # output of each stage's validation script
  tune_ledger.json      # append-only across runs; frozen/unfrozen entries
                         # per ground rule 2
  rules_discrepancies.csv
fitted_params.json
```

## rules.py — format rules are data, not a config flag, and carry provenance

Deciding-set rules vary BY TOURNAMENT AND YEAR within the dataset and
must be a lookup table, not a boolean:
- Australian Open: 10-point deciding-set tiebreak from 2019.
- Wimbledon: final-set tiebreak at 12-12 from 2019; 10-point at 6-6
  from 2022. Before 2019: advantage final set.
- Roland Garros: advantage final set until 2022; 10-point TB from 2022.
- US Open: final-set tiebreak throughout the modern era (7-point until
  2022, 10-point from 2022).
- Non-major tour and Challenger events vary by era; build the table
  from documented rule histories where citable.

PROVENANCE FIELD, REQUIRED PER (tournament, year) ENTRY: `documented`
(citable rule-history source, e.g. the four majors above) or `inferred`
(reconstructed from TML score-string patterns — e.g. a 12-10 or 70-68
final set implies advantage rules were in force). Grand Slam entries
are `documented`; a large share of Challenger/futures/ITF-level entries
will necessarily be `inferred`, since that's exactly the tier without a
citable rule-history source. This is not a defect to fix — it's a fact
to carry forward. `rules_for(tournament_id, year)` returns the format
spec and the provenance, consumed by engine.py and by every stage that
measures against historical outcomes. Stage 7 in particular MUST
condition on the rule in force at match time, or the measured "engine
understates tiebreaks" gap will be partly a rule-mismatch artifact
rather than a modeling error.

Validation gate (tests/test_rules.py):
- Spot-check against known matches (Isner–Mahut 70-68 at Wimbledon 2010
  must be legal under its assigned rules; a 2023 Wimbledon 6-6 final
  set must resolve by tiebreak).
- Empirical scan across TML: for every completed match, check the
  actual score string against the format assigned by rules.py.
  DO NOT HARD-FAIL ON MISMATCH. Tennis score strings are commonly
  messy — typos, walkover artifacts, retirements recorded mid-tiebreak
  — and a hard assertion can't distinguish "the rules table is wrong"
  from "this one row has a data-entry error." Log every mismatch to
  `reports/rules_discrepancies.csv` with match ID, assigned format, and
  actual score string, AND set that match's `score_string_suspect` flag
  (ground rule 8) on the canonical match record so every downstream
  consumer sees it, not just this scan. Set a manual-review threshold
  (e.g. if discrepancy rate for a given tournament exceeds a few
  percent, flag that tournament's table entries for manual re-check
  before proceeding); isolated one-off mismatches are treated as
  probable data-entry errors, flagged suspect, and excluded from
  scoring wherever the suspect flag is checked — not treated as
  evidence the table itself is wrong.

## Stage 0 — Data audit (data_audit.py)

Load the TML data. Produce a written data dictionary in reports/:
- Exact column names and dtypes per file.
- Row counts by year and tour level (tour / challenger / futures).
- Which per-match stats exist (serve points won, first serves in, aces,
  double faults, break points) and their coverage % by year and level.
- How retirements, walkovers, and defaults are flagged — and the
  documented decision (ground rule 6) on how each downstream stage
  treats each category.
- Surface labels used and their counts. Venue/tournament identifier
  fields and whether they are stable across years.
- SURFACE-CHANGE FLAGS: explicitly list tournaments whose surface
  changes across years in the dataset (common at Challenger level), and
  city/sponsor renames affecting identifier stability. venue.py keys on
  (venue, surface) — this audit is what makes that keying trustworthy.
- SCORE-STRING SUSPECT FLAG: populate `score_string_suspect` on the
  canonical match record (ground rule 8) from any parsing failure or
  anomaly found here, in addition to what rules.py's discrepancy scan
  contributes later. This is a shared field, not owned by any one
  downstream module.
- Duplicate detection: same date + same two players + same tournament.

Validation gate: the data dictionary exists, ingestion of every year
runs without error, coverage numbers are printed not assumed, and the
surface-change and retirement-handling sections are complete.

## Stage 1 — Core engine (engine.py) — NO DATA DEPENDENCY

Pure closed-form tennis scoring math (Barnett & Clarke style recursion).
Inputs: pa, pb = each player's probability of winning a point on their
own serve, plus a format spec from rules.py. Assume iid points
(corrections come in Stage 7).

Implement, all closed-form, no simulation:
- p_hold(p): P(server wins a game at point-win prob p), via the
  standard deuce recursion: from deuce, win the game with probability
  p² / (p² + (1−p)²).
- p_tiebreak(pa, pb, first_to): 7-point and 10-point tiebreaks, win by
  2, accounting for serve rotation; handle the level-score continuation
  with the two-point-block recursion.
- p_set(pa, pb, format): P(A wins a set) and the full set-score
  distribution (6-0 … 7-6 and mirrors) by dynamic programming over game
  states, with tiebreak — or advantage continuation — at 6-6 per the
  format spec. Advantage sets need the geometric continuation from 6-6
  handled in closed form (two-game blocks).
- p_match(pa, pb, format): best-of-3 and best-of-5, with the deciding-
  set rule (advantage / 7-pt TB / 10-pt TB / match tiebreak) taken from
  the format spec, not a global flag.
- total_games_distribution(pa, pb, format): exact pmf over total games,
  from the per-set score distribution and the conditional structure of
  who serves first each set. Advantage-set formats have an unbounded
  tail — truncate at a stated mass threshold (e.g. 1e-9). THIS
  TRUNCATION POINT IS THE LADDER'S UPPER BOUND for advantage-set
  formats — there is no additional "true" unbounded tail beyond it once
  truncated; say so in one place rather than leaving it to be inferred
  from separate paragraphs. `price.py` must flag in metadata that lines
  beyond the truncation point are not priced, rather than silently
  omitting them.
- Fair prices: probability → decimal price = 1/p, with push handling on
  whole-number lines (stake returned on exact tie).
- ladder(pa, pb, format): every over/under games line from minimum to
  the truncation point (see above), with P(over), P(under), fair
  prices. This is the debug view.

Validation gate (tests/test_engine.py):
- p_hold(0.5) == 0.5 exactly; p_hold monotonic in p; p_hold(0.62) in
  the known ~0.78 region — verify against brute-force enumeration, not
  memory.
- Brute-force: enumerate all point sequences for a single game and a
  single tiebreak at several p values; closed form matches to 1e-12.
- Set/match: high-N Monte Carlo (10^7 point-level sims via simulate.py)
  matches closed form within MC error at several (pa, pb) pairs
  including asymmetric ones, for EACH format variant in rules.py.
- Symmetry: p_match(pa, pb) + p_match_b(pa, pb) == 1; swapping players
  mirrors all distributions.
- Tails: P(under minimum possible games) == 0; ladder monotone in the
  line; advantage-set truncation mass below the stated threshold.
- Any historical match used as a validation target checks
  `score_string_suspect` first and is excluded from validation targets
  if flagged (ground rule 8).

## Stage 2 — Player rates (player_rates.py)

From match-level TML stats, estimate each player's serve point-win
probability, per surface, as of any given date — as-of-date discipline
per ground rule 5; retirement/walkover matches excluded from fitting
per ground rule 6; matches flagged `score_string_suspect` (ground rule
8) excluded from any computation that relies on game-level detail from
the score string.

Components, each toggleable so ablations are possible:
- Recency weighting with exponential half-life. Half-life is fitted:
  scan a grid (e.g. 90/180/365/730 days) on FIT data, select on TUNE by
  log-loss predicting next-match serve stats (rolling-origin within
  FIT+TUNE only). Log the FIT-internal rolling-origin value (and its
  fold-to-fold spread, per ground rule 2), the TUNE value, and the
  `frozen` state to the ledger.
- Surface split with partial pooling: a player's clay rate borrows
  strength from their overall rate rather than being estimated on clay
  matches alone. Pooling strength fitted the same way, logged the same
  way.
- Opponent adjustment: serve points won vs. a strong returner means
  more. This is a paired-comparison system (server skill vs. returner
  skill, mutually adjusting) and needs an IDENTIFIABILITY ANCHOR, not
  just "iterate to convergence": fix the tour-level average serve-win%
  in each period as the normalization, so the iteration cannot drift in
  scale while converging in relative terms. Specify the convergence
  criterion explicitly (e.g. max abs rate change < 1e-6, max 50
  iterations, hard-fail if unconverged) — this must not become a silent
  source of rate drift across refits. A two-pass approximation is
  acceptable if the full iteration is documented as future work.
- Shrinkage toward tour-LEVEL average for thin samples (empirical-
  Bayes: weight = n / (n + n0), n0 fitted, logged). Averages computed
  per level (ATP vs Challenger) — never shrink a Challenger player
  toward the ATP mean.
- Carry effective sample size n through as metadata on every rate.
  Never emit a rate without its n.

Validation gate (TUNE set, logged to ledger): the full rate model
predicts next-match serve points won better (lower MAE, and better
match-winner log-loss via the Stage 1 engine) than (a) raw career
average and (b) last-10-matches average.

## Stage 3 — Surface Elo (elo.py)

Standard Elo with surface-specific ratings, fit on FIT, selected on
TUNE, all selections logged to the ledger (value, FIT-internal
rolling-origin value and spread, `frozen` state):
- K-factor and surface blending (surface rating regressed toward
  overall) selected by TUNE log-loss.
- Inactivity handling: rating regresses toward a reference level during
  absence; regression rate fitted, not assumed.
- New/unrated players get a conservative default per tour level.
- Challenger and tour results both update ratings; whether they carry
  different K is a fitted choice.
- Retirements count per the Stage 0 decision; walkovers never update.
  Matches flagged `score_string_suspect` still count for win/loss Elo
  updates (the win/loss outcome itself is rarely in doubt even when the
  detailed score string is suspect) but are excluded from anything in
  this stage that reads game-level detail.
- CROSS-LEVEL CONSISTENCY CHECK (a known failure mode of single-scale
  Elo, distinct from K tuning): on TUNE data, evaluate calibration
  specifically on matches where one player's rating was built ≥80% on
  Challenger results and the opponent is tour-rated. If these are
  systematically mis-calibrated relative to overall calibration, report
  it and fit a level-offset parameter before accepting the stage.

Validation gate (TUNE set, logged): decile calibration of Elo-implied
win probabilities tracks observed win rates; Brier score beats a
rankings-based baseline; the cross-level check is reported either way.

## Stage 4 — Combine rates and ratings (combine.py)

The engine needs (pa, pb). Set the LEVEL (pa + pb) from the serve-rate
model and the SPLIT (pa − pb) as a blend:

  split = (1 − w) * serve_gap + w * elo_gap

- w fitted on FIT, selected on TUNE by match-winner log-loss, logged
  with FIT-internal spread and `frozen` state. Fit w overall AND per
  sample-size bucket (both well-sampled / one thin / both thin) —
  expect thin-data matches to want higher w.
- elo_gap → serve-gap conversion: map Elo win probability to a serve
  gap by numerically inverting the Stage 1 engine (find the gap
  reproducing that win probability at the given level). The inversion
  is well-posed — p_match is monotonic in split at fixed level — but it
  is called at scale during w-fitting: precompute an interpolation grid
  over (level, target win prob) rather than re-solving per pair.
- Output BOTH the blended (pa, pb) and the raw disagreement in
  percentage points between elo-implied and serve-implied match win
  probability, plus each side's effective n. The disagreement is
  metadata for downstream stake/confidence decisions; it must not be
  silently absorbed into the blend.

Validation gate (TUNE set, logged): fitted w beats w=0 and w=1 on
log-loss; per-bucket w values and the disagreement distribution
reported.

## Stage 5 — Cohort prior for thin-data players (cohort.py)

For players below an effective-n threshold (reuse Stage 2 metadata):
- Style vector per player: ace rate, double-fault rate, serve/return
  point-win split, clay-lean (surface differential), hand, height.
  Standardize features; handle missing height/hand gracefully.
- AS-OF-DATE DISCIPLINE RESTATED (ground rule 5), INCLUDING THE
  STANDARDIZATION STATISTICS: every per-player style feature is
  computed from matches strictly before the pricing date. In addition,
  the mean/std (or other standardization statistics) used to z-score
  those features must themselves be computed from FIT data only (or
  as-of the pricing date, using the same as-of-date matches, not the
  full dataset) — do not compute one global mean/std once from
  FIT+TUNE+everything and reuse it for every pricing date. A thin
  player's 2015 ace-rate z-score must not be implicitly informed by the
  tour-wide ace-rate distribution as it looked in 2023. Write a test
  that prices a historical match and asserts no feature input, AND no
  standardization statistic, post-dates it — this is a separate code
  path from Stage 2 and the classic way lookahead sneaks back in
  through the kNN prior.
- k-nearest neighbors among adequately-sampled players (k selected on
  TUNE, evaluated on thin-data matches only, logged).
- The thin player's prior becomes the cohort's average serve/return
  profile, and — where the OPPONENT is well known — the opponent's
  historical performance against that cohort adjusts the matchup.
- Replaces flat tour-average shrinkage only below the threshold; above
  it, Stage 2 shrinkage unchanged.

Validation gate (TUNE set, thin-data matches only, logged): cohort
prior beats tour-average prior on match-winner log-loss and
total-games CRPS. If it does not, keep the module but ship it switched
off, and say so in the report.

## Stage 6 — Venue index (venue.py)

Per-(VENUE, SURFACE) serve-dominance multiplier relative to same-
surface tour average, from historical hold/ace rates:
- Keyed on (venue, surface), never venue alone — Stage 0's surface-
  change audit is the prerequisite. A venue that switched from hard to
  clay must not blend the two histories under one multiplier; handle
  city/sponsor renames per the Stage 0 identifier-stability findings.
- Venue effect estimated with shrinkage toward the surface mean
  (empirical-Bayes; a venue with 15 matches of history should barely
  move). Shrinkage strength fitted on FIT, selected on TUNE, logged
  with FIT-internal spread and `frozen` state.
- Indoor/outdoor and altitude used as features if present in TML;
  otherwise venue identity alone.
- Applied as an adjustment to the LEVEL of (pa + pb) before the engine,
  never to the split.
- Unmeasured venues (first editions) get exactly the surface average —
  flagged in output metadata, same pattern as rules.py's provenance
  flag (ground rule 7).
- As-of-date: a venue's multiplier for pricing a match uses only that
  venue's history before the match date. Matches flagged
  `score_string_suspect` are excluded from the hold/ace-rate history
  used to fit a venue's multiplier.

Validation gate (TUNE set, logged): total-games log-loss/CRPS improves
at venues with above-median history vs. the no-venue baseline; no
degradation at unmeasured venues.

## Stage 7 — IID failure-mode corrections (corrections.py)

Measure, separately, on FIT+TUNE data, CONDITIONING ON THE FORMAT RULE
IN FORCE AT MATCH TIME (via rules.py) and EXCLUDING retirements/
walkovers (ground rule 6) AND matches flagged `score_string_suspect`
(ground rule 8):

**PROVENANCE WEIGHTING — RESOLVED, NOT LEFT OPEN (ground rule 4).**
Compute both corrections two ways: (i) using `documented`-provenance
matches only, and (ii) using `documented` + `inferred` matches
together. Compare the two on TUNE-set calibration exactly like any
other hyperparameter choice, and select whichever weighting scheme
(documented-only, pooled, or a fitted downweighting of `inferred`
matches — pick the specific scheme, don't leave a range open) produces
better TUNE-set tiebreak-occurrence and total-games calibration. Write
the selected scheme and its TUNE-set comparison to `fitted_params.json`
and the ledger, the same as half-life or shrinkage strength. This
replaces any open-ended "consider weighting documented more heavily"
language — by the time this stage is marked passed, the scheme is
decided and logged, not a runtime option.

(a) Tiebreak frequency: does the engine's P(set reaches 6-6)
    systematically understate observed tiebreak frequency? Fit a
    correction (small inflation at the 5-5/6-6 states, or a mixing
    distribution) sized by the measured gap.
(b) Total-games variance: is the engine's games distribution too narrow
    on long formats? Fit a variance-widening mix over (pa, pb) — a
    normal wobble σ is acceptable — sized by measured over-confidence.

These are separate corrections validated separately. Do not assume one
parameter fixes both. Produce before/after calibration plots for each.
Log both to the ledger with FIT-internal spread and `frozen` state.

Validation gate (TUNE set, logged): post-correction, tiebreak-
occurrence calibration and games-total PIT/coverage are both
acceptable, and neither correction degraded the other's calibration.

## Stage 8 — Recalibration (recalibrate.py)

Isotonic regression maps for the probability families actually priced
(match winner, over/under by line region, set scores), fit on FIT+TUNE
predictions vs. outcomes. Maps are frozen artifacts saved to disk,
applied as the final step in price.py.

Validation gate — THIS IS THE ONE AND ONLY USE OF THE PRE-CUTOFF TEST
SET: reliability diagrams before/after on the pre-cutoff test set. No
map is fit or refit on it, and nothing is fit on anything past the
final cutoff.

IF THIS GATE FAILS:
1. Read `reports/tune_ledger.json`, filtered to `frozen: true` entries
   only (the lineage actually feeding `price.py`, not the full dev
   history — ground rule 2).
2. Flag any stage whose TUNE-set gain exceeds its own FIT-internal
   rolling-origin fold-to-fold spread by more than the multiple fixed
   in `constants.py` (ground rule 2) — this is the defined signature of
   a stage that fit TUNE-set noise, not a subjective read.
3. Fix the implicated stage(s), re-run their validation, mark the new
   result `frozen: true` (superseding, not deleting, the old entry),
   and only then re-run Stage 8.
4. Document which stage(s) were implicated and what changed.
5. The pre-cutoff test set is burned at that point. Carve a replacement
   TEST window by moving the TUNE/TEST boundary — never the final
   HOLDOUT cutoff, which is permanently fixed (ground rule 1) — before
   iterating further, and only if enough pre-holdout data remains to do
   so; if it doesn't, that is itself a finding to report, not a reason
   to move the holdout boundary.

## simulate.py — Monte Carlo (supporting module, not a stage)

Point-level simulator matching the engine's assumptions and consuming
the same rules.py format specs, used for: (a) engine verification in
Stage 1 tests across all format variants, (b) quantities without closed
form — ace-count distributions (needs an ace-rate-per-serve-point input
from player data), any exotic prop. Seeded, with sample-size guidance
in docstrings.

## price.py — the product

One entry point:
  price_match(player_a, player_b, tournament_id, year, surface, venue,
              as_of_date)
(format comes from rules.py via tournament+year, not a caller flag)
returns a structured object with, for every market:
  market, line/selection, fair probability, fair decimal price,
plus metadata: (pa, pb) used, each player's effective n, elo/serve
disagreement in pp, venue multiplier and whether the venue was
measured, format spec used AND its provenance (documented/inferred,
ground rule 7), which corrections and calibration maps were applied
(including which Stage 7 provenance-weighting scheme was selected),
any thin-data/cohort flags, and — for advantage-set formats — a flag
on any requested line beyond the truncation point indicating it is not
priced. Everything the ladder can express within its bounds is
included — full over/under ladder, set scores, handicaps, tiebreak
yes/no, per-player games.

No bookmaker odds appear anywhere in the model path. Comparing fair
prices to market prices is a separate downstream concern and must not
feed back into any fitted component.

## Final stage — Backtest (HOLDOUT, run once everything above is frozen)

- Run price_match over all post-cutoff matches (completed matches only;
  matches that ended in retirement are excluded from outcome scoring
  and reported separately; matches flagged `score_string_suspect` are
  reported separately too, not silently pooled with clean matches).
- Report calibration (reliability, Brier, log-loss, CRPS for
  distributions) by market family: match winner, totals, handicap, set
  betting, tiebreak. Split by tour level — Challenger markets are the
  likeliest place for edge — and by format rule AND by its provenance
  (documented vs inferred), to check whether inferred-rule matches
  calibrate noticeably worse.
- Expect NOT to beat efficient match-winner markets; the question is
  whether totals/handicap/tiebreak calibration is tight enough to find
  mispriced lines.
- Ablations: rerun with cohort off, venue off, corrections off — report
  each component's marginal contribution on the holdout.

## Style requirements

- Python 3.11+, numpy/pandas/scipy for the model (sklearn allowed for
  isotonic + kNN). No deep-learning dependencies.
- Type hints throughout. Docstrings state the math, not just the code.
- Every fitted constant AND every resolved design decision (e.g. Stage
  7's provenance-weighting scheme) lives in fitted_params.json produced
  by the fitting scripts, never hardcoded in module bodies.
- reports/ artifacts are regenerable by scripts, committed alongside,
  except tune_ledger.json, which is append-only across runs and whose
  `frozen: true` entries constitute the audit trail of what is actually
  live in the shipped pipeline.