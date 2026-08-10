# Tennis Prop Model — Project Memory

This file is read automatically every session. It holds the ground rules
that must never be re-derived or silently relaxed. The full build spec is
`MODEL_PROMPT.md`; the stage-by-stage execution procedure is the `/loop`
command (`.claude/commands/loop.md`). This file is the third leg: what's
always true, regardless of which stage is active.

## The goal (changed 2026-08-08)

**Accurate match projections.** Judge every change on Brier, log-loss, ECE
and CRPS over matches the model has not seen.

Betting ROI and CLV are a *performance metric only* — a useful external
benchmark, never the objective. A change that improves Brier and loses ROI
is a win. A change that improves ROI without improving Brier is not
evidence of anything.

This replaces the original goal ("find mispriced lines"). Reports written
before 2026-08-08 are framed against the old goal; read them with that in
mind. In particular, match-winner accuracy now matters — it was previously
dismissed as uninteresting because bookmakers price it sharply.

## Data split boundaries (defined once, never renegotiated in a session)

Read the actual values from `model/constants.py` — do not infer split
boundaries from conversation. The module docstring is the authority on both
the current boundaries and the 2026-08-08 re-split that produced them.

- FIT: parameter fitting only.
- TUNE: hyperparameter selection. Reused sequentially — see ground rule 2
  in MODEL_PROMPT.md and `reports/tune_ledger.json`.
- PRE-CUTOFF TEST: touched exactly once. Never for selection.
- HOLDOUT: touched only by a final evaluation, after every gate has passed.

**The 2026-08-08 re-split.** The original build finished and spent its
holdout. The goal then changed, and no unused window existed, so the human
chose to re-split: FIT now runs to 2023-12-31, TUNE/TEST cover 2024–2025,
and HOLDOUT is 2026-01-01 onward. `HOLDOUT_CUTOFF` moved once, knowingly,
as part of that decision — it does not move again, and never moves to
rescue a failing gate.

Carry the caveat, don't bury it: the new TUNE, TEST and HOLDOUT were all
read once by the original backtest at the aggregate level. No parameter was
fitted against them, but they are not pristine. Say so when reporting a
2026 number. `constants.PRIOR_EVALUATION_WINDOWS` records what was read
when.

## Non-negotiables (do not relax these under any circumstance, including
## being asked to "just quickly check" something)

- Never write code that reads HOLDOUT-range data before the final
  evaluation, including for "just looking" or debugging.
- Never fit or select a parameter against the PRE-CUTOFF TEST set outside
  the one stage entitled to it.
- Never move `HOLDOUT_CUTOFF` again without an explicit human decision, and
  never as a response to a failing gate. The 2026-08-08 move was one such
  decision, recorded in `constants.py`; it is a precedent for asking, not
  for moving.
- Never hardcode a fitted constant in a module body — it goes in
  `fitted_params.json`, produced by a fitting script, per ground rule 4.
- Never treat "the code runs" or "my test passes" as equivalent to "the
  gate in MODEL_PROMPT.md passed." Check the actual spec text for that
  stage before declaring a gate green.
- Never merge `rules.py`'s `documented` / `inferred` format-rule
  provenance into a single undifferentiated spec anywhere downstream.
- Every stage that fits or selects a hyperparameter logs to
  `reports/tune_ledger.json` (ground rule 2) before its gate is
  considered passed.

## Where things live

- `MODEL_PROMPT.md` — the full spec, stage by stage. Re-read the relevant
  section before starting or resuming a stage; don't work from memory of
  it.
- `PROGRESS.json` — durable, cross-session stage status. Read first on
  every session start or resume.
- `reports/tune_ledger.json` — append-only. Required reading before
  starting Stage 7, and the first place to look if Stage 8 fails.
- `.claude/commands/loop.md` — the `/loop` command that drives the stage
  loop. Invoke it to continue the build; don't reinvent the procedure
  ad hoc in conversation.

## Human checkpoints — stop, don't self-approve

After Stage 0, after the rules.py discrepancy scan, before Stage 7, on
any Stage 8 gate failure, and before the HOLDOUT backtest runs: stop and
wait for explicit go-ahead. Full detail in `.claude/commands/loop.md`.
This applies even in Auto Mode / auto-accept settings — these are
judgment stops, not permission prompts, and are not satisfied by a
tool-permission approval.
