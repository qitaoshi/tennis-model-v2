# Tennis Prop Model — Project Memory

This file is read automatically every session. It holds the ground rules
that must never be re-derived or silently relaxed. The full build spec is
`MODEL_PROMPT.md`; the stage-by-stage execution procedure is the `/loop`
command (`.claude/commands/loop.md`). This file is the third leg: what's
always true, regardless of which stage is active.

## Data split boundaries (defined once, never renegotiated in a session)

Read the actual values from `model/constants.py` — do not infer split
boundaries from conversation. If `constants.py` doesn't exist yet, creating
it is the first task of the build, per MODEL_PROMPT.md ground rule 1.

- FIT: parameter fitting only.
- TUNE: hyperparameter selection (Stages 2–7). Reused sequentially — see
  ground rule 2 in MODEL_PROMPT.md and `reports/tune_ledger.json`.
- PRE-CUTOFF TEST: touched exactly once, by Stage 8. Never for selection.
- HOLDOUT: touched only by the final backtest, after every stage gate has
  passed.

## Non-negotiables (do not relax these under any circumstance, including
## being asked to "just quickly check" something)

- Never write code that reads HOLDOUT-range data before the final backtest
  stage, including for "just looking" or debugging.
- Never fit or select a parameter against the PRE-CUTOFF TEST set outside
  Stage 8.
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
