# Tennis Match Projection Model

Pre-match tennis pricer. Two players in, fair probabilities and decimal
prices out for every standard market: match winner, exact set score, total
games (full ladder), game handicap, tiebreak occurrence, per-player games.

**Goal: accurate projections.** Every change is judged on Brier, log-loss,
ECE and CRPS over matches the model has not seen. Betting ROI and CLV are an
external benchmark only, never the objective. (Goal changed 2026-08-08 from
"find mispriced lines"; reports before that date are framed against the old
goal.)

## How it works

Closed-form Barnett & Clarke scoring math on two point-win-on-serve
probabilities, `pa` and `pb`. Monte Carlo (`model/simulate.py`) is used only
to verify the closed form and for quantities it cannot express.

`pa`/`pb` come from a staged pipeline (`model/price.py` runs it end to end):

| Stage | Module | What it adds |
|---|---|---|
| 0 | `data_audit.py` | Canonical match record from TML-Database season files (`vendor/`), ATP + Challenger, 2010+ |
| 1 | `engine.py` | Closed-form game/set/match/tiebreak math, per format (`rules.py` gives format by tournament and year, with `documented` vs `inferred` provenance) |
| 2 | `player_rates.py` | Serve/return rates as of any date, from matches strictly before it |
| 3 | `elo.py` | Surface Elo, updated in date order |
| 4 | `combine.py` | Rates give the *level* (serve dominance), Elo gives the *split* between players |
| 5 | `cohort.py` | Prior for thin-data players, replacing the flat tour average |
| 6 | `venue.py` | Court-speed index per (venue, surface) |
| 7 | `corrections.py` | Corrections for the two systematic failures of the iid-points assumption |
| 8 | `recalibrate.py` | Isotonic recalibration per market family |

Every fitted constant lives in `fitted_params.json`, produced by
`scripts/fit_stage*.py`. Module bodies hold no fitted numbers.

## Data discipline

Four-way split, defined once in `model/constants.py` and never inferred
from conversation:

```
FIT      2010-01-01 .. 2023-12-31   parameter fitting only
TUNE     2024-01-01 .. 2025-06-30   hyperparameter selection (reused sequentially, traced in reports/tune_ledger.json)
TEST     2025-07-01 .. 2025-12-31   touched once, never for selection
HOLDOUT  2026-01-01 ..              final evaluation only
```

These boundaries are the result of a deliberate re-split on 2026-08-08
after the original holdout was spent; the docstring in `constants.py` is
the authority on what that cost. TUNE/TEST/HOLDOUT were each read once at
aggregate level by the original backtest, so 2026 numbers are reported
with that caveat. The full rule set is in `CLAUDE.md` and `MODEL_PROMPT.md`.
`docs/holdout_hook.md` sketches a PreToolUse hook that hard-blocks
HOLDOUT reads.

## Where it stands (HOLDOUT, 2026-01..07, 6,202 matches, run once 2026-08-09)

- Pooled: Brier 0.183, log-loss 0.542, ECE 0.003, games CRPS 3.44.
- Distribution families (sets, totals, handicap, tiebreak) all calibrate
  under ECE 0.025.
- `match_winner` ECE 0.067 on two independent holdouts. Open defect: model
  overrates underdogs. Not fixed by the calibration refit.
- Versus de-vigged bookmaker consensus on TUNE: model captures ~52% of the
  market's improvement over a coin flip. Better calibrated, much less
  discriminating.
- All five markets unprofitable against real odds. Positive match-winner
  CLV traces to the underdog bias, not skill.

Full findings: `reports/*.md`. Durable stage status: `PROGRESS.json`.

## Layout

```
model/       pipeline modules (one per stage) + price.py entrypoint
scripts/     fitting, backtests, diagnostics, odds/results fetchers, paper_trade.py
tests/       pytest, one file per module
reports/     findings, tune_ledger.json, stage validations
data/        raw + processed (calibration maps, etc.)
vendor/      TML-Database season files (CC BY-NC-SA, non-commercial)
paper/       14-day forward paper-trading ledgers (started 2026-08-10)
deploy/      cloud routine for the daily paper-trade job
web/         model-performance dashboard (in progress)
```

## Usage

Python 3.12+. `pip install -r requirements.txt` (pinned; sklearn pin
matters because calibration maps are pickled estimators).

```python
from datetime import date
from model.price import price_match

# player IDs and tournament code as in the TML data; format resolved from (code, year)
priced = price_match(player_a_id, player_b_id, "540", 2025,
                     "Grass", "Wimbledon", date(2025, 6, 30))
priced.selections         # one Selection per line: market, selection, probability, price
```

Refuses HOLDOUT-range dates unless `allow_holdout=True`, which only the
final backtest passes.

```
python -m pytest tests/
python -m scripts.paper_trade --dry-run --date 2026-08-20
python -m scripts.backtest          # HOLDOUT; human go-ahead required first
```

Build procedure for continuing stage work: the `/loop` command
(`.claude/commands/loop.md`). Read `PROGRESS.json` first.
