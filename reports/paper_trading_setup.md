# Paper trading: a fourteen-day forward test

Built 2026-08-10. **Paper only** — no bookmaker account, no real stake, and no
code in this repo can place a bet.

## What this expects to find

**Flat-to-negative PnL.** That is stated here at the top because it is the
honest prior, not a disclaimer bolted on afterwards:

- Backtesting found no profitable edge on any of five markets. Flat-stake ROI
  on total games was **-5.01% at Bet365** over 966 bets on TUNE, and -3.41%
  even assuming you always found the best of seven books, which is itself
  fiction (`reports/totals_roi.md`). No configuration had a confidence
  interval clear of zero.
- As a forecaster the model is **better calibrated than the market and much
  worse at discrimination** — it captures about 52% of the bookmakers' edge
  over ignorance, and on the half of matches where the two disagree by more
  than 10pp the market is sharper (`reports/model_vs_market.md`).
- The one positive signal is that on total games the model's edge estimate
  carries **ordering** information: ROI improves monotonically as the edge
  threshold rises. That is a precondition for a selective strategy, not
  evidence of one.

So this run is a measurement, not an attempt to make money. Nothing in the
design tries to dodge the expected loss, and no result at the end should be
read as though it might.

## Why run it at all

Every evaluation window in this project is spent or contaminated. Vendor match
data ends 2026-07-20 and the holdout backtest already read through it
(`PROGRESS.json: holdout_backtest_2026_08_09`). Matches from today forward are
the only data no version of this model has ever seen, and a prediction
committed **before** a match starts cannot be retro-fitted afterwards. That
makes this the cleanest out-of-sample evidence the project can currently
generate.

That property is the deliverable and it is fragile. Two rules protect it:

1. **The ledger is append-only.** A pick row is written with a UTC timestamp
   before the match starts. Settlement appends a second row. No past row is
   ever edited or deleted. Adding a *column* is the sole exception, and only
   through `_widen_ledger()`, which appends empty cells and verifies every
   existing value survives unchanged — see "The schema widened" below.
2. **Nothing is tuned on the results.** No mid-run parameter change, no
   dropping bad days, no changing the staking rule. Changing the staking rule
   starts a new run with a new ledger.

## What was decided

| | |
|---|---|
| Markets | `total_games` and `games_handicap` only |
| Excluded | `match_winner` — proven to have no edge; the CLV that looked positive traced to overrating longshots (`reports/match_winner_diagnosis.md`) |
| Bankroll | 100 units, paper |
| Staking | Both schemes run in parallel on the same picks: **flat** 2 units per bet, and **half-Kelly** (0.5 x Kelly fraction x bankroll) hard-capped at 5 units |
| Odds | **PointsBet AU's JSON API** (`scripts/fetch_pointsbet.py`), since 2026-08-10. One book, not a consensus — see "The odds source changed on day 1" below. Was: OddsPortal via the OddsHarvester wrapper |
| Probabilities | This repo's model: `model/price.py` over `model/engine.py`, `model/corrections.py` and `data/processed/calibration_maps.pkl` |
| Delivery | Slack |
| Duration | 14 days |
| CLV | **Not computed.** The scrape carries no Pinnacle, so there is no sharp reference and any CLV number would be noise. Same-book line movement *is* recorded from 2026-08-12 (`close` rows) and is not CLV |

The blocker named in the brief did not materialise: OddsHarvester's
`CommandEnum` does carry `UPCOMING_MATCHES = "scrape_upcoming"`, so fixtures
can be scraped with the same wrapper that scraped history.

## How it works

`scripts/paper_trade.py` is a plain deterministic script. Scraping, de-vigging,
pricing, Kelly arithmetic, ledger writes, settlement and PnL all live there,
where they can be read and re-run. Order of the daily job:

1. **Settle** yesterday's open positions first, so today's Kelly bankroll
   reflects them. Final scores come from ESPN (`scripts/fetch_results.py`),
   fetched once per match date, and a settlement row is appended. A match with
   no result yet stays open rather than voiding — see "The odds source changed
   on day 1".
2. **Fetch** unstarted ATP fixtures for today and tomorrow, with totals and
   handicap quotes.
3. **Price** each fixture with the model as of today, and compute edge.
4. **Size** under both schemes, apply the 5-unit cap, append pick rows.
5. **Post** to Slack: bets placed, positions settled, running PnL and bankroll
   for both schemes, and how many matches were skipped and why.

Three judgment calls are left to the model rather than the script: writing the
summary in readable prose, reconciling player names the resolver could not
match, and flagging data that looks wrong. None of them can create or change a
bet.

### Decisions inside the script worth knowing about

**Edge is measured against the raw price, not the de-vigged probability.**
The brief said de-vigged. `reports/totals_roi.md` makes the opposite case and
it is the right one: comparing the model against a de-vigged number counts a
bet as positive-edge when the price actually on offer is negative-edge, which
is how paper edges get manufactured. Selection therefore requires
`model_p > 1 / decimal_odds`. The de-vigged probability and the de-vigged edge
are still written to every row, so the looser rule can be evaluated after the
fact without re-running anything.

**One bet per market per match.** A totals ladder's rungs are one correlated
cluster. Staking all of them turns a single opinion into nine bets and makes
the sample look larger than it is. The script takes the highest positive-edge
line in each market.

**One bookmaker per match** — Bet365 when it is quoting, otherwise whichever
book has the most lines up. Not the best price across books: that assumes
perfect shopping and ignores limits and account restriction.

**Skips are never filled in.** Unknown or ambiguous player, tournament not in
the match history, no two-sided quotes, a format the engine does not price, or
a pricing failure — the match is skipped and counted in the summary. Names are
resolved on given-name initial plus shared surname tokens, and a name
compatible with two active players is left unresolved rather than guessed; the
model is already overconfident about players it does not know
(`reports/unknown_players.md`).

**Two guards against nonsense.** Model probabilities outside 5%-95% are not
staked, because they come from a handful of pmf cells at the ladder's
extremes. A quote implying more than a 25pp edge is flagged and skipped as
suspect data, not taken as an opportunity.

**Unfinished matches are voided, not settled.** A retirement leaves a partial
score that would settle a totals under as a win for the wrong reason, so the
stake is returned. Whether a match finished is decided using the `best_of` the
match was *priced* under, carried on the pick row: a 2-1 score alone cannot
say whether a best-of-five was abandoned or a best-of-three completed.

## What changed on 2026-08-12 (measurement only)

A review of the harness found five problems. All five are about what the run
**measures** — the scoring basis and the operational safety net — and none of
them touches the model, the pricing, or the staking rule. Under rule 2 above,
that means the run continues on the same ledger: no parameter moved, nothing
was tuned on the results so far, and `model_fingerprint()` is unchanged. The
ledger's first rows (2026-08-10) remain valid and untouched.

**1. The board row is no longer edge-selected.** `board_lines()` used to pick,
per market, the quote with the largest model-minus-implied gap, and
`calibration_scores()` then scored that row as if it represented "every match
priced". It did not: it was the line where model and book disagree most, which
`reports/model_vs_market.md` already identifies as where the market beats this
model. Scoring a sample chosen by the model's own disagreement measures the
selection rule. The board row is now the quoted line nearest the model's own
median — the `total_games_pmf` median for totals, the game-margin median for
handicaps — on a side fixed per market, so neither the line nor the side can
depend on price. The old rule survives as `max_gap_line()` for anyone who
wants that view; it just no longer decides what gets scored.

**Every Brier, log-loss and ECE number from before this date is on the old,
edge-selected basis and is not comparable to the ones after it.** Two days of
board rows exist under the old rule.

**2. The book is scored on its de-vigged price.** Board rows carried no
`market_p_devig`, so the scoring fell back to `1 / decimal_odds` — a price with
the bookmaker's margin still in it, which is not a forecast and handed the
model a free advantage of unknown size. The two-sided de-vig `quotes()` already
computes is now written on every board row. Rows written before this change
still score off the raw price; that fallback flatters the model, never the
book, so it cannot manufacture an edge for us.

**3. The whole distribution is logged, not one binary probability.** The
pricer computes a full pmf and only a single `model_p` per market used to reach
the ledger, which collapses every settled row to one win/loss Brier point —
unable to distinguish a projection that was confidently wrong from one that
was vaguely right. Four additive columns now carry it: `model_pmf_start` and
`model_pmf` (mass on consecutive integers: total games for `total_games`, game
margin for `games_handicap`), `model_median`, and `outcome_value`, written at
settlement. Both board and pick rows carry them. `distribution_scores()`
reports discrete CRPS and a continuous log-score **per market, never pooled**
(games and margins are different units) and **with no book column** — a
bookmaker quotes lines, not a distribution, so it has nothing to be compared
against here.

**4. Closing prices are captured.** A new `close` row re-prices each open pick
at the last look before its match, carrying `open_decimal_odds` alongside the
closing `decimal_odds`, plus `hours_to_start`. This is **not CLV** and must
never be reported as CLV: it is PointsBet against itself, and there is still no
sharp reference in this run. The capture reads the board fetch the daily job
already makes — `includeLive=false` means a fixture still listed has not
started, so the last run before a match is the closest look at its close this
cadence can reach, and `hours_to_start` records exactly how close. A pick
priced on the current run is skipped, since its "close" would be its open.

**5. A failed run now alarms.** `.github/workflows/paper-trade.yml` reached
Slack only on the success path, so a blocked source or a failed self-check
ended the day in silence. An `if: failure()` step posts the date and a link to
the failed job. It uses inline `curl` rather than `post_slack`, because the
failure may be that the module does not import.

That fix alone was in the wrong place, because **the live path is the Claude
cloud routine (`deploy/routine_prompt.md`), not this repo's Actions.** The
routine's prompt does tell it to post verification failures to Slack, but no
instruction inside a prompt can report the prompt never being read — which is
the failure that actually occurred:

> **2026-08-11 never ran.** The ledger jumps from `run_date` 2026-08-10 to
> 2026-08-12. Day 2 of 14 is missing and nothing said so. It cannot be
> backfilled: a pick row has to be timestamped before its match, and those
> matches are played. The day is a permanent gap in the record and the run is
> 13 days, not 14.

So the check now lives outside the thing it checks:

- **`.github/workflows/paper-trade-watchdog.yml`** — read-only. It parses
  `paper/ledger.csv` (as CSV, not grep: a bare date also appears in
  `match_date` and `ts_utc`) and alerts Slack if a day inside the run window
  produced no rows. It never prices, scrapes, commits or touches `paper/`, so
  it does not violate the "one or the other, never both" rule, which is about
  two things *writing* the ledger. It also alerts if the check itself fails —
  a watchdog that can die quietly is not a watchdog.
- It checks **yesterday**, at 12:00 UTC. The routine's slot is not fixed
  (2026-08-10 saw runs at 13:11 and 23:11 UTC), so a same-day check would
  false-alarm on a routine that simply had not run yet, and a daily false alarm
  gets muted. Up to ~36h of detection lag buys zero false positives.
- **`paper-trade.yml`'s cron is now off**, `workflow_dispatch` only. An armed
  schedule next to a live routine is the double-write rule waiting to be broken
  by whoever re-enables Actions on the repo. To make the workflow the live path
  again: restore the cron *and* stop the routine, in that order.

### The schema widened

`LEDGER_COLUMNS` gained six columns (four for the distribution, two for the
close row). Appending a column without widening the file writes rows with more
fields than the header, which does not parse, so `_widen_ledger()` rewrites
`paper/ledger.csv` once with the new header. This is the one rewrite the
append-only rule permits: it only adds empty cells, it refuses to drop or
rename a column, and it asserts the round trip reproduces every existing value
before replacing the file. Verified against a copy of the pre-change ledger —
14 rows, all values identical. No row's content was edited.

## Verifying it before it runs

```
python -m scripts.paper_trade --self-check   # arithmetic: Kelly, settlement, PnL
python -m scripts.paper_trade --rehearse     # whole chain on a cached 2024 match
python -m scripts.paper_trade --dry-run      # live path, writes nothing, posts nothing
```

`--rehearse` exists because `--dry-run` only proves the live path when
fixtures are on the board, and there are days with no ATP main-tour play at
all — on 2026-08-10 the only tennis on OddsPortal was a Challenger in Todi.
The rehearsal replays one already-scraped match through the same functions.

It earned its keep immediately: it caught a score-parsing bug. OddsPortal
writes a tiebreak set as `7:6 6` when the home player wins it and `6 4 :7`
when the away player does, so the tiebreak digit crosses the colon. A plain
`(\d+):(\d+)` silently dropped the second form — a five-setter came back as
four sets and 13 games short, which would have settled totals bets against the
wrong number.

## Where it runs

**Chosen: GitHub Actions on a private repo**
(`.github/workflows/paper-trade.yml`), on a 09:00 UTC cron.

The first choice was a Managed Agents scheduled deployment, and the setup
script for it is still here (`deploy/paper_trading_deployment.sh`) and still
works. It was dropped for cost: Managed Agents bills per run against API
credit, and this job does not need an agent. Every step that decides money is
already deterministic and in `scripts/paper_trade.py`, which posts to Slack
itself. The agent was doing three things — writing the summary in prose,
reconciling unmatched player names, and flagging odd-looking data — and only
the first is genuinely lost. Unresolvable names were always going to be
skipped rather than guessed, and the absurd-edge guard is in the script. The
Slack summary is now assembled by the script and still carries the skip counts
and reasons.

launchd on the Mac was the other free option and lost for the original reason:
it only fires when the laptop is awake and online, which over fourteen days
means missed days. The forward window is the whole point.

Costs and caveats of running on GitHub:

- The repo goes to GitHub, including `fitted_params.json` and about 11 MB of
  vendored match data. `.gitignore` now tracks exactly three files under
  `data/processed/` — `matches.parquet`, `matches_with_holdout.parquet` and
  `calibration_maps.pkl` — because the sandbox mounts the repo and the pricer
  cannot run without them. Everything else under `data/processed/` stays out.
  **Make the repo private.**
- **Free at this volume.** A private repo gets 2,000 Actions minutes a month;
  fourteen runs at roughly fifteen minutes is about 200.
- **Two repository secrets**, `SLACK_BOT_TOKEN` and `SLACK_CHANNEL`. The
  script prefers the bot token when both are set and falls back to
  `SLACK_WEBHOOK_URL` for local runs. No secret is committed either way.
  (The bot token, rather than an incoming webhook, was originally forced by
  the Managed Agents sandbox — vault secrets substitute at egress into headers
  or the body, and a webhook keeps its secret in the URL path. It is kept
  because it also works everywhere else.)
- **The runner's filesystem is discarded**, so the workflow commits
  `paper/` back to the repo at the end of every run. An uncommitted ledger is
  a lost day. The commit step adds `paper/` only and never amends.
- ~~**Unverified: whether headless Chromium can drive OddsPortal from a GitHub
  runner.**~~ **This risk fired on day 1** and the source was replaced. See
  the next section.
- The schedule is 09:00 **UTC** rather than a local wall clock: a time skipped
  by a spring-forward never fires and one repeated by a fall-back fires twice,
  and a fourteen-day run cannot absorb either.
- The workflow runs `--self-check` and `--rehearse` before the live job, so a
  broken build fails before it can write a ledger row.

## The odds source changed on day 1 (2026-08-10)

The biggest named risk in this document fired immediately. On day 1 the job
scraped nothing: a plain `curl` to oddsportal.com returned HTTP 200 instantly,
while a headless-Chromium navigation to the same page failed at once with
`net::ERR_CONNECTION_RESET`, on both markets and both dates, twice. That is
Cloudflare rejecting the headless fingerprint from a datacentre IP — not a
timeout, and not a quiet day with no matches. Nothing was committed.

The replacement is **PointsBet AU's unofficial JSON API**
(`scripts/fetch_pointsbet.py`), ported from the v1 model's `data/fetch_odds.py`
with the `Handicap Games` market added. It answers plain `urllib` from the
same host in about 0.14s. No browser, no fingerprint to defeat, and the
sandbox allowlist narrows to `api.pointsbet.com` plus Slack and GitHub.

It emits records in the OddsPortal shape, so de-vig, name resolution, pricing,
staking, edge and the ledger are untouched by the swap.

**What it costs, stated plainly:**

- **One book, not a consensus.** OddsPortal quoted many bookmakers and the
  harness picked Bet365. Every price is now PointsBet's. Any number from this
  run is measured against a different market than every 2024 report in this
  repo, and is not directly comparable to them.
- **Settlement needed a second source.** PointsBet's endpoints carry
  `score: null` and the event disappears once the match ends, so there is no
  result behind a pick. Results now come from **ESPN's scoreboard API**
  (`scripts/fetch_results.py`), the same source the v1 model used for the same
  job — though its host has moved, `site.api.espn.com` now answering 403 where
  `site.web.api.espn.com` answers 200. It gives per-set `linescores`, so total
  games and the game margin are exact rather than inferred, and an explicit
  `STATUS_RETIRED` / `STATUS_WALKOVER`, so an unfinished match is identified
  by status rather than by parsing "ret" out of prose.

  Three settlement outcomes are kept strictly distinct, which is the part
  worth reviewing: a finished match **settles**; a retirement or walkover
  **voids**; a match simply *not in the feed yet* stays **open**. Collapsing
  the last two is the dangerous bug — a void returns the stake and marks the
  row settled, so voiding a match that merely has no result yet would close a
  live position and let a ledger of unresolved bets read as a completed test.
  A failed results fetch leaves picks open for the same reason.
- **Two new failure modes, handled.** PointsBet's competition names pass the
  harness's `atp`-and-not-`challenger` filter for doubles, futures and
  outrights, which a singles model would have priced as though two players had
  walked on court; and it names Masters events by city ("ATP Montreal") where
  the match history names them by event ("Canada Masters"), which the fuzzy
  resolver cannot bridge. The first is filtered in the adapter, the second is
  `paper/tournament_aliases.json` — data, reviewed, visible in a diff, on the
  same rules as `player_aliases.json`.

Verified on the live board for 2026-08-10: both ATP Montreal singles matches
priced, both markets, 8 of 8 players resolved, no skips. Settlement verified
against real played matches on 2026-08-09: Berrettini v Navone settled from
ESPN at 31 games and margin −5 (over 22.5 won, home −3.5 lost), and the
Popyrin v Kokkinakis retirement voided rather than settling.

Whether these runners are blocked the same way was never actually tested, only
inferred. `.github/workflows/cloudflare-probe.yml` is a manual-only diagnostic
that answers it: it compares a plain `curl` against a Playwright navigation to
the same OddsPortal page and reports which of block / challenge / success it
got. Even a green result is not on its own a reason to move pricing back — a
source that can begin challenging mid-run is what cost day 1.

## Reading the result at the end

**A fortnight of bets on two markets cannot establish the presence or absence
of an edge.** With roughly one or two bets a day the sample will be in the low
tens, and at those numbers the outcome is dominated by variance: a positive
PnL would be consistent with a model that has no edge, and a negative one is
what a model with a genuine edge would show reasonably often. The Slack
summaries say so every day, and nothing written at the end should quietly drop
the caveat.

What the run *can* deliver:

- Whether the whole chain works forward — fixtures, names, pricing,
  settlement — on data nobody has seen. Several of the failure modes here
  (unknown players, missing lines, vanished matches) have never been
  exercised outside a backtest.
- An honest count of how often the model can price a fixture at all, and why
  it cannot when it cannot.
- A committed, timestamped record that later work can extend rather than
  reconstruct. Fourteen days is not enough; fourteen days that are genuinely
  out of sample is a start that no amount of re-backtesting can produce.
