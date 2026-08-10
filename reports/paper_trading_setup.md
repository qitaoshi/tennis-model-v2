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
   ever edited or deleted.
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
| Odds | OddsPortal via the existing OddsHarvester wrapper, in its upcoming-fixtures mode |
| Probabilities | This repo's model: `model/price.py` over `model/engine.py`, `model/corrections.py` and `data/processed/calibration_maps.pkl` |
| Delivery | Slack |
| Duration | 14 days |
| CLV | **Not computed.** The scrape carries no Pinnacle, so there is no sharp reference and any CLV number would be noise |

The blocker named in the brief did not materialise: OddsHarvester's
`CommandEnum` does carry `UPCOMING_MATCHES = "scrape_upcoming"`, so fixtures
can be scraped with the same wrapper that scraped history.

## How it works

`scripts/paper_trade.py` is a plain deterministic script. Scraping, de-vigging,
pricing, Kelly arithmetic, ledger writes, settlement and PnL all live there,
where they can be read and re-run. Order of the daily job:

1. **Settle** yesterday's open positions first, so today's Kelly bankroll
   reflects them. Each settlement re-scrapes that match link for its final
   score and appends a settlement row.
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
- **Unverified: whether headless Chromium can drive OddsPortal from a GitHub
  runner.** OddsPortal sits behind Cloudflare and datacentre IPs are the ones
  most likely to be challenged. This is the single biggest open risk in the
  whole setup. Trigger the workflow manually with `dry_run: true` and confirm
  before trusting the schedule; if it is blocked, launchd on the Mac is the
  fallback, at the cost of missed days.
- The schedule is 09:00 **UTC** rather than a local wall clock: a time skipped
  by a spring-forward never fires and one repeated by a fall-back fires twice,
  and a fourteen-day run cannot absorb either.
- The workflow runs `--self-check` and `--rehearse` before the live job, so a
  broken build fails before it can write a ledger row.

## Reading the result at the end

**A fortnight of bets on two markets cannot establish the presence or absence
of an edge.** With roughly one or two bets a day the sample will be in the low
tens, and at those numbers the outcome is dominated by variance: a positive
PnL would be consistent with a model that has no edge, and a negative one is
what a model with a genuine edge would show reasonably often. The Slack
summaries say so every day, and nothing written at the end should quietly drop
the caveat.

What the run *can* deliver:

- Whether the whole chain works forward — fixtures, names, pricing, settlement
  — on data nobody has seen. Several of the failure modes here (unknown
  players, missing lines, vanished matches) have never been exercised outside
  a backtest.
- An honest count of how often the model can price a fixture at all, and why
  it cannot when it cannot.
- A committed, timestamped record that later work can extend rather than
  reconstruct. Fourteen days is not enough; fourteen days that are genuinely
  out of sample is a start that no amount of re-backtesting can produce.
