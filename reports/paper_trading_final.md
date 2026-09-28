# Paper trading — final results (run stopped 2026-09-28)

Forward paper-trading run on real PointsBet lines, `total_games` and
`games_handicap` markets, flat $1 stakes. Planned for 14 days from
2026-08-19; the Managed Agents deployment kept firing daily past that
window and was stopped here, giving 35 days of data (2026-08-19 to
2026-09-23) instead.

| Variant | Bets settled | W-L-Push | Staked | P&L | ROI |
|---|---|---|---|---|---|
| primary (nocohort) | 96 | 43-47-6 | $34 | +$2.61 | +7.7% |
| cascade | 95 | 41-48-6 | $33 | +$1.11 | +3.4% |

Both variants lose more bets than they win; small positive ROI comes from
wins landing on higher-payout lines. At ~95 bets and $1 stakes this is far
too small a sample to update [[tennis-model-next-step-clv]] or
[[multi-market-no-edge-totals-signal]] — treat it as consistent with "no
demonstrated edge," not as evidence of one. The agent's own operating
instructions said as much going in: "two weeks on two markets is far too
small a sample to show an edge either way."

## Why this stopped here

Per [[qitao-goal-betting-edge]], ROI/CLV are a benchmark, not the
objective — the goal is calibration and projection accuracy (Brier,
log-loss, ECE, CRPS), which this run never measured. The infrastructure
(PointsBet scraping, ESPN settlement, Slack posting, the Managed Agents
cron) was removed 2026-09-28 along with the branch cleanup. What's kept:

- This summary.
- Raw ledgers: `paper/ledger.csv`, `paper/ledger-cascade.csv` (final state
  as of 2026-09-23, pulled from the last cloud-routine commit before
  teardown).
- `paper/archive/run1-2026-08-10/`, `paper/archive/run2-2026-08-15/` — the
  two staking-rule iterations that preceded the final run.

What's removed: `scripts/fetch_pointsbet.py`, `scripts/fetch_results.py`,
`scripts/paper_trade.py`, `scripts/backfill_run2_staking.py`,
`deploy/paper_trading_deployment.sh`, `deploy/routine_prompt.md`,
`.github/workflows/paper-trade.yml`,
`.github/workflows/paper-trade-watchdog.yml`,
`.github/workflows/cloudflare-probe.yml`.

The Managed Agents deployment itself ("Tennis paper trading (14 days)")
needs archiving via the Anthropic API separately — no API key was available
in the session that did this cleanup. See git history for
`deploy/paper_trading_deployment.sh` for the archive call.
