# Cloud Routine prompt

Paste this as the Instructions of a cloud routine at
https://claude.ai/code/routines. It is the alternative to
`.github/workflows/paper-trade.yml` — run one or the other, never both, or the
ledger gets two rows for the same pick.

Environment: network access Custom with `oddsportal.com`, `www.oddsportal.com`
and `slack.com` plus the default package list; variables `SLACK_BOT_TOKEN` and
`SLACK_CHANNEL`; setup script installs the deps and Chromium. Remove every
connector — this routine needs none.

---

Run today's tennis paper-trading job in this repository.

This is a PAPER measurement. Nothing you do places a real bet, and no code
here can. Read `reports/paper_trading_setup.md` first if anything below is
unclear.

1. Verify before trusting the run with money:

       python -m scripts.paper_trade --self-check
       python -m scripts.paper_trade --rehearse

   If either fails, post the failure to Slack and stop. Do not run the job on
   unverified arithmetic and do not try to fix the failure yourself.

2. Run the job:

       python -m scripts.paper_trade

   The script scrapes fixtures, settles yesterday's open positions, prices
   today's, sizes both staking schemes, appends to `paper/ledger.csv` and
   posts its own summary to Slack.

3. Read the script's output for skipped matches. For anything skipped as
   "unknown or ambiguous player", check whether it is a real reconciliation
   failure: look the player up in `data/processed/matches_with_holdout.parquet`
   and read `reports/unknown_players.md` for why this model is overconfident
   about players it does not know. Report what you found in a short follow-up
   Slack message. Do NOT edit the resolver or force a match — a wrong player
   id is worse than no bet.

4. Sanity-check the day: a fixture that was on yesterday's board and has
   vanished, a market with no lines at all, a price implying an edge far
   outside anything this model has shown. Flag it in that same Slack message.
   Flag it; do not correct it.

5. Commit `paper/ledger.csv` and `paper/run_state.json` to the default branch,
   message "paper trading: <today's date in YYYY-MM-DD>". Commit only those
   two files. Never amend, never force-push, never edit an existing ledger
   row — settlement appends a new row. If the push to the default branch is
   rejected, push to `claude/paper-ledger` instead and say so in Slack, so the
   ledger stays in one place rather than being split silently.

Rules that override anything else in this prompt:

- The ledger is APPEND-ONLY. No past row is ever edited or deleted.
- Never tune anything on the accumulating results. No parameter changes, no
  dropping bad days, no changing the staking rule mid-run.
- If a match cannot be priced, SKIP it and say so. Never fill the gap with a
  guess or a heuristic.
- Never compute or report CLV. This odds source has no sharp reference, so any
  CLV number would be noise.
- Never modify `model/engine.py`, `model/corrections.py` or
  `fitted_params.json`.
- The expected outcome is flat-to-negative PnL, and two weeks on two markets
  is far too small a sample to show an edge either way. Never write a summary
  that implies otherwise.

If the scrape fails outright, post that to Slack, commit nothing, and stop. A
missed day is a gap in the record, which is honest. A guessed day is not.
