# Forward run 1 — 2026-08-10 to 2026-08-15 (stopped, not completed)

Closed early on 2026-08-15 at the human's instruction, five days into a
fourteen-day window. Kept whole because a stopped run is still a record of what
the model claimed before the matches were played, which is the only property a
forward test has that a backtest does not.

## Why it stopped

Three changes, any one of which alone would have ended it — `check_unchanged`
refuses to append to a ledger whose model or staking rule has moved, and all
three moved at once:

1. **Model.** The Stage 3→4→7→calibration cascade landed on 2026-08-15. Every
   row here was written against the pre-cascade parameters.
2. **Staking.** Flat staking off, Kelly cap 5u → 1u.
3. **Selection.** Bets now need a raw edge above 2%; this run staked every
   positive edge.

## What is in here

`ledger.csv` — 328 rows over run dates 2026-08-10, 08-12, 08-13, 08-14:
88 picks, 43 settled, 122 board, 62 board settled, 13 closing prices.
`run_state.json` — the staking rule and model fingerprint those rows assume.

Two days are missing and cannot be backfilled: a pick has to be timestamped
before its match starts. 2026-08-11 the routine did not fire at all, which is
what prompted the watchdog in `.github/workflows/paper-trade-watchdog.yml`.

## How to read the numbers

Don't, as evidence. 43 settled bets across two markets over four days is
dominated by variance, and the run was stopped rather than completed, so even
the fortnight's worth of noise it was designed to accumulate is not here. It
recorded -31.64u flat (-38.6%) and -35.17u Kelly (-25.4%) against a claimed
+6.7% EV, with 45 bets still open and now never to settle. That gap is the kind
of thing this harness exists to notice over months, not something four days can
speak to.
