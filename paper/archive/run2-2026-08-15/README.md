# Forward run 2 — 2026-08-15 to 2026-08-19 (stopped, not completed)

Closed on 2026-08-19 at the human's instruction, five days into a fourteen-day
window. Kept whole for the same reason run 1 was: a stopped run still records
what the model claimed before the matches were played, which is the only
property a forward test has that a backtest does not.

## Why it stopped

The paired comparison it existed to produce was broken, and could not be
repaired after the fact.

Run 2's whole purpose was to price the same fixtures under two models on the
same day, so that the only difference between the two ledgers is the model.
That held on 2026-08-15 and again on 2026-08-19, and on no other day. The
routine deployed at claude.ai/code/routines was running a copy of
`deploy/routine_prompt.md` that predated the cascade variant, so it only ever
ran the incumbent; 08-16 and 08-17 wrote incumbent rows alone, and 08-18 wrote
nothing at all. See the 2026-08-19 section of `reports/paper_trading_setup.md`
for the full account.

Those days cannot be backfilled — a pick has to be timestamped before its match
starts — so of five elapsed days the run holds **two** on which the two models
actually saw the same board. Continuing would have meant carrying a
three-day hole through to day 14 and then comparing two ledgers that never
covered the same fixtures.

Nothing about the models changed. Both fingerprints are unmoved, so run 3 is
the same two models over a clean paired window, not a new experiment.

Run 3 started the same day, 2026-08-19, so some Cincinnati fixtures appear as
picks in both this archive and run 3's day 1 — the ones still unstarted when
run 3 opened. They are separate records of separate runs, not duplicates
within one: run 2's copies are closed here and will never settle, and run 3
priced its own afresh at the prices then on the board. Do not pool the two
files.

## What is in here

| | incumbent | cascade |
|---|---|---|
| `ledger.csv` / `ledger-cascade.csv` | 233 rows | 105 rows |
| run dates | 08-15, 08-16, 08-17, 08-19 | 08-15, 08-19 |
| picks | 44 | 16 |
| settled | 37 | 11 |
| left permanently open | 7 | 5 |
| flat | -4.04u | +2.19u |
| Kelly | -3.13u | +2.24u |

`run_state.json` / `run_state-cascade.json` — the staking rule and model
fingerprint those rows assume.

## How to read the numbers

Don't, as evidence — and the cascade's column least of all.

The two are not comparable to each other. The incumbent's figures cover 37
settled bets over four days; the cascade's cover 11, all of them day-1 picks
that sat open until 08-19 because it did not run in between. Different bets
over a different window is not a controlled comparison, and the cascade's
+2.19u against the incumbent's -4.04u is a measure of which days each one
happened to be running, not of which model is better.

Neither is either column evidence on its own. 37 and 11 settled bets across
two markets are both dominated by variance, and the run was stopped rather
than completed.

**The decision to stop was made with these numbers visible.** That is recorded
here deliberately. The reason for stopping is the broken pairing, which was
known before the day's results were, and the models were not changed — but a
run stopped after its results are known is exactly the shape of a
result-dependent decision, and the honest response is to say so in the record
rather than to rely on the motive being clean. Run 3 starts from zero on both
sides, which is what makes it readable regardless.
