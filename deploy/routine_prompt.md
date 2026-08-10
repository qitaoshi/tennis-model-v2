# Cloud Routine prompt

Paste this as the Instructions of a cloud routine at
https://claude.ai/code/routines. It is the alternative to
`.github/workflows/paper-trade.yml` — run one or the other, never both, or the
ledger gets two rows for the same pick.

Environment: variables `SLACK_BOT_TOKEN` and `SLACK_CHANNEL`. Remove every
connector — this routine needs none.

Network access: an allowlist of `api.pointsbet.com` (odds),
`site.web.api.espn.com` (results) and `slack.com` is enough, and is preferred
over full access. Both data sources are plain JSON APIs called with `urllib`,
so there is no browser, no CDN fetch, and nothing whose hosts cannot be
enumerated ahead of time.

That changed on 2026-08-10. The original source was OddsPortal via Playwright,
which needed full network access for `cdn.playwright.dev` and for the page
assets. It also did not work: OddsPortal sits behind Cloudflare and reset every
headless-Chromium connection from the routine's datacentre IP
(`net::ERR_CONNECTION_RESET`), collecting nothing on day 1. `scripts/
fetch_pointsbet.py` replaced it. If you restore the OddsPortal path, restore
full network access with it.

Setup script — no Playwright, no `oddsharvester`, no standalone interpreter:

    pip install uv
    uv venv /root/venv
    uv pip install --python /root/venv/bin/python pandas numpy pyarrow scipy \
        scikit-learn beautifulsoup4 lxml tabulate
    /root/venv/bin/python -c "import sklearn, pandas; print('deps ok')"

---

Run today's tennis paper-trading job in this repository.

This is a PAPER measurement. Nothing you do places a real bet, and no code
here can. Read `reports/paper_trading_setup.md` first if anything below is
unclear.

1. Verify before trusting the run with money:

       /root/venv/bin/python -m scripts.fetch_pointsbet --self-check
       /root/venv/bin/python -m scripts.fetch_results --self-check
       /root/venv/bin/python -m scripts.paper_trade --self-check
       /root/venv/bin/python -m scripts.paper_trade --rehearse

   If any of the four fails, post the failure to Slack and stop. Do not run
   the job on unverified arithmetic and do not try to fix the failure
   yourself.

   `--rehearse` replays a cached 2024 OddsPortal match through the pricing
   chain. That is still the right rehearsal: it exercises quote flattening,
   de-vig, resolution, pricing, staking and settlement on a record with a
   known final score. The live path uses PointsBet, and
   `fetch_pointsbet --self-check` is what covers the parsing of that source.

2. Run the job:

       /root/venv/bin/python -m scripts.paper_trade

   The script scrapes fixtures, settles yesterday's open positions, prices
   today's, sizes both staking schemes, appends to `paper/ledger.csv` and
   posts its own summary to Slack.

3. Resolve names, so the same fixture is not skipped again tomorrow.

   Read `paper/unresolved_names.json`. Each entry is a bookmaker display name
   the resolver could not match, with candidate model players. For each one,
   decide whether it is genuinely the same person.

   Names arrive from PointsBet as "Last, First" and are converted to
   OddsPortal's "Last F." before resolution, so alias keys stay in the "Last
   F." form — that is what `player_aliases.json` is keyed on and what you
   write. Do not add a "Last, First" key; it will never be looked up.

   Confirm before you write anything. Check
   `data/processed/matches_with_holdout.parquet` for the candidate's
   `winner_name` / `loser_name` spellings and recent matches, and satisfy
   yourself the surname and given-name initial actually agree. Read
   `reports/unknown_players.md` for why this model is overconfident about
   players it does not know.

   Only when a name resolves to exactly ONE player beyond doubt, add it to
   `paper/player_aliases.json`:

       {
         "Alcaraz Garfia C.": {
           "player_id": "A0E2",
           "model_name": "Carlos Alcaraz",
           "added": "2026-08-11",
           "reason": "OddsPortal prints both surnames; sole active match"
         }
       }

   Rules for that file, no exceptions:
   - One player only. If two candidates are plausible, add NOTHING and say so
     in Slack. A wrong id prices the wrong player, which is worse than no bet.
   - Never remove or overwrite an existing entry.
   - Never edit `scripts/paper_trade.py` to force a match. The resolver is not
     yours to change mid-run; the alias file is data, and it shows in a diff.
   - An alias only affects FUTURE picks. Never revisit a past ledger row
     because a name resolved later.

   Say in Slack which aliases you added and which you refused, and why.

   If a fixture was skipped as "tournament not in match history", the same
   rules apply to `paper/tournament_aliases.json`, which maps a bookmaker
   competition name to a `tourney_name` in the match history. PointsBet names
   Masters events by city ("ATP Montreal"); the history names them by event
   ("Canada Masters"). Add a mapping only when the two are unambiguously the
   same tournament — check the surface and the calendar week agree. If you
   are not certain, add nothing and say so. A wrong mapping prices a match on
   the wrong surface, which is worse than skipping it.

4. Sanity-check the day: a fixture that was on yesterday's board and has
   vanished, a market with no lines at all, a price implying an edge far
   outside anything this model has shown. Flag it in that same Slack message.
   Flag it; do not correct it.

5. Commit `paper/ledger.csv`, `paper/run_state.json`,
   `paper/player_aliases.json`, `paper/tournament_aliases.json` and
   `paper/unresolved_names.json` to the
   default branch, message "paper trading: <today's date in YYYY-MM-DD>".
   Commit only files under `paper/`. Never amend, never force-push, never edit an existing ledger
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

Handling failures:

- If the odds fetch fails, retry the job ONCE. If it fails again, post what
  the error actually was to Slack — say plainly whether it looks like a block,
  a timeout, or an empty board, since an empty board is a normal quiet day and
  a block is not — then commit nothing and stop. A day with ATP play but zero
  records is a source failure, not a quiet day; say which one you saw.
- A day where the board is real but no match has totals or handicap open yet
  IS normal. Books post those closer to the match, and the job scans today and
  tomorrow precisely because tomorrow's lines are often not up.
- Settlement runs off ESPN, not the odds source. Three outcomes, and the
  script decides which — never you: a finished match SETTLES, a retirement or
  walkover VOIDS and returns the stake, and a match with no result yet stays
  OPEN and settles on a later run. If a position has been open for more than
  a day or two, say so in Slack; do not settle it by hand from another source
  and do not mark it void to tidy the ledger.
- Never retry by weakening a check, widening a threshold, or editing the
  script. A missed day is a gap in the record, which is honest. A guessed day
  is not.
