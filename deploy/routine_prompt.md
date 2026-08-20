# Cloud Routine prompt

Paste this as the Instructions of a cloud routine at
https://claude.ai/code/routines. It is the alternative to
`.github/workflows/paper-trade.yml` — run one or the other, never both, or the
ledger gets two rows for the same pick. (That rule is about the two DEPLOYMENT
paths. It is not about the two model variants in step 2, which write separate
ledgers and are both meant to run every day.)

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

Setup script — it builds an empty interpreter and nothing else. No
Playwright, no `oddsharvester`, no package names:

    pip install uv
    uv venv --python 3.12 /root/venv

The dependency install is deliberately NOT here. It lives in step 1 of the
prompt below. The setup script runs before the repository is checked out, so
its working directory has no `requirements.txt` in it and never will: on
2026-08-20 a setup script ending in
`uv pip install --python /root/venv/bin/python -r requirements.txt` created the
venv and then died with `error: File not found: requirements.txt`, exit 2, and
the session never started. Re-pasting cannot fix that — the file is not missing
from the repo, it is missing from the sandbox at the moment setup runs.

`--python 3.12` is load-bearing and must not be dropped. The sandbox's default
interpreter is 3.11, and the pinned `numpy` requires 3.12 or newer, so
`uv venv` without it fails the install in step 1 outright with "requirements
are unsatisfiable" — no dependencies, no run. It was dropped once already, on
2026-08-10, when `oddsharvester` (the other thing needing 3.12) was removed;
that was harmless while the package list was unpinned and became fatal the
moment it was pinned. The GitHub workflow pins the same 3.12 via
`actions/setup-python`.

Because the setup script now names no packages at all, it no longer goes stale
when `requirements.txt` changes, and re-pasting it is only needed if these two
lines themselves change. That closes the drift that cost 2026-08-19: the
deployed setup script was still the one from 2026-08-10 14:06, installing
Playwright and `oddsharvester` and no scikit-learn, so `model/recalibrate.py`
failed at import and the day was lost.

Everything below the line is the prompt. Paste it whole, version line included.

---

PROMPT VERSION: 2026-08-20

Run today's tennis paper-trading job in this repository.

This is a PAPER measurement. Nothing you do places a real bet, and no code
here can. Read `reports/paper_trading_setup.md` first if anything below is
unclear.

0. Check you are not running a stale copy of these instructions. Read the
   `PROMPT VERSION` line near the top of `deploy/routine_prompt.md` in the
   repository and compare it to the one above.

   If they differ, the prompt pasted into this routine is older than the
   repository's. STOP. Post to Slack that the deployed routine prompt is
   stale, quoting both versions, and run nothing else — not even step 1.

   This is not a formality. A stale prompt does not fail loudly; it quietly
   does an older job. From 2026-08-16 to 2026-08-18 the deployed prompt
   predated the cascade variant, so every day ran the incumbent alone and
   `paper/ledger-cascade.csv` stopped tracking the same fixtures, while every
   Slack summary and every watchdog check reported a healthy day. Three days
   of the paired comparison were lost before anyone looked.

   Re-paste from `deploy/routine_prompt.md` to fix it. The setup script no
   longer names any package, so it does not drift with the prompt any more;
   re-paste it too only if `deploy/routine_prompt.md` shows it changed.

1. Install the dependencies, then verify before trusting the run with money.
   The install runs here, not in the setup script, because this is the first
   point at which the repository exists:

       uv pip install --python /root/venv/bin/python -r requirements.txt
       /root/venv/bin/python -c "import sklearn, pandas; print('deps ok')"

   If the install fails, post the failure to Slack and stop. The `import
   sklearn` line is not decoration — `model/recalibrate.py` needs it, and a
   day was lost in 2026-08-19 to an environment that silently lacked it.

   Then:

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

2. Run the job, both variants, in this order:

       /root/venv/bin/python -m scripts.paper_trade
       /root/venv/bin/python -m scripts.paper_trade --variant cascade

   The script scrapes fixtures, settles yesterday's open positions, prices
   today's, sizes both staking schemes, appends to its ledger and posts its
   own summary to Slack.

   The two runs price the SAME fixtures under two different models. The
   default variant is the incumbent and writes `paper/ledger.csv`; the
   cascade variant writes `paper/ledger-cascade.csv` and reads its model from
   `paper/model-cascade/`. Neither touches the other's ledger or run state.

   Run BOTH every day, or the comparison is broken: a day the cascade misses
   is a day the two ledgers no longer cover the same fixtures, and the
   difference between them stops being the model. If the second command
   fails, say so explicitly in Slack — do not quietly ship a day of
   incumbent-only rows as though nothing were missing.

   If the first command fails, do not run the second. A day with cascade rows
   and no incumbent rows is worse than a day with neither.

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
   `paper/ledger-cascade.csv`, `paper/run_state-cascade.json`,
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
