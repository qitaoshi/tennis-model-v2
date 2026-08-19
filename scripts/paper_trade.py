"""Forward paper-trading harness: totals and games handicap, 14 days.

Every evaluation window in this project is spent or contaminated. Vendor data
ends 2026-07-20 and the holdout backtest already read through it. Matches from
today forward are the only data no version of this model has seen, and a pick
committed BEFORE the match cannot be retro-fitted. That property — not the
PnL — is the deliverable.

**The expected outcome is flat-to-negative PnL.** Backtesting found no
profitable edge on any of five markets (`reports/totals_roi.md`: -5.01% flat
at Bet365 on TUNE), and the model is a worse discriminator than the market
(`reports/model_vs_market.md`: ~52% of the bookmakers' edge over ignorance).
Nothing here is designed to avoid that; the run is a measurement.

What protects the forward window, and must not be relaxed:

* The ledger is append-only. A pick row is written with a UTC timestamp
  before the match starts. Settlement appends a second row; no past row is
  ever edited.
* Nothing is tuned on the accumulating results. No mid-run parameter change,
  no dropping bad days. Changing the staking rule starts a NEW ledger.
* No CLV is computed. The OddsPortal scrape carries no Pinnacle, so there is
  no sharp reference and any CLV number would be noise
  (`reports/model_vs_market.md`, memory `oddsportal-no-pinnacle`).

Paper only. Nothing here can place a real bet.

Run:  python -m scripts.paper_trade --dry-run
      python -m scripts.paper_trade
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model import constants as C
from model import price as PZ
from model import recalibrate as RC
from scripts import fetch_pointsbet as PB
from scripts import fetch_results as RES
from scripts.multi_market_clv import _keys
from scripts.odds_portal_table import _rows

# --- decided, not tunable ---------------------------------------------------
# Sized so the Kelly cap binds sometimes rather than always. Half-Kelly on a
# 4% edge stakes ~0.5u here and an 8% edge reaches the 1u cap; against the
# previous 100u every bet clearing the 2% floor exceeded 1u, so `stake_kelly`
# was a byte-for-byte copy of `stake_flat` and tracking both recorded the same
# number twice. This is a scale for the Kelly fraction, not money at risk.
BANKROLL_START = 25.0
# Both staking rules are tracked, on the same bets: flat weights every bet
# equally, which is what the projection-accuracy goal wants, while Kelly weights
# by the model's own confidence. Neither is the objective — they answer
# different questions about the same picks.
FLAT_STAKE = 1.0
KELLY_SCALE = 0.5          # half-Kelly
KELLY_CAP = 1.0            # hard cap, units, per bet
#: Minimum raw-price edge for a bet. Measured against the RAW price, the same
#: quantity `pick_bets` ranks on — see the note there on why not the de-vigged
#: one. Run 1 staked every positive edge and filled the ledger with sub-1%
#: opinions the harness cannot resolve in a fortnight; this is the floor asked
#: for on 2026-08-15.
MIN_EDGE = 0.02
MARKETS = ("total_games", "games_handicap")
RUN_DAYS = 14

PAPER_DIR = C.REPO_ROOT / "paper"
LEDGER = PAPER_DIR / "ledger.csv"
STATE = PAPER_DIR / "run_state.json"

# --------------------------------------------------------------------------
# Model variants
# --------------------------------------------------------------------------
# A variant is a (parameters, calibration maps) pair with its own ledger and
# its own run state. Two of them run over the SAME fixtures and the SAME
# quotes on the same day, so the only thing that differs between the ledgers
# is the model — which is what makes them comparable at all.
#
# Why this exists: every historical window in this project has been read, so
# the cascade refit on `claude/tennis-calibration-validation-la21r5` can only
# be judged on matches nobody has seen. Switching the live run onto it would
# have thrown away the run already in flight and restarted the clock; running
# both keeps the incumbent's record intact and starts the challenger's beside
# it. See reports/cascade_confirmation_theirs.md for what is being tested.
#
# The primary variant's paths are the originals, so its fingerprint does not
# move and its ledger is appended to exactly as before.
VARIANTS = {
    "primary": {
        "params": C.FITTED_PARAMS_PATH,
        "maps": RC.MAPS_PATH,
        "ledger": PAPER_DIR / "ledger.csv",
        "state": PAPER_DIR / "run_state.json",
        "label": "incumbent",
    },
    "cascade": {
        "params": PAPER_DIR / "model-cascade" / "fitted_params.json",
        "maps": PAPER_DIR / "model-cascade" / "calibration_maps.pkl",
        "ledger": PAPER_DIR / "ledger-cascade.csv",
        "state": PAPER_DIR / "run_state-cascade.json",
        "label": "cascade",
    },
}

#: Which variant this process is running. Set once by select_variant() before
#: any work happens; never read before that.
VARIANT = "primary"
MODEL_PARAMS = VARIANTS["primary"]["params"]
MODEL_MAPS = VARIANTS["primary"]["maps"]


def select_variant(name: str) -> None:
    """Point the module's ledger, state and model files at one variant.

    Rebinding module globals rather than threading a config object through
    thirty call sites: the variant is fixed for the whole process and every
    one of those sites wants the same answer.
    """
    global VARIANT, LEDGER, STATE, MODEL_PARAMS, MODEL_MAPS
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; "
                         f"expected one of {sorted(VARIANTS)}")
    v = VARIANTS[name]
    VARIANT = name
    LEDGER, STATE = v["ledger"], v["state"]
    MODEL_PARAMS, MODEL_MAPS = v["params"], v["maps"]
    if not MODEL_PARAMS.exists() or not MODEL_MAPS.exists():
        raise SystemExit(
            f"variant {name!r} is missing its model files: "
            f"{MODEL_PARAMS} / {MODEL_MAPS}")
# Curated OddsPortal display name -> model player id. Reviewed and committed,
# so every entry is visible in a diff. Only ever affects FUTURE picks: a past
# ledger row is never revisited because a name was resolved later.
ALIASES = PAPER_DIR / "player_aliases.json"
# Written fresh each run: the names that could not be resolved, with the
# model players that might match. A work list, not a decision.
UNRESOLVED = PAPER_DIR / "unresolved_names.json"
# Curated bookmaker competition name -> tourney_name in the match history.
# PointsBet names Masters events by city, the history by event, so the fuzzy
# resolver matches neither. Same rules as ALIASES: data, reviewed, in a diff.
TOURNAMENT_ALIASES = PAPER_DIR / "tournament_aliases.json"

LEDGER_COLUMNS = [
    "row_type", "ts_utc", "run_date", "match_key", "match_link", "league_slug",
    "match_date", "tournament", "player_a", "player_b", "model_id_a",
    "model_id_b", "best_of", "market", "line", "book_side", "model_selection",
    "bookmaker", "decimal_odds", "market_p_devig", "model_p", "edge_raw",
    "edge_devig", "kelly_full", "stake_flat", "stake_kelly", "bankroll_kelly",
    "result", "pnl_flat", "pnl_kelly", "note",
    # Added 2026-08-12. The model prices a whole distribution and only one
    # binary probability per market used to survive to the ledger, which
    # throws away everything CRPS and a continuous log-score need. `model_pmf`
    # is the mass on consecutive integers starting at `model_pmf_start` —
    # total games for total_games, game margin for games_handicap — and
    # `outcome_value` is what actually happened, written at settlement.
    "model_pmf_start", "model_pmf", "model_median", "outcome_value",
    # Added 2026-08-12 for the `close` row type: the same bet re-priced at the
    # last look before the match. `decimal_odds` on a close row is the closing
    # price and `open_decimal_odds` is what the pick row was written at, so
    # same-book line movement is available even with no sharp reference.
    "open_decimal_odds", "hours_to_start",
]

# Matches whose model probability sits at the ladder's extremes are priced off
# a handful of pmf cells; a 1% quote is not a claim this harness should stake.
MIN_MODEL_P = 0.05
MAX_MODEL_P = 0.95
# A quote implying a 25pp edge against a market that beats us on discrimination
# is a data error, not an opportunity.
ABSURD_EDGE = 0.25

# The side each board row is anchored on. Fixed per market, never chosen by
# price or edge — see `board_lines`. Which side it is does not matter for a
# calibration score (the two are complements); that it is not chosen by the
# model's own opinion does.
ANCHOR_SIDE = {"total_games": "over", "games_handicap": "home"}
# The priced ladder pads its support with rungs carrying essentially no mass.
# Below this they are dropped from the stored pmf.
PMF_TRIM = 1e-7


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def _widen_ledger() -> None:
    """Give the on-disk ledger any column `LEDGER_COLUMNS` has gained.

    Appending a column to the schema without doing this writes rows with more
    fields than the header, which does not parse at all. Widening is the one
    rewrite the append-only rule permits, because it only adds empty cells:
    every value already written stays in its own column, byte for byte, and
    that is asserted before the new file replaces the old one. Dropping or
    renaming a column is refused outright — a ledger that loses a column has
    lost evidence, and no schema change is worth that.
    """
    if not LEDGER.exists():
        return
    with LEDGER.open(newline="") as fh:
        have = next(csv.reader(fh), [])
    if have == LEDGER_COLUMNS:
        return
    lost = [c for c in have if c not in LEDGER_COLUMNS]
    if lost:
        raise SystemExit(
            f"{LEDGER} has column(s) the schema no longer names: {lost}. "
            "Widening only ever adds columns; removing one would discard "
            "rows already written. Restore them to LEDGER_COLUMNS.")
    old = pd.read_csv(LEDGER, dtype=str, keep_default_na=False)
    wide = old.reindex(columns=LEDGER_COLUMNS).fillna("")
    tmp = LEDGER.with_suffix(".csv.widening")
    wide.to_csv(tmp, index=False)
    check = pd.read_csv(tmp, dtype=str, keep_default_na=False)
    if len(check) != len(old) or not check[have].equals(old[have]):
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"refusing to widen {LEDGER}: round trip did not "
                         "reproduce the existing rows exactly")
    tmp.replace(LEDGER)
    print(f"widened {LEDGER} with {len(LEDGER_COLUMNS) - len(have)} new "
          "column(s); no existing value changed")


def _load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    _widen_ledger()
    return pd.read_csv(LEDGER, dtype=str, keep_default_na=False)


def _append(rows: list[dict], dry_run: bool) -> None:
    """Append rows. Never rewrites an existing line."""
    if not rows:
        return
    frame = pd.DataFrame(rows).reindex(columns=LEDGER_COLUMNS)
    if dry_run:
        print(f"[dry-run] would append {len(frame)} row(s) to {LEDGER}")
        print(frame.to_string(index=False))
        return
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    header = not LEDGER.exists()
    with LEDGER.open("a", newline="") as fh:
        frame.to_csv(fh, header=header, index=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# scraping
# ---------------------------------------------------------------------------


def _scrape(args: list[str]) -> list[dict]:
    """Run the OddsHarvester wrapper in a subprocess and return its records."""
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        cmd = [sys.executable, "-m", "scripts.odds_harvester_dynamic",
               "--output", tmp.name, *args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"scrape failed: {result.stderr[-1500:]}")
        return json.loads(Path(tmp.name).read_text())


def fixtures(days: list[date]) -> dict[str, dict]:
    """Unstarted ATP main-tour singles fixtures on ``days``, with quotes.

    Sourced from PointsBet AU's JSON API, not OddsPortal. OddsPortal is behind
    Cloudflare and reset every headless-Chromium connection from the routine's
    datacentre IP, collecting nothing on 2026-08-10 (day 1 of this run);
    `scripts/fetch_pointsbet` explains the swap. The records come back in the
    OddsPortal shape, so everything downstream of here is unchanged.

    Both today and tomorrow are scanned, because a book does not post a full
    board a day ahead — on 2026-08-10 the next day's two Montreal matches were
    listed with no totals or handicap open yet. Scanning only tomorrow would
    silently produce empty days. `includeLive=false` drops matches already
    under way, so nothing here can bet a match in progress.

    One book means one price. OddsPortal quoted many and the harness picked
    Bet365; every quote here is PointsBet's, which is a different measurement
    from the 2024 reports and must be reported as such.
    """
    try:
        records = PB.fetch_records(set(days))
    except Exception as exc:
        # Same contract the OddsPortal path had: a source failure is loud and
        # empty, never a quiet day. The routine treats an empty board and a
        # blocked source differently, so this must not be swallowed.
        raise RuntimeError(f"PointsBet fetch failed: {exc}") from exc
    merged: dict[str, dict] = {}
    for rec in records:
        link = rec.get("match_link")
        if link:
            merged[link] = rec
    return merged


def match_day(record: dict, fallback: date) -> date:
    stamp = pd.to_datetime(record.get("match_date"), errors="coerce", utc=True)
    return fallback if pd.isna(stamp) else stamp.date()


def settle_scrape(link: str) -> dict | None:
    """Re-scrape one played OddsPortal match for its final score.

    Only reachable for ledger rows written before the 2026-08-10 source
    change. PointsBet picks settle from ESPN instead (`_final_score`), because
    PointsBet carries no score and drops the event once the match ends.
    """
    try:
        rows = _scrape(["--match-link", link, "--family", "total_games"])
    except RuntimeError as exc:
        print(f"WARNING: settlement scrape failed for {link}: {exc}")
        return None
    return rows[0] if rows else None


def league_slug(league_name: str) -> str:
    """OddsPortal's own league slug, as used by the historic cache paths."""
    name = re.sub(r"\b(19|20)\d{2}\b", "", str(league_name)).strip()
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------------------------------------------------------------------------
# name and tournament reconciliation
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """History the pricer needs, indexed for forward lookups."""

    players: dict[str, list[tuple[frozenset, str]]]
    tournaments: dict[str, tuple[str, str]]
    aliases: dict[str, str]
    known_ids: set[str]
    tournament_aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, as_of: date, active_years: int = 2) -> "Reference":
        m = pd.read_parquet(C.PROCESSED_DIR / "matches_with_holdout.parquet")
        m = m[m["in_scope"] & (m["tour"] == "atp")]
        recent = m[m["date"] >= as_of - timedelta(days=365 * active_years)]

        players: dict[str, list[tuple[frozenset, str]]] = {}
        seen: set[str] = set()
        for name, pid in {
            **dict(zip(recent["winner_name"], recent["winner_id"])),
            **dict(zip(recent["loser_name"], recent["loser_id"])),
        }.items():
            if pid in seen:
                continue
            seen.add(pid)
            initial, surnames = _keys(name)
            players.setdefault(initial, []).append((surnames, str(pid)))

        # Tournament name -> (code, surface), most recent staging wins.
        tourns: dict[str, tuple[str, str]] = {}
        for row in m.sort_values("date").itertuples(index=False):
            tourns[_norm(row.tourney_name)] = (str(row.tourney_code),
                                               str(row.surface))
        aliases = {}
        if ALIASES.exists():
            raw = json.loads(ALIASES.read_text())
            aliases = {_norm(k): str(v["player_id"]) if isinstance(v, dict)
                       else str(v) for k, v in raw.items()}
        ids = set(m["winner_id"].astype(str)) | set(m["loser_id"].astype(str))
        t_alias = {}
        if TOURNAMENT_ALIASES.exists():
            t_alias = {_norm(k): str(v)
                       for k, v in json.loads(
                           TOURNAMENT_ALIASES.read_text()).items()
                       if not k.startswith("_")}
        return cls(players, tourns, aliases, ids, t_alias)

    def player(self, display_name: str) -> str | None:
        """Resolve one OddsPortal display name to a model player id.

        The curated alias file wins, then the automatic match on given-name
        initial plus a shared surname token.

        Ambiguity is left unresolved rather than guessed: a name compatible
        with two active players returns None and the match is skipped. The
        model is already overconfident about players it does not know
        (`reports/unknown_players.md`), so a wrong id is worse than no bet —
        which is also why an alias is only honoured when its player id exists
        in the match table. A typo in the alias file must fail loudly as a
        skip, not quietly price the wrong player.
        """
        alias = self.aliases.get(_norm(display_name))
        if alias:
            return alias if alias in self.known_ids else None
        initial, surnames = _keys(display_name, odds_style=True)
        hits = {pid for known, pid in self.players.get(initial, ())
                if known & surnames}
        return hits.pop() if len(hits) == 1 else None

    def candidates(self, display_name: str, limit: int = 6) -> list[dict]:
        """Plausible model players for a name the resolver could not match.

        This is the work list handed to whoever reviews unresolved names. It
        proposes; it never decides. Anything acted on has to be written into
        the alias file explicitly, where it shows up in a diff.
        """
        _, surnames = _keys(display_name, odds_style=True)
        out = []
        for initial, entries in self.players.items():
            for known, pid in entries:
                shared = known & surnames
                if shared:
                    out.append({"player_id": pid, "initial": initial,
                                "shared_surname_tokens": sorted(shared),
                                "model_name_tokens": sorted(known)})
        return out[:limit]

    def tournament(self, league_name: str) -> tuple[str, str] | None:
        """Resolve a bookmaker league name to (tourney_code, surface).

        The curated alias file wins, then the fuzzy match. PointsBet names
        Masters events by city ('ATP Montreal') where the history names them
        by event ('Canada Masters'), and no amount of substring matching
        bridges that — it is a fact about the two naming schemes, so it is
        recorded as data rather than guessed at.
        """
        alias = self.tournament_aliases.get(_norm(league_name))
        if alias:
            # An alias naming a tournament the history does not contain is a
            # typo; fail to a skip rather than price on the wrong surface.
            return self.tournaments.get(_norm(alias))
        key = _norm(re.sub(r"\b(19|20)\d{2}\b", "",
                           re.sub(r"^atp\b", "", str(league_name).strip(),
                                  flags=re.IGNORECASE)))
        if key in self.tournaments:
            return self.tournaments[key]
        hits = [v for k, v in self.tournaments.items()
                if k and (k in key or key in k)]
        return hits[0] if len(hits) == 1 else None


def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# ---------------------------------------------------------------------------
# quotes and edge
# ---------------------------------------------------------------------------


def quotes(record: dict) -> pd.DataFrame:
    """Flatten one match record and de-vig within each bookmaker and line."""
    flat = pd.DataFrame(_rows(record))
    if flat.empty:
        return flat
    flat = flat[flat["market"].isin(MARKETS)].copy()
    if flat.empty:
        return flat
    flat["implied_raw"] = 1.0 / flat["decimal_odds"]
    denom = flat.groupby("vig_group")["implied_raw"].transform("sum")
    # Both sides of a line are needed for the de-vig to mean anything; a
    # one-sided group would normalise to 1.0 and claim the book has no margin.
    sides = flat.groupby("vig_group")["side"].transform("nunique")
    flat = flat[sides.eq(2)].copy()
    flat["market_p_devig"] = flat["implied_raw"] / denom
    return flat


def model_selection(market: str, side: str, line: float) -> str:
    """Name the pricer's selection matching a bookmaker quote.

    Player A is always the home player here, because the match is priced with
    home as A. The handicap conversion is the one from
    `scripts/multi_market_clv._orient_selection`: the book quotes the HOME
    player's handicap, while the pricer names a handicap by the margin
    threshold itself, so the signs are opposite for the same bet.
    """
    if market == "total_games":
        return f"{side} {line:g}"
    if side == "home":
        return f"A {-line:+g}"
    return f"B {line:+g}"


def kelly(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max((b * p - (1.0 - p)) / b, 0.0)


def model_distribution(priced, market: str) -> tuple[int, list[float]]:
    """Reconstruct the model's discrete distribution for ``market``.

    ``PricedMatch`` exposes per-line probabilities, not the pmf behind them —
    but the ladder steps in halves across the whole support, so its
    half-integer rungs ARE the CDF: ``under 21.5`` is P(total <= 21), and
    ``A +4.5`` is P(margin > 4.5), so 1 minus it is P(margin <= 4). Taking
    first differences recovers the pmf exactly, without reaching into
    `model/` for it.

    Returns ``(first_value, probabilities)``, where ``probabilities[i]`` is
    the mass on ``first_value + i``. Whole-number rungs are skipped: they
    carry a push, so their over/under pair does not partition the support.
    """
    cdf: dict[int, float] = {}
    for s in priced.selections:
        if s.market != market:
            continue
        who, _, value = s.selection.partition(" ")
        try:
            line = float(value)
        except ValueError:
            continue
        if float(line).is_integer():
            continue
        k = int(np.floor(line))
        if market == "total_games" and who == "under":
            cdf[k] = float(s.probability)
        elif market == "games_handicap" and who == "A":
            cdf[k] = 1.0 - float(s.probability)
    if not cdf:
        return 0, []
    keys = sorted(cdf)
    pmf, prev = [], 0.0
    for k in keys:
        pmf.append(max(cdf[k] - prev, 0.0))
        prev = cdf[k]
    # Trim the near-zero tails the ladder pads the support with, so the stored
    # row is a distribution rather than a column of zeros.
    lo, hi = 0, len(pmf) - 1
    while lo < hi and pmf[lo] < PMF_TRIM:
        lo += 1
    while hi > lo and pmf[hi] < PMF_TRIM:
        hi -= 1
    return keys[0] + lo, pmf[lo:hi + 1]


def model_median(priced, market: str) -> float | None:
    """The model's own median total games / game margin, or None."""
    start, pmf = model_distribution(priced, market)
    if not pmf:
        return None
    cum = 0.0
    for i, p in enumerate(pmf):
        cum += p
        if cum >= 0.5:
            return float(start + i)
    return float(start + len(pmf) - 1)


def pmf_fields(priced, market: str) -> dict:
    """The ledger columns that carry the model's whole distribution.

    A single binary probability per market settles into one Brier point and
    nothing else. The distribution behind it settles into a CRPS and a
    continuous log-score, which is a far sharper read on the projection —
    which is the thing this project is judged on (CLAUDE.md). It costs one
    string per row to keep, and it cannot be reconstructed after the fact
    because the pricer's state moves on.
    """
    start, pmf = model_distribution(priced, market)
    if not pmf:
        return {"model_pmf_start": "", "model_pmf": "", "model_median": ""}
    return {"model_pmf_start": int(start),
            "model_pmf": ",".join(f"{p:.6g}" for p in pmf),
            "model_median": model_median(priced, market)}


def parse_pmf(start: object, blob: object) -> tuple[int, np.ndarray] | None:
    """Read `pmf_fields` back off a ledger row, or None if it carries none."""
    try:
        first = int(float(str(start)))
        probs = np.array([float(x) for x in str(blob).split(",") if x != ""],
                         dtype=float)
    except (TypeError, ValueError):
        return None
    return (first, probs) if probs.size else None


def book_threshold(market: str, line: float) -> float:
    """The outcome threshold a bookmaker line sits at, in model units.

    A total is quoted at the threshold itself. A handicap is quoted as the
    HOME player's handicap, so a quoted 4.5 is the margin threshold -4.5 —
    the same sign flip `model_selection` applies.
    """
    return line if market == "total_games" else -line


def max_gap_line(priced, quote_frame: pd.DataFrame) -> list[dict]:
    """Per market, the quote with the largest model-minus-implied gap.

    What `board_lines` used to do. Kept because it answers a real question —
    where does the model disagree with the book most — but it must not choose
    the row the calibration score runs over: that is the line where the two
    disagree most, and `reports/model_vs_market.md` already found the market
    wins there. Selecting on the disagreement and then scoring the
    disagreement measures the selection, not the model.
    """
    probs = {(s.market, s.selection): s for s in priced.selections}
    out = []
    for market in MARKETS:
        rows = quote_frame[quote_frame["market"] == market]
        pick = None
        for row in rows.itertuples(index=False):
            try:
                line = float(row.line)
            except (TypeError, ValueError):
                continue
            sel = probs.get((market, model_selection(market, row.side, line)))
            if sel is None:
                continue
            implied = 1.0 / row.decimal_odds
            cand = {"market": market, "line": f"{line:g}", "side": row.side,
                    "decimal_odds": row.decimal_odds, "implied": implied,
                    "market_p_devig": float(row.market_p_devig),
                    "model_p": sel.probability,
                    "edge": sel.probability - implied}
            if pick is None or cand["edge"] > pick["edge"]:
                pick = cand
        if pick is not None:
            out.append({**pick, **pmf_fields(priced, market)})
    return out


def board_lines(priced, quote_frame: pd.DataFrame) -> list[dict]:
    """One representative quote per market, chosen WITHOUT looking at the edge.

    This is the reporting counterpart to `pick_bets`: same pairing of a
    priced selection to a real quote, but with none of the edge, staking or
    sanity filters, so a market shows up here whether or not it produced a
    bet.

    The line is the quoted one nearest the model's OWN median (nearest total
    for a total, nearest margin threshold for a handicap), on a fixed side per
    market. Both halves of that choice matter: the anchor cannot depend on the
    price, or the calibration denominator becomes the subsample where model
    and book disagree most — which is where the market is already known to
    beat this model — and the side cannot depend on the edge either, or the
    same bias comes back through the back door. `max_gap_line` still computes
    the old, edge-selected choice for anyone who wants it.

    Falls back to the max-gap line only if the ladder yielded no distribution
    to take a median from, which would mean the pricer returned no usable
    half-integer rungs for the market at all.
    """
    probs = {(s.market, s.selection): s for s in priced.selections}
    fallback = {row["market"]: row for row in max_gap_line(priced, quote_frame)}
    out = []
    for market in MARKETS:
        median = model_median(priced, market)
        if median is None:
            if market in fallback:
                out.append(fallback[market])
            continue
        rows = quote_frame[(quote_frame["market"] == market)
                           & (quote_frame["side"] == ANCHOR_SIDE[market])]
        cands = []
        for row in rows.itertuples(index=False):
            try:
                line = float(row.line)
            except (TypeError, ValueError):
                continue
            sel = probs.get((market, model_selection(market, row.side, line)))
            if sel is None:
                continue
            # Ties break toward the smaller absolute line, then the smaller
            # line: an anchor has to be reproducible, not just unbiased.
            rank = (abs(book_threshold(market, line) - median), abs(line), line)
            cands.append((rank, line, row, sel))
        if not cands:
            if market in fallback:
                out.append(fallback[market])
            continue
        _, line, row, sel = min(cands, key=lambda c: c[0])
        implied = 1.0 / row.decimal_odds
        out.append({"market": market, "line": f"{line:g}", "side": row.side,
                    "decimal_odds": row.decimal_odds, "implied": implied,
                    # Two-sided, from `quotes`: the raw price still carries the
                    # vig, and scoring the book on it flatters the model by the
                    # book's whole margin.
                    "market_p_devig": float(row.market_p_devig),
                    "model_p": sel.probability,
                    "edge": sel.probability - implied,
                    **pmf_fields(priced, market)})
    return out


def pick_bets(priced, quote_frame: pd.DataFrame) -> list[dict]:
    """One bet per market: the highest positive raw-price edge.

    Edge is measured against the RAW price on offer, not the de-vigged
    probability. `reports/totals_roi.md` makes the case: comparing against
    the de-vigged number counts a bet as positive-edge when the real price is
    negative-edge, which is how paper edges get manufactured. The de-vigged
    probability is still recorded on every row, so the looser rule can be
    evaluated after the fact without re-running anything.

    One bet per market per match, because a totals ladder's rungs are one
    correlated cluster and staking all of them would multiply a single
    opinion into nine bets.
    """
    probs = {(s.market, s.selection): s for s in priced.selections}
    out = []
    for market in MARKETS:
        rows = quote_frame[quote_frame["market"] == market]
        best = None
        for row in rows.itertuples(index=False):
            try:
                line = float(row.line)
            except (TypeError, ValueError):
                continue
            sel = probs.get((market, model_selection(market, row.side, line)))
            if sel is None:
                continue
            if not MIN_MODEL_P <= sel.probability <= MAX_MODEL_P:
                continue
            edge_raw = sel.probability - 1.0 / row.decimal_odds
            if edge_raw < MIN_EDGE:
                continue
            if edge_raw > ABSURD_EDGE:
                print(f"  FLAG absurd edge {edge_raw:+.1%} on {market} "
                      f"{row.side} {line:g} at {row.decimal_odds} "
                      f"({row.bookmaker}) — skipped as suspect data")
                continue
            cand = {
                "market": market, "line": f"{line:g}", "book_side": row.side,
                "model_selection": model_selection(market, row.side, line),
                "bookmaker": row.bookmaker,
                "decimal_odds": float(row.decimal_odds),
                "market_p_devig": float(row.market_p_devig),
                "model_p": float(sel.probability),
                "edge_raw": edge_raw,
                "edge_devig": float(sel.probability) - float(row.market_p_devig),
            }
            if best is None or cand["edge_raw"] > best["edge_raw"]:
                best = cand
        if best:
            # The bet row carries the same distribution the board row does. A
            # bet is still a projection, and scoring it as one costs nothing.
            out.append({**best, **pmf_fields(priced, market)})
    return out


def closing_rows(ledger: pd.DataFrame, found: dict[str, dict],
                 today: date) -> list[dict]:
    """Re-price every open pick that is still on the board: the closing look.

    There is no sharp reference in this run (`oddsportal-no-pinnacle`), so no
    CLV is computable and none is claimed. What IS measurable is same-book
    movement: PointsBet's own price on the exact bet that was taken, at the
    last moment it was quoted. If the book moves toward a pick after it was
    committed, that is worth having recorded; if it moves away, that is worth
    having recorded too.

    "The last moment it was quoted" is what the run can actually reach. The
    job fires once a day and `fetch_pointsbet` sets ``includeLive=false``, so
    a fixture still on the board has not started — the final run before a
    match is the closest look at its close that exists here, and PointsBet
    drops the event outright once it is under way. `hours_to_start` records
    how close that was, so nobody has to assume.

    A pick priced on THIS run is skipped: re-reading the same fetch would
    write an open price into the close column and manufacture a zero move.

    Purely additive. A close row stakes nothing, settles nothing and is not a
    pick, so `pnl`, `settle` and `calibration_scores` all pass over it, and
    `check_unchanged`'s fingerprint is untouched — no existing row changes
    meaning, so no new ledger is required.
    """
    picks = ledger[ledger["row_type"] == "pick"]
    if picks.empty:
        return []
    done = ledger[ledger["row_type"] == "close"]
    have = set(done["match_key"] + "|" + done["market"] + "|" + done["line"])
    cache: dict[str, pd.DataFrame] = {}
    out = []
    for _, r in picks.iterrows():
        link = str(r["match_link"])
        if link not in found or str(r["run_date"]) >= today.isoformat():
            continue
        if f"{r['match_key']}|{r['market']}|{r['line']}" in have:
            continue
        if link not in cache:
            cache[link] = quotes(found[link])
        q = cache[link]
        if q.empty:
            continue
        line = float(r["line"])
        hit = q[(q["market"] == r["market"]) & (q["side"] == r["book_side"])
                & (q["bookmaker"] == r["bookmaker"])
                & (pd.to_numeric(q["line"], errors="coerce") == line)]
        if hit.empty:
            # The book pulled or moved the rung. A different line is a
            # different bet, so nothing is recorded rather than something
            # nearby being passed off as the close.
            continue
        quote = hit.iloc[0]
        starts = pd.to_datetime(found[link].get("match_date"),
                                errors="coerce", utc=True)
        hours = ("" if pd.isna(starts) else
                 round((starts - pd.Timestamp.now(tz="UTC")).total_seconds()
                       / 3600.0, 2))
        open_odds = float(r["decimal_odds"])
        close_odds = float(quote["decimal_odds"])
        model_p = float(r["model_p"]) if r["model_p"] != "" else float("nan")
        out.append({
            **r.to_dict(), "row_type": "close", "ts_utc": _now(),
            "run_date": today.isoformat(),
            "decimal_odds": close_odds, "open_decimal_odds": open_odds,
            "hours_to_start": hours,
            "market_p_devig": float(quote["market_p_devig"]),
            # The projection did not change, so the edge at the close is the
            # same opinion against a new price.
            "edge_raw": model_p - 1.0 / close_odds,
            "edge_devig": model_p - float(quote["market_p_devig"]),
            # A close row is a price observation, never a position.
            "stake_flat": 0.0, "stake_kelly": 0.0, "kelly_full": "",
            "bankroll_kelly": "", "result": "", "pnl_flat": "",
            "pnl_kelly": "", "outcome_value": "",
            "note": f"close, as_of {today.isoformat()}, "
                    f"{open_odds:.2f} -> {close_odds:.2f}",
        })
    return out


# ---------------------------------------------------------------------------
# settlement
# ---------------------------------------------------------------------------


def _sets(partial_results: object) -> list[tuple[int, int]]:
    """Set scores from OddsPortal's ``partial_results`` string.

    The tiebreak points ride next to the side that LOST the tiebreak, and
    which side that is moves the digit across the colon: ``7:6 6`` is 7-6 and
    ``6 4 :7`` is 6-7. A plain ``(\\d+):(\\d+)`` misses the second form
    entirely and silently drops the set — which is how a five-setter came
    back as four sets and 13 games short. Take the first number on each side
    of the colon instead; the tiebreak digit is always the trailing one.
    """
    out = []
    for token in str(partial_results).strip("()").split(","):
        if ":" not in token:
            continue
        left, right = token.split(":", 1)
        home = re.search(r"\d+", left)
        away = re.search(r"\d+", right)
        if home and away:
            out.append((int(home.group()), int(away.group())))
    return out


def parse_score(record: dict, best_of: int) -> tuple[int, int, int] | None:
    """(total games, home games - away games, sets played) or None.

    Returns None when the match did not finish — a retirement leaves a
    partial score that would settle a totals under as a win for the wrong
    reason. Unfinished matches are voided, not settled.

    ``best_of`` comes from the format the match was PRICED under and is
    carried on the pick row. A 2-1 score alone cannot say whether a
    best-of-five was abandoned or a best-of-three completed, so the score is
    not asked to.
    """
    sets = _sets(record.get("partial_results", ""))
    if not sets:
        return None
    home = sum(a for a, _ in sets)
    away = sum(b for _, b in sets)
    # A set is only won once someone reaches six games; "2:1" is a set
    # abandoned mid-way, not a set led.
    won_home = sum(a > b and a >= 6 for a, b in sets)
    won_away = sum(b > a and b >= 6 for a, b in sets)
    if max(won_home, won_away) < int(best_of) // 2 + 1:
        return None
    return home + away, home - away, len(sets)


def settle_one(row: pd.Series, total: int, diff: int) -> tuple[str, float]:
    """Return (result, profit per unit staked)."""
    line = float(row["line"])
    sel = str(row["model_selection"])
    if row["market"] == "total_games":
        if total == line:
            return "push", 0.0
        won = (total > line) if sel.startswith("over") else (total < line)
    else:
        who, value = sel.split()
        limit = float(value) if who == "A" else -float(value)
        if diff == limit:
            return "push", 0.0
        won = (diff > limit) if who == "A" else (diff < limit)
    if won:
        return "win", float(row["decimal_odds"]) - 1.0
    return "loss", -1.0


def _results_by_day(days: set[str]) -> dict[str, list[dict]]:
    """ESPN results for each date that has an open pick on it.

    A fetch failure yields no results for that day, which leaves its picks
    open rather than voiding them — the same distinction `settle` draws
    everywhere else. An unreachable feed must never close a live position.
    """
    out: dict[str, list[dict]] = {}
    for day in sorted(d for d in days if d):
        try:
            out[day] = RES.results_for(date.fromisoformat(day))
        except (RuntimeError, ValueError) as exc:
            print(f"WARNING: results fetch failed for {day}: {exc}")
            out[day] = []
    return out


def _final_score(link: str, row: pd.Series, results: dict[str, list[dict]],
                 best_of: int) -> tuple[tuple[int, int, int] | None, str | None]:
    """Resolve one match to (score, note).

    Three outcomes, and the difference between the last two is the whole
    point of this function:

    * ``(score, None)``  settle it;
    * ``(None, reason)`` void it — the match is known not to have finished;
    * ``(None, None)``   leave it open — no result is available yet.

    PointsBet picks settle from ESPN. OddsPortal links keep their original
    path, so a ledger written before 2026-08-10 still settles the way it was
    written.
    """
    if not link.startswith("pointsbet:"):
        record = settle_scrape(link)
        parsed = parse_score(record, best_of) if record else None
        if parsed is None:
            return None, ("match did not finish or result unavailable; "
                          "stake returned")
        return parsed, None

    day = str(row["match_date"])
    hit = RES.find_result(results.get(day, []), str(row["player_a"]),
                          str(row["player_b"]))
    if hit is None:
        return None, None  # not in the feed yet, or ambiguous: stay open
    if not hit["finished"]:
        return None, (f"{hit['status']} — match did not finish; "
                      f"stake returned")
    # The harness's player A is the bookmaker's home player, which need not be
    # ESPN's. Getting this backwards would invert every handicap settlement
    # while leaving totals looking correct, so the margin is flipped here
    # rather than assumed to agree.
    margin = -hit["game_margin"] if hit["flipped"] else hit["game_margin"]
    return (hit["total_games"], margin, hit["sets_played"]), None


def settle(ledger: pd.DataFrame, today: date, dry_run: bool) -> list[dict]:
    # Board rows settle by the same arithmetic as picks — they carry a
    # `model_selection` and a line like any other row — but they settle into
    # `board_settle`, never `settle`, so no downstream PnL query can pick
    # them up by accident. Their stakes are zero regardless.
    SETTLES = {"pick": "settle", "board": "board_settle"}
    picks = ledger[ledger["row_type"].isin(SETTLES)]
    done = ledger[ledger["row_type"].isin(SETTLES.values())]
    settled = set(done["match_key"] + "|" + done["market"] + "|"
                  + done["row_type"])
    rows = []
    open_picks = [
        r for _, r in picks.iterrows()
        if f"{r['match_key']}|{r['market']}|{SETTLES[r['row_type']]}"
        not in settled
        and str(r["match_date"]) < today.isoformat()]
    by_link: dict[str, list] = {}
    for r in open_picks:
        by_link.setdefault(str(r["match_link"]), []).append(r)

    # Results are fetched once per match date, not once per pick.
    results = _results_by_day({str(r["match_date"]) for r in open_picks})

    still_open: list[str] = []
    for link, group in by_link.items():
        best_of = int(float(group[0].get("best_of") or 3))
        parsed, note = _final_score(link, group[0], results, best_of)
        if parsed is None and note is None:
            # No result yet — NOT a void. A void returns the stake and marks
            # the row settled, so voiding a match merely missing from the feed
            # would quietly close a live position and let a ledger of
            # unresolved bets read as a completed test. Leave it open; the
            # next run tries again.
            still_open.append(link)
            continue
        for r in group:
            kind = SETTLES[r["row_type"]]
            if parsed is None:
                rows.append({**r.to_dict(), "row_type": kind,
                             "ts_utc": _now(), "run_date": today.isoformat(),
                             "result": "void", "pnl_flat": 0.0,
                             "pnl_kelly": 0.0, "note": note})
                continue
            total, diff, _ = parsed
            result, per_unit = settle_one(r, total, diff)
            # The realised value on the row's OWN scale: what the stored pmf
            # is a forecast of, so CRPS and the continuous log-score can be
            # computed later without re-deriving which market means what.
            outcome = total if r["market"] == "total_games" else diff
            rows.append({**r.to_dict(), "row_type": kind,
                         "ts_utc": _now(), "run_date": today.isoformat(),
                         "result": result, "outcome_value": int(outcome),
                         "pnl_flat": per_unit * float(r["stake_flat"] or 0.0),
                         "pnl_kelly": per_unit * float(r["stake_kelly"] or 0.0),
                         "note": f"final {total} games, margin {diff:+d}"})
    if still_open:
        print(f"{len(still_open)} played match(es) have no result yet; left "
              f"open, not voided. They settle on a later run.")
    return rows


# ---------------------------------------------------------------------------
# PnL
# ---------------------------------------------------------------------------


def model_fingerprint() -> dict:
    """Hash of everything that decides a projection, plus the staking rule.

    A forward test is only evidence if the thing being tested held still for
    the whole window. Swap a calibration map on day 6 and days 1-5 measure a
    model that no longer exists — silently, because nothing in the output
    would look different.
    """
    out = {}
    for label, path in (("fitted_params", MODEL_PARAMS),
                        ("calibration_maps", MODEL_MAPS)):
        out[label] = (hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                      if path.exists() else "absent")
    # MIN_EDGE is part of the staking rule, not a detail: it decides which
    # bets exist at all, so a ledger written under a different floor is not
    # the same experiment. Omitting it here would let that change slip past
    # `check_unchanged` silently, which is the one thing this hash is for.
    out["staking"] = (f"flat={FLAT_STAKE} scale={KELLY_SCALE} "
                      f"cap={KELLY_CAP} min_edge={MIN_EDGE} "
                      f"markets={','.join(sorted(MARKETS))}")
    return out


def check_unchanged(state: dict) -> None:
    """Refuse to add to a ledger whose model or staking rule has moved.

    Loud on purpose. The failure this prevents is not a crash — it is a
    fortnight of rows that look fine and mean nothing. Continuing after a
    real change requires a NEW ledger, per the module docstring, not a
    quieter check.
    """
    was = state.get("fingerprint")
    if not was:
        return
    now = model_fingerprint()
    moved = [k for k, v in now.items() if was.get(k) != v]
    if moved:
        raise SystemExit(
            "model or staking rule changed mid-run: "
            + ", ".join(f"{k} {was.get(k)!r} -> {now[k]!r}" for k in moved)
            + f"\nThe run started {state.get('start_date')} and every row "
              "since assumes these held still. Start a new ledger rather "
              "than continuing this one.")


def crps(start: int, pmf: np.ndarray, outcome: float) -> float:
    """Discrete CRPS: sum over integer thresholds of (F(v) - 1{v >= y})^2.

    Zero for a distribution that put all its mass exactly on what happened,
    and it punishes being confidently wrong by MORE than being vaguely wrong
    — which a single over/under Brier point cannot distinguish at all. The
    grid is widened to reach the outcome, so a result outside the priced
    support is scored rather than dropped.
    """
    cdf = np.cumsum(pmf)
    end = start + len(pmf) - 1
    lo, hi = min(start, int(np.floor(outcome))), max(end, int(np.ceil(outcome)))
    grid = np.arange(lo, hi + 1)
    idx = np.clip(grid - start, 0, len(cdf) - 1)
    F = np.where(grid < start, 0.0, np.where(grid > end, 1.0, cdf[idx]))
    return float(np.sum((F - (grid >= outcome)) ** 2))


def distribution_scores(settled: pd.DataFrame) -> dict:
    """CRPS and continuous log-score per market, on rows carrying a pmf.

    Reported per market and never pooled: a CRPS in games and a CRPS in
    margin are different units, and averaging them would produce a number
    with no meaning. There is no book column here — the bookmaker quotes one
    line, not a distribution, so it has nothing to be scored against.
    """
    out: dict[str, dict] = {}
    needed = {"market", "model_pmf_start", "model_pmf", "outcome_value"}
    if settled.empty or not needed <= set(settled.columns):
        return out
    for market, rows in settled.groupby("market"):
        vals, logs = [], []
        for _, r in rows.iterrows():
            parsed = parse_pmf(r.get("model_pmf_start"), r.get("model_pmf"))
            try:
                y = float(r.get("outcome_value"))
            except (TypeError, ValueError):
                continue
            if parsed is None or not np.isfinite(y):
                continue
            start, pmf = parsed
            vals.append(crps(start, pmf, y))
            i = int(round(y)) - start
            mass = float(pmf[i]) if 0 <= i < len(pmf) else 0.0
            logs.append(-float(np.log(max(mass, 1e-9))))
        if vals:
            out[str(market)] = {"n": len(vals), "crps": float(np.mean(vals)),
                                "log_score": float(np.mean(logs))}
    return out


def calibration_scores(ledger: pd.DataFrame) -> dict:
    """Brier, log-loss and ECE on settled board rows, model against book.

    This is the metric the project is actually judged on (CLAUDE.md, goal
    changed 2026-08-08); ROI is a benchmark. It runs over board rows rather
    than bets so the denominator is every match priced, not the subset where
    an edge cleared a threshold.

    The book's implied probability is scored on exactly the same outcomes.
    A Brier of 0.241 means nothing on its own — beside the book's 0.229 it
    says the model is the worse forecaster on this sample, which is the
    comparison worth reporting.
    """
    s = ledger[ledger["row_type"] == "board_settle"]
    s = s[s["result"].isin(("win", "loss", "push"))]
    # The distribution scores take pushes too: a push is a dead heat on one
    # LINE, but the match still produced a total and a margin, which is what
    # the pmf forecast. Only the binary scores below have to drop it.
    dist = distribution_scores(s)
    s = s[s["result"].isin(("win", "loss"))]
    if s.empty:
        return {"n": 0, "dist": dist}
    p = pd.to_numeric(s["model_p"], errors="coerce").to_numpy(float)
    odds = pd.to_numeric(s["decimal_odds"], errors="coerce").to_numpy(float)
    y = (s["result"] == "win").to_numpy(float)
    # The book is scored on its DE-VIGGED probability, computed two-sided at
    # quote time and carried on the row. The raw 1/odds still contains the
    # book's margin, so scoring it hands the model a free win of unknown size —
    # it is a price, not a forecast. Rows written before the de-vig was stored
    # fall back to the raw number; they are flattering the model, not the book,
    # so the fallback cannot manufacture an edge for us.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / odds
    devig = pd.to_numeric(s["market_p_devig"], errors="coerce").to_numpy(float) \
        if "market_p_devig" in s.columns else np.full(len(s), np.nan)
    implied = np.where(np.isfinite(devig) & (devig > 0) & (devig < 1), devig, raw)
    ok = np.isfinite(p) & np.isfinite(implied) & np.isfinite(y)
    if not ok.any():
        return {"n": 0, "dist": dist}
    p, implied, y = p[ok], implied[ok], y[ok]
    return {
        "n": int(ok.sum()), "dist": dist,
        "brier": RC.brier(p, y), "brier_book": RC.brier(implied, y),
        "log_loss": RC.log_loss(p, y), "log_loss_book": RC.log_loss(implied, y),
        "ece": RC.calibration_error(p, y),
        "ece_book": RC.calibration_error(implied, y),
    }


def _ev_roi(rows: pd.DataFrame) -> float:
    """Stake-weighted mean of the edge the model claimed, on settled bets."""
    stake = pd.to_numeric(rows["stake_flat"], errors="coerce").fillna(0.0)
    edge = pd.to_numeric(rows["edge_raw"], errors="coerce").fillna(0.0)
    return float((edge * stake).sum() / stake.sum()) if stake.sum() else 0.0


def pnl(ledger: pd.DataFrame) -> dict:
    s = ledger[ledger["row_type"] == "settle"]
    # A void returned the stake, so it is neither a win nor a loss and does
    # not belong in an ROI denominator — it would drag any ROI toward zero
    # purely by sitting there. It still counts as settled.
    resolved = s[s["result"] != "void"]
    staked_flat_r = pd.to_numeric(
        resolved["stake_flat"], errors="coerce").fillna(0.0)
    staked_kel_r = pd.to_numeric(
        resolved["stake_kelly"], errors="coerce").fillna(0.0)
    flat_r = pd.to_numeric(resolved["pnl_flat"], errors="coerce").fillna(0.0)
    kel_r = pd.to_numeric(resolved["pnl_kelly"], errors="coerce").fillna(0.0)
    flat = pd.to_numeric(s["pnl_flat"], errors="coerce").fillna(0.0)
    kel = pd.to_numeric(s["pnl_kelly"], errors="coerce").fillna(0.0)
    staked_flat = pd.to_numeric(s["stake_flat"], errors="coerce").fillna(0.0)
    staked_kel = pd.to_numeric(s["stake_kelly"], errors="coerce").fillna(0.0)
    return {
        "n_settled": int(len(s)),
        "flat_pnl": float(flat.sum()),
        "kelly_pnl": float(kel.sum()),
        "flat_bankroll": BANKROLL_START + float(flat.sum()),
        "kelly_bankroll": BANKROLL_START + float(kel.sum()),
        "flat_roi": float(flat_r.sum() / staked_flat_r.sum()) if staked_flat_r.sum() else 0.0,
        "kelly_roi": float(kel_r.sum() / staked_kel_r.sum()) if staked_kel_r.sum() else 0.0,
        "n_open": int((ledger["row_type"] == "pick").sum() - len(s)),
        # What the model *claimed* it would make, on the same settled bets:
        # the edge it saw at the time, staked flat. Realised flat ROI landing
        # far below this is the model overrating its own edge, which is the
        # failure `reports/totals_roi.md` already found on 2024 data. It is
        # not a forecast — it is the claim being scored.
        # Voids are excluded: the stake came back, so there was no wager for
        # the claim to be scored against. Counting them would credit the model
        # with edge on bets that never resolved.
        "ev_roi": _ev_roi(s[s["result"] != "void"]),
    }


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


def post_slack(text: str, dry_run: bool) -> None:
    """Post the summary.

    Two transports, because the Managed Agents sandbox cannot use an incoming
    webhook: vault secrets are substituted at egress into headers or the body,
    and a webhook's secret lives in the URL path. The bot-token path is the
    one that works there; the webhook path is kept for local runs.
    """
    token, channel = os.environ.get("SLACK_BOT_TOKEN"), os.environ.get("SLACK_CHANNEL")
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if dry_run:
        print("\n[dry-run] would post to Slack:\n" + text)
        return
    if token and channel:
        url = "https://slack.com/api/chat.postMessage"
        body = json.dumps({"channel": channel, "text": text}).encode()
        headers = {"Content-Type": "application/json; charset=utf-8",
                   "Authorization": f"Bearer {token}"}
    elif webhook:
        url, body = webhook, json.dumps({"text": text}).encode()
        headers = {"Content-Type": "application/json"}
    else:
        print("no SLACK_BOT_TOKEN/SLACK_CHANNEL or SLACK_WEBHOOK_URL set; "
              "printing instead:\n" + text)
        return
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = resp.read().decode()
    if token and '"ok":false' in payload:
        print(f"WARNING: Slack rejected the post: {payload[:300]}")


def _market_label(market: object) -> str:
    """Short market name for Slack. The ledger keeps the full one."""
    return {"total_games": "games", "games_handicap": "hcap"}.get(
        str(market), str(market))


def _last_name(name: object) -> str:
    """`Davidovich Fokina A.` -> `Davidovich Fokina`. Feed names, not ids."""
    parts = str(name).split()
    if len(parts) > 1 and parts[-1].endswith(".") and len(parts[-1]) <= 2:
        parts = parts[:-1]
    return " ".join(parts) or str(name)


def _selection_label(row: dict) -> str:
    """The bet as a human would say it.

    Handicaps read as `Sinner -4.5`, meaning Sinner to win by 5 games or more.

    The sign is the NEGATION of the one in `model_selection`. That field holds
    a margin threshold, not a handicap: `settle_one` reads "A -4.5" as "home
    margin > -4.5", which pays out when the home player loses by four games —
    that bet is home +4.5 to anyone placing it. Taking the stored sign at face
    value prints every handicap backwards, favourite for underdog, which is
    how this first shipped.

    Totals read as `o24.5` / `u24.5`. The market name is dropped with the word
    — an o/u prefix on a games line is not mistakable for anything else.
    """
    sel = str(row["model_selection"])
    if str(row["market"]) == "total_games":
        side, _, line = sel.partition(" ")
        short = {"over": "o", "under": "u"}.get(side)
        return f"{short}{line}" if short else f"{_market_label('total_games')} {sel}"
    if str(row["market"]) != "games_handicap":
        return f"{_market_label(row['market'])} {sel}"
    who, _, threshold = sel.partition(" ")
    player = row["player_a"] if who == "A" else row["player_b"]
    return f"{_last_name(player)} {-float(threshold):+g}"


def _by_tournament(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows under their tournament, first-seen order."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get("tournament") or "—"), []).append(r)
    return out


def summary(day: date, bets: list[dict], settlements: list[dict],
            skips: dict[str, list[str]], totals: dict, run_day: int,
            board: list[dict] | None = None,
            scores: dict | None = None,
            closes: list[dict] | None = None) -> str:
    # Read top to bottom, each block is skippable once the one above it is
    # read: today's result, then today's actions, then the running totals,
    # then the standing caveats. The caveats are not decoration — they are
    # what stops a +2u day being read as an edge — but they are identical
    # every day, so they sit at the bottom where they can be skimmed past
    # rather than in front of the numbers.
    settled_today = [s for s in settlements
                     if str(s.get("result")) in ("win", "loss", "push")]
    won = sum(1 for s in settled_today if s["result"] == "win")
    day_kelly = sum(float(s["pnl_kelly"] or 0) for s in settlements)

    # The variant is named in the header because both variants post to the same
    # Slack channel, one after the other, and until now their messages were
    # identical down to the header — the only way to tell them apart was to
    # know that the incumbent runs first. Two unlabelled summaries a day is a
    # misreading waiting to happen, and the two ledgers are not comparable on
    # any day one of them missed.
    lines = [f"*Tennis paper trading — {VARIANTS[VARIANT]['label']} — "
             f"day {run_day}/{RUN_DAYS}* "
             f"· {day.isoformat()} · _paper only_"]

    # The one line to read if you read nothing else.
    if settlements:
        # Voids are excluded from the won/played count on purpose — the stake
        # came back, so there was no wager to win. Naming them keeps that
        # count from looking like it lost track of a row in the list below.
        day_flat = sum(float(s["pnl_flat"] or 0) for s in settlements)
        voided = len(settlements) - len(settled_today)
        note = f" · {voided} void" if voided else ""
        lines.append(f"*Yesterday:* {day_flat:+.2f}u flat · "
                     f"{day_kelly:+.2f}u kelly · "
                     f"{won}/{len(settled_today)} won{note}")
    lines.append(f"*Running:* {totals['flat_pnl']:+.2f}u flat "
                 f"({totals['flat_roi']:+.1%}) · "
                 f"{totals['kelly_pnl']:+.2f}u kelly "
                 f"({totals['kelly_roi']:+.1%}) · "
                 f"{totals['n_settled']} settled, {totals['n_open']} open")
    lines.append("")

    # One line per bet, and "flat / kelly" said once in the section heading
    # rather than on every row. The per-row model-vs-implied breakdown is gone
    # from the post and still written to every ledger row.
    if settlements:
        lines.append(f"*Settled ({len(settlements)})* _flat / kelly_")
        for tourney, rows in _by_tournament(settlements).items():
            lines.append(f"_{tourney}_")
            for s in rows:
                mark = {"win": "✅", "loss": "❌", "push": "➖",
                        "void": "⬜"}.get(str(s["result"]), "•")
                lines.append(
                    f"{mark} {s['player_a']} v {s['player_b']} · "
                    f"{_selection_label(s)} · "
                    f"{float(s['pnl_flat'] or 0):+.2f} / "
                    f"{float(s['pnl_kelly'] or 0):+.2f}u")
        lines.append("")

    if bets:
        lines.append(f"*New bets ({len(bets)})* _flat / kelly_")
        for tourney, rows in _by_tournament(bets).items():
            lines.append(f"_{tourney}_")
            for b in rows:
                # Not "*": the line already carries a bold *edge* pair, and a
                # lone trailing asterisk makes an odd number on the line,
                # which Slack renders as stray bold rather than a marker.
                capped = " (cap)" if b["stake_kelly"] >= KELLY_CAP else ""
                lines.append(
                    f"• {b['player_a']} v {b['player_b']} · "
                    f"{_selection_label(b)} @ {b['decimal_odds']:.2f} · "
                    f"edge *{b['edge_raw']:+.1%}* · "
                    f"{b['stake_flat']:.2f} / "
                    f"{b['stake_kelly']:.2f}u{capped}")
    else:
        lines.append("*No bets today.*")
    lines.append("")

    # Every match priced, bet or not, is still WRITTEN to the ledger as a board
    # row and still scored — that is the calibration denominator and it has not
    # changed. Only the Slack rendering is gone, by request: the post lists
    # positions taken, not everything looked at. Read `reports`/the ledger for
    # the full board.
    if board:
        priced_n = sum(1 for m in board if m["lines"])
        lines.append(f"_{priced_n} priced · board in ledger._")
        lines.append("")

    if closes:
        # "not CLV" stays, shortened but never dropped: same book, no sharp
        # reference in this run, so calling it CLV would be wrong.
        moves = [float(c["decimal_odds"]) / float(c["open_decimal_odds"]) - 1.0
                 for c in closes]
        toward = sum(1 for m in moves if m < 0)
        lines.append(f"*Closes ({len(closes)})* {toward} shortened, "
                     f"{len(moves) - toward} drifted · "
                     f"mean {sum(moves) / len(moves):+.1%} "
                     f"_(same book, not CLV)_")
        lines.append("")

    # Both score blocks on one line each. Lower is better throughout; the
    # distribution block has no book column because a bookmaker quotes lines,
    # not a distribution — which is why it carries no ✅/❌ either.
    if scores and scores.get("dist"):
        parts = [f"{_market_label(m)} CRPS {d['crps']:.2f} "
                 f"log {d['log_score']:.2f} _({d['n']})_"
                 for m, d in sorted(scores["dist"].items())]
        lines.append("*Distribution* " + " · ".join(parts))
        lines.append("")

    if scores and scores.get("n"):
        def verdict(model: float, book: float) -> str:
            return "✅" if model < book else "❌"
        lines.append(
            f"*Accuracy vs book* _({scores['n']})_ · "
            f"Brier {scores['brier']:.3f}/{scores['brier_book']:.3f} "
            f"{verdict(scores['brier'], scores['brier_book'])} · "
            f"Log-loss {scores['log_loss']:.3f}/{scores['log_loss_book']:.3f} "
            f"{verdict(scores['log_loss'], scores['log_loss_book'])} · "
            f"ECE {scores['ece']:.3f}/{scores['ece_book']:.3f} "
            f"{verdict(scores['ece'], scores['ece_book'])}")
        lines.append("")

    n_skip = sum(len(v) for v in skips.values())
    if n_skip:
        lines.append(f"*Skipped ({n_skip})*")
        for reason, names in skips.items():
            shown = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
            lines.append(f"• {reason}: {len(names)} — {shown}")
        lines.append("")

    # The EV line changes daily — the model's own claim scored against what
    # happened — so it stays.
    #
    # The standing caveat is one sentence now instead of five. What was cut is
    # static configuration (staking rule, bankroll, edge floor, which book,
    # which results source, which direction each metric runs) that never
    # changed between posts and is documented in
    # reports/paper_trading_setup.md. What is KEPT is the only sentence that
    # blocks a wrong reading rather than describing the setup: that this
    # sample cannot show an edge in either direction. That one is not
    # shortened away — a +2u day read as evidence is the specific failure
    # every version of this footer has existed to prevent.
    lines.append("───")
    if totals["n_settled"]:
        lines.append(f"_Model claimed {totals['ev_roi']:+.1%} EV; realised "
                     f"{totals['flat_roi']:+.1%} flat, "
                     f"{totals['kelly_roi']:+.1%} kelly._")
    lines.append(
        "_Paper only. 14 days on 2 markets cannot show an edge either way, "
        "and flat-to-negative is expected — setup and caveats in "
        "reports/paper_trading_setup.md._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the daily job
# ---------------------------------------------------------------------------


def run(today: date, dry_run: bool) -> str:
    ledger = _load_ledger()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    check_unchanged(state)
    start = date.fromisoformat(state.get("start_date", today.isoformat()))
    run_day = (today - start).days + 1

    # 1. settle yesterday first, so today's Kelly bankroll includes it.
    resolved = settle(ledger, today, dry_run)
    _append(resolved, dry_run)
    if resolved and not dry_run:
        ledger = _load_ledger()
    elif resolved:
        ledger = pd.concat([ledger, pd.DataFrame(resolved)
                            .reindex(columns=LEDGER_COLUMNS)],
                           ignore_index=True)
    # Only staked rows are reported as positions; board rows settled in the
    # same pass feed the calibration score instead.
    settlements = [r for r in resolved if r["row_type"] == "settle"]
    totals = pnl(ledger)
    scores = calibration_scores(ledger)

    # 2. fixtures still to be played, today and tomorrow.
    tomorrow = today + timedelta(days=1)
    found = fixtures([today, tomorrow])
    print(f"{len(found)} unstarted ATP fixture(s) for {today}..{tomorrow}")

    # 2a. the closing look at yesterday's picks, off the same board fetch.
    closes = closing_rows(ledger, found, today)
    if closes:
        print(f"captured {len(closes)} closing price(s)")

    ref = Reference.build(today)
    pricer = PZ.Pricer(today, allow_holdout=True,
                       history_parquet="matches_with_holdout.parquet",
                       fitted=json.loads(MODEL_PARAMS.read_text()),
                       maps_path=MODEL_MAPS)

    skips: dict[str, list[str]] = {}
    unresolved: dict[str, dict] = {}
    bets: list[dict] = []
    board: list[dict] = []
    board_rows: list[dict] = []
    bankroll_kelly = totals["kelly_bankroll"]
    already = set(ledger[ledger["row_type"] == "pick"]["match_key"] + "|"
                  + ledger[ledger["row_type"] == "pick"]["market"])
    already_board = set(ledger[ledger["row_type"] == "board"]["match_key"] + "|"
                        + ledger[ledger["row_type"] == "board"]["market"])

    for link, rec in sorted(found.items()):
        label = f"{rec.get('home_team')} v {rec.get('away_team')}"

        def skip(reason: str) -> None:
            skips.setdefault(reason, []).append(label)

        id_a = ref.player(rec.get("home_team", ""))
        id_b = ref.player(rec.get("away_team", ""))
        if not id_a or not id_b:
            for who, resolved in ((rec.get("home_team", ""), id_a),
                                  (rec.get("away_team", ""), id_b)):
                if not resolved and who:
                    unresolved[who] = {
                        "odds_name": who,
                        "seen_on": match_day(rec, tomorrow).isoformat(),
                        "example_match": label,
                        "candidates": ref.candidates(who)}
            skip("unknown or ambiguous player")
            continue
        resolved = ref.tournament(rec.get("league_name", ""))
        if resolved is None:
            skip("tournament not in match history")
            continue
        code, surface = resolved

        q = quotes(rec)
        if q.empty:
            skip("no two-sided quotes")
            continue
        # One book, so the de-vig and the price come from the same market.
        # Bet365 when it is quoting, otherwise whichever book has the most
        # lines up — not the best price across books, which assumes perfect
        # shopping and is fiction (`reports/totals_roi.md`). Under the
        # PointsBet source there is only ever one book, so this selects it.
        books = q["bookmaker"].value_counts()
        book = "bet365" if "bet365" in books.index else books.index[0]
        q = q[q["bookmaker"] == book]

        played_on = match_day(rec, tomorrow)
        try:
            priced = pricer.price(id_a, id_b, code, played_on.year, surface)
        except Exception as exc:
            skip(f"pricing failed ({type(exc).__name__})")
            continue
        if not priced.selections:
            skip("format not priced by the engine")
            continue

        # What the model thought and what was on offer, for every match that
        # got as far as being priced — including the ones no bet came out of.
        # A day with no bets is still a day the model made a projection, and
        # that projection is the thing being measured, not the staking.
        shown = board_lines(priced, q)
        board.append({
            "label": label, "book": book, "best_of":
                priced.metadata["format"]["best_of"],
            "quotes": len(q), "lines": shown})

        # The same projections as ledger rows, so they can be settled and
        # scored later. These are measurement only and never stake anything:
        # `pnl` reads settled *picks*, so a board row cannot reach PnL, ROI
        # or the bankroll. Scoring only the bet rows would score only the
        # matches where an edge cleared a threshold — a subsample chosen by
        # the model's own errors, which is exactly the wrong denominator for
        # a calibration metric.
        for ln in shown:
            key = f"{link}|{ln['market']}"
            if key in already_board:
                continue
            board_rows.append({
                "row_type": "board", "ts_utc": _now(),
                "run_date": today.isoformat(), "match_key": link,
                "match_link": link,
                "league_slug": league_slug(rec.get("league_name", "")),
                "match_date": played_on.isoformat(),
                "tournament": rec.get("league_name", ""),
                "player_a": rec.get("home_team", ""),
                "player_b": rec.get("away_team", ""),
                "model_id_a": id_a, "model_id_b": id_b,
                "best_of": priced.metadata["format"]["best_of"],
                "market": ln["market"], "line": ln["line"],
                "book_side": ln["side"],
                "model_selection": model_selection(
                    ln["market"], ln["side"], float(ln["line"])),
                "bookmaker": book, "decimal_odds": ln["decimal_odds"],
                "market_p_devig": ln["market_p_devig"],
                "model_p": ln["model_p"], "edge_raw": ln["edge"],
                "edge_devig": ln["model_p"] - ln["market_p_devig"],
                "model_pmf_start": ln["model_pmf_start"],
                "model_pmf": ln["model_pmf"],
                "model_median": ln["model_median"],
                "stake_flat": 0.0, "stake_kelly": 0.0,
                "result": "", "pnl_flat": "", "pnl_kelly": "",
                "note": f"board, as_of {today.isoformat()}, book {book}",
            })

        for bet in pick_bets(priced, q):
            key = f"{link}|{bet['market']}"
            if key in already:
                continue
            full = kelly(bet["model_p"], bet["decimal_odds"])
            stake_kelly = min(KELLY_SCALE * full * bankroll_kelly, KELLY_CAP)
            bets.append({
                **bet, "row_type": "pick", "ts_utc": _now(),
                "run_date": today.isoformat(), "match_key": link,
                "match_link": link,
                "league_slug": league_slug(rec.get("league_name", "")),
                "match_date": played_on.isoformat(),
                "tournament": rec.get("league_name", ""),
                "player_a": rec.get("home_team", ""),
                "player_b": rec.get("away_team", ""),
                "model_id_a": id_a, "model_id_b": id_b,
                "best_of": priced.metadata["format"]["best_of"],
                "kelly_full": full, "stake_flat": FLAT_STAKE,
                "stake_kelly": round(stake_kelly, 4),
                "bankroll_kelly": round(bankroll_kelly, 4),
                "result": "", "pnl_flat": "", "pnl_kelly": "",
                "note": f"as_of {today.isoformat()}, book {book}",
            })

    _append(bets + board_rows + closes, dry_run)
    if dry_run:
        print(f"[dry-run] would write {len(unresolved)} unresolved name(s) "
              f"to {UNRESOLVED}")
    else:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        UNRESOLVED.write_text(json.dumps(
            {"run_date": today.isoformat(),
             "names": sorted(unresolved.values(), key=lambda r: r["odds_name"])},
            indent=2, ensure_ascii=False) + "\n")
    text = summary(today, bets, settlements, skips, totals, run_day, board,
                   scores, closes)
    post_slack(text, dry_run)

    if not dry_run and not STATE.exists():
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"start_date": today.isoformat(),
                                     "run_days": RUN_DAYS,
                                     "bankroll_start": BANKROLL_START,
                                     "flat_stake": FLAT_STAKE,
                                     "kelly_scale": KELLY_SCALE,
                                     "kelly_cap": KELLY_CAP,
                                     "min_edge": MIN_EDGE,
                                     "markets": list(MARKETS),
                                     "variant": VARIANT,
                                     "fingerprint": model_fingerprint()},
                                    indent=2))
    return text


def rehearse() -> None:
    """Drive the whole chain on a cached historic match, writing nothing.

    ``--dry-run`` proves the live path only when fixtures are on the board,
    and there are days with no ATP main-tour play at all. This rehearsal
    replays one already-scraped match through exactly the same functions —
    quote flattening, de-vig, name and tournament resolution, pricing, edge,
    both stakes, ledger row shape, settlement — so the chain can be verified
    on a quiet day. It reads a 2024 file and prices as of that date, so it
    touches no holdout data.
    """
    cached = sorted((C.PROCESSED_DIR / "odds_portal").glob("*_2024_*.json"))
    if not cached:
        raise SystemExit("no cached OddsPortal records to rehearse on")
    # Merge the per-family caches back into one record per match, the same
    # shape the live fixture scrape produces.
    merged: dict[str, dict] = {}
    for path in cached:
        for candidate in json.loads(path.read_text()).get("records", []):
            link = candidate.get("match_link")
            if link:
                merged.setdefault(link, {}).update(candidate)
    record = next((r for r in merged.values()
                   if r.get("partial_results") and r.get("home_team")
                   and not quotes(r).empty), None)
    if record is None:
        raise SystemExit("no cached record carries two-sided quotes")

    played = pd.to_datetime(record["match_date"], utc=True).date()
    as_of = played - timedelta(days=1)
    print(f"rehearsing on {record['home_team']} v {record['away_team']} "
          f"({record['league_name']}, played {played}), priced as of {as_of}")

    ref = Reference.build(as_of)
    id_a, id_b = ref.player(record["home_team"]), ref.player(record["away_team"])
    resolved = ref.tournament(record["league_name"])
    print(f"  players -> {id_a}, {id_b};  tournament -> {resolved}")
    if not (id_a and id_b and resolved):
        raise SystemExit("rehearsal match did not resolve; chain not exercised")
    code, surface = resolved

    q = quotes(record)
    print(f"  {len(q)} two-sided quote(s) across "
          f"{q['bookmaker'].nunique()} bookmaker(s)")
    books = q["bookmaker"].value_counts()
    book = "bet365" if "bet365" in books.index else books.index[0]
    q = q[q["bookmaker"] == book]

    priced = PZ.Pricer(as_of, fitted=json.loads(MODEL_PARAMS.read_text()),
                       maps_path=MODEL_MAPS).price(
        id_a, id_b, code, played.year, surface)
    best_of = priced.metadata["format"]["best_of"]
    bets = pick_bets(priced, q)
    print(f"  {len(bets)} bet(s) at {book}, best_of {best_of}")

    bankroll = BANKROLL_START
    parsed = parse_score(record, best_of)
    for bet in bets:
        full = kelly(bet["model_p"], bet["decimal_odds"])
        stake_kelly = min(KELLY_SCALE * full * bankroll, KELLY_CAP)
        row = pd.Series({**bet, "stake_flat": FLAT_STAKE,
                         "stake_kelly": stake_kelly})
        print(f"   {bet['market']:14s} {bet['book_side']:5s} {bet['line']:>6s} "
              f"@ {bet['decimal_odds']:.2f}  model {bet['model_p']:.3f}  "
              f"edge {bet['edge_raw']:+.3f}  flat {FLAT_STAKE:.2f}u  "
              f"kelly {stake_kelly:.2f}u")
        if parsed is None:
            print("      settles: void (match did not finish)")
            continue
        total, diff, _ = parsed
        result, per_unit = settle_one(row, total, diff)
        print(f"      settles: {result} on {total} games / margin {diff:+d} "
              f"-> flat {per_unit * FLAT_STAKE:+.2f}u  "
              f"kelly {per_unit * stake_kelly:+.2f}u")
    missing = set(LEDGER_COLUMNS) - set(bets[0]) - {
        "row_type", "ts_utc", "run_date", "match_key", "match_link",
        "league_slug", "match_date", "tournament", "player_a", "player_b",
        "model_id_a", "model_id_b", "best_of", "kelly_full", "stake_flat",
        "stake_kelly", "bankroll_kelly", "result", "pnl_flat", "pnl_kelly",
        "note", "outcome_value",
        # Written at settlement / at the closing look, never on a pick.
        "open_decimal_odds", "hours_to_start"} if bets else set()
    assert not missing, missing
    print("rehearsal complete — nothing written, nothing posted")


@dataclass
class _FakePriced:
    """Just enough of `PricedMatch` for the self-check: a list of selections."""

    selections: list


def _ladder(pmf: dict[int, float]) -> list:
    """A totals ladder in the pricer's own shape, from a known pmf.

    The self-check reads the pmf back out of this, so building it here from a
    distribution nobody has to trust is the point: if `model_distribution`
    ever mis-reads the ladder, the round trip fails.
    """
    lo, hi = min(pmf), max(pmf)
    out, cum = [], 0.0
    for g in range(lo - 1, hi + 1):
        cum += pmf.get(g, 0.0)
        line = g + 0.5
        out.append(PZ.Selection("total_games", f"under {line}", cum, 1.0))
        out.append(PZ.Selection("total_games", f"over {line}", 1.0 - cum, 1.0))
    return out


def demo() -> None:
    """Self-check on the arithmetic that decides money (ground rule 10)."""
    assert abs(kelly(0.6, 2.0) - 0.2) < 1e-9, kelly(0.6, 2.0)
    assert kelly(0.4, 2.0) == 0.0
    assert min(KELLY_SCALE * kelly(0.99, 10.0) * 100.0, KELLY_CAP) == KELLY_CAP

    # The edge floor. A quote the model likes by less than MIN_EDGE must
    # produce no bet at all — the failure mode is silent, since a staked
    # sub-threshold row looks exactly like a legitimate one in the ledger.
    class _P:
        def __init__(self, p):
            self.selections = [PZ.Selection("total_games", "over 21.5", p, 1.0)]
    qf = pd.DataFrame([{"market": "total_games", "side": "over", "line": "21.5",
                        "decimal_odds": 2.0, "bookmaker": "x",
                        "market_p_devig": 0.5}])
    assert pick_bets(_P(0.5 + MIN_EDGE / 2), qf) == []
    over = pick_bets(_P(0.5 + MIN_EDGE * 2), qf)
    assert len(over) == 1 and over[0]["edge_raw"] >= MIN_EDGE, over

    assert model_selection("total_games", "over", 21.5) == "over 21.5"
    assert model_selection("games_handicap", "home", -4.5) == "A +4.5"
    assert model_selection("games_handicap", "away", -4.5) == "B -4.5"

    # Slack labels. The handicap must name the player the stake is on and carry
    # the sign from `model_selection`, not from the book's home-quoted line —
    # naming the wrong player is the failure that would read as a real bet.
    hcap = {"market": "games_handicap", "model_selection": "A -4.5",
            "player_a": "Darderi L.", "player_b": "Nakashima B."}
    assert _selection_label(hcap) == "Darderi +4.5", _selection_label(hcap)
    assert _selection_label({**hcap, "model_selection": "B +4.5"}) \
        == "Nakashima -4.5"

    # The label is checked against `settle_one`, not against itself. The sign
    # here is the negation of the stored margin threshold, and the first
    # version of this printed the threshold raw — every handicap backwards,
    # favourite shown as underdog, on a message people act on. Anchoring the
    # test to what actually pays out is what stops that recurring.
    def _pays(sel: str, diff: int) -> bool:
        row = pd.Series({"market": "games_handicap", "line": "4.5",
                         "model_selection": sel, "decimal_odds": "2.00"})
        return settle_one(row, 20, diff)[0] == "win"
    # "Darderi +4.5": survives losing by four, dies losing by five.
    assert _selection_label(hcap).endswith("+4.5")
    assert _pays("A -4.5", -4) and not _pays("A -4.5", -5)
    # "Nakashima -4.5": needs the away player to win by five, not four.
    assert _selection_label({**hcap, "model_selection": "B +4.5"}) \
        .endswith("-4.5")
    assert _pays("B +4.5", -5) and not _pays("B +4.5", -4)
    assert _selection_label(
        {"market": "total_games", "model_selection": "over 21.5"}) == "o21.5"
    assert _selection_label(
        {"market": "total_games", "model_selection": "under 24.5"}) == "u24.5"
    # An unrecognised side must stay legible rather than render as a bare line.
    assert _selection_label(
        {"market": "total_games", "model_selection": "sideways 9"}
    ) == "games sideways 9"
    assert _last_name("Davidovich Fokina A.") == "Davidovich Fokina"
    assert _last_name("Alcaraz C.") == "Alcaraz"
    grouped = _by_tournament([{"tournament": "ATP Cincinnati"},
                              {"tournament": "ATP Montreal"},
                              {"tournament": "ATP Cincinnati"}])
    assert list(grouped) == ["ATP Cincinnati", "ATP Montreal"], list(grouped)
    assert len(grouped["ATP Cincinnati"]) == 2

    row = pd.Series({"market": "total_games", "line": "21.5",
                     "model_selection": "over 21.5", "decimal_odds": "1.90"})
    won, profit = settle_one(row, 23, 3)
    assert (won, round(profit, 6)) == ("win", 0.9), (won, profit)
    assert settle_one(row, 20, 2) == ("loss", -1.0)
    row = pd.Series({"market": "total_games", "line": "22",
                     "model_selection": "under 22", "decimal_odds": "1.90"})
    assert settle_one(row, 22, 2) == ("push", 0.0)

    # Home -4.5 handicap: home covers only by winning 5+ games clear.
    row = pd.Series({"market": "games_handicap", "line": "-4.5",
                     "model_selection": "A +4.5", "decimal_odds": "1.90"})
    assert settle_one(row, 24, 6)[0] == "win"
    assert settle_one(row, 24, 4)[0] == "loss"

    # The board row must be anchored on the model's own median, not on where
    # the model and the book disagree most. This is the whole reason the
    # calibration denominator is trustworthy, so it is checked directly: the
    # 20.5 line below is quoted at a price that makes it far and away the
    # biggest edge, and it must still not be the row that gets scored.
    fake = _FakePriced(_ladder({20: 0.1, 21: 0.3, 22: 0.4, 23: 0.2}))
    start, back = model_distribution(fake, "total_games")
    assert start == 20 and np.allclose(back, [0.1, 0.3, 0.4, 0.2]), (start, back)
    assert model_median(fake, "total_games") == 22.0
    qf = pd.DataFrame([
        {"market": "total_games", "side": s, "line": ln, "decimal_odds": od,
         "market_p_devig": dv, "bookmaker": "x"}
        for ln, s, od, dv in ((20.5, "over", 4.00, 0.24), (20.5, "under", 1.25, 0.76),
                              (21.5, "over", 1.80, 0.53), (21.5, "under", 2.00, 0.47),
                              (22.5, "over", 3.00, 0.32), (22.5, "under", 1.40, 0.68))])
    anchored = board_lines(fake, qf)
    assert [r["line"] for r in anchored] == ["21.5"], anchored
    assert anchored[0]["side"] == "over", anchored
    # ...whereas the old, edge-selected rule lands exactly on 20.5.
    assert max_gap_line(fake, qf)[0]["line"] == "20.5", max_gap_line(fake, qf)

    # The whole distribution reaches the row, and comes back off it intact.
    fields = pmf_fields(fake, "total_games")
    assert fields["model_median"] == 22.0, fields
    rt = parse_pmf(fields["model_pmf_start"], fields["model_pmf"])
    assert rt and rt[0] == 20 and np.allclose(rt[1], [0.1, 0.3, 0.4, 0.2]), rt
    assert parse_pmf("", "") is None
    assert anchored[0]["model_pmf"] == fields["model_pmf"]

    # CRPS: zero for a point mass on the truth, and strictly worse for being
    # confidently wrong than for being spread out around the same answer. A
    # single Brier point cannot tell those two apart, which is why this exists.
    assert crps(20, np.array([1.0]), 20) == 0.0
    sure_wrong = crps(20, np.array([1.0]), 23)
    vague = crps(20, np.array([0.25, 0.25, 0.25, 0.25]), 23)
    assert sure_wrong > vague > 0, (sure_wrong, vague)
    # An outcome outside the priced support is scored, not silently dropped.
    assert crps(20, np.array([0.5, 0.5]), 40) > crps(20, np.array([0.5, 0.5]), 21)

    dist_rows = pd.DataFrame([
        {"market": "total_games", "model_pmf_start": "20",
         "model_pmf": "0.1,0.3,0.4,0.2", "outcome_value": "22"},
        # No pmf: an old row, skipped rather than scored as if it had one.
        {"market": "total_games", "model_pmf_start": "",
         "model_pmf": "", "outcome_value": "22"},
    ])
    ds = distribution_scores(dist_rows)["total_games"]
    assert ds["n"] == 1, ds
    assert abs(ds["log_score"] + np.log(0.4)) < 1e-9, ds

    # A retirement must not settle; a completed match must.
    assert parse_score({"partial_results": "(6:4, 2:1)"}, 3) is None
    assert parse_score({"partial_results": "(6:4, 3:6, 7:5)"}, 3) == (31, 1, 3)
    # Same 2-1 score, best-of-five: abandoned, so voided rather than settled.
    assert parse_score({"partial_results": "(6:4, 3:6, 7:5)"}, 5) is None
    assert parse_score({"partial_results": "(1:6, 2:6, 7:6 6 , 3:6)"},
                       5) == (13 + 24, 13 - 24, 4)
    # The tiebreak digit sits beside whoever lost the tiebreak, on either
    # side of the colon. Both forms must yield the same set score.
    assert _sets("(7:6 6, 6 4 :7)") == [(7, 6), (6, 7)]
    assert parse_score({"partial_results": "(6 4 :7, 6:2, 3:6, 7:5, 4:6)"},
                       5) == (52, 0, 5)
    assert league_slug("ATP Australian Open 2024") == "atp-australian-open"

    # Settlement of a PointsBet pick runs off the ESPN results feed, and the
    # three outcomes must stay distinct. Voiding a match that merely has no
    # result yet would return the stake and CLOSE a live position; leaving a
    # genuine retirement open would never resolve it.
    pick = pd.Series({
        "row_type": "pick", "match_key": "pointsbet:1",
        "match_link": "pointsbet:1", "market": "total_games", "line": "22.5",
        "model_selection": "over 22.5", "decimal_odds": "1.90",
        "match_date": "2026-08-10", "best_of": "3", "stake_flat": "2",
        "stake_kelly": "1", "player_a": "Jodar R.", "player_b": "Fils A."})
    finished = {"finished": True, "total_games": 26, "game_margin": 4,
                "sets_played": 3, "status": "STATUS_FINAL",
                "home_name": "Rafael Jodar", "away_name": "Arthur Fils"}
    feed = {"2026-08-10": [finished]}
    assert _final_score("pointsbet:1", pick, feed, 3) == ((26, 4, 3), None)
    # Not in the feed: open, never void.
    assert _final_score("pointsbet:1", pick, {"2026-08-10": []}, 3) \
        == (None, None)
    # Retired: void, with the status named in the note.
    ret = {**finished, "finished": False, "status": "STATUS_RETIRED",
           "total_games": None, "game_margin": None}
    score, note = _final_score("pointsbet:1", pick, {"2026-08-10": [ret]}, 3)
    assert score is None and note and "STATUS_RETIRED" in note, note
    # Orientation: ESPN listing the players the other way round must flip the
    # margin, or every handicap settles backwards while totals look fine.
    flip = pd.Series({**pick.to_dict(), "player_a": "Fils A.",
                      "player_b": "Jodar R."})
    assert _final_score("pointsbet:1", flip, feed, 3) == ((26, -4, 3), None)

    # And the loop leaves an unresolved position open: zero rows, not a void.
    # The feed is stubbed rather than called; a self-check that needs the
    # network is a self-check that fails on a flaky day for no reason.
    real = RES.results_for
    RES.results_for = lambda day, tour="atp": []
    try:
        assert settle(pd.DataFrame([pick.to_dict()]), date(2026, 8, 12),
                      dry_run=True) == []
    finally:
        RES.results_for = real

    # Running PnL keeps the two schemes separate and counts open positions.
    book = pd.DataFrame([
        {"row_type": "pick", "stake_flat": "2", "stake_kelly": "3",
         "result": "", "edge_raw": "0.04"},
        {"row_type": "pick", "stake_flat": "2", "stake_kelly": "1",
         "result": "", "edge_raw": "0.04"},
        {"row_type": "settle", "stake_flat": "2", "stake_kelly": "3",
         "pnl_flat": "1.8", "pnl_kelly": "2.7", "result": "win",
         "edge_raw": "0.05"},
        # A void: stake returned, so it settles but must not dilute any ROI.
        {"row_type": "settle", "stake_flat": "2", "stake_kelly": "3",
         "pnl_flat": "0.0", "pnl_kelly": "0.0", "result": "void",
         "edge_raw": "0.99"},
    ])
    # A curated alias resolves; an alias pointing at an id the match table
    # does not contain must fail to a skip, never to the wrong player.
    ref = Reference({"c": [(frozenset({"alcaraz", "garfia"}), "A0E2")]},
                    {"canada masters": ("M001", "Hard")},
                    {"nadal r": "N409", "typo": "NOT_AN_ID"}, {"A0E2", "N409"},
                    {"atp montreal": "Canada Masters",
                     "atp nowhere": "No Such Masters"})
    # The bookmaker's city name resolves through the alias file; an alias
    # naming a tournament the history lacks must skip, not price blind.
    assert ref.tournament("ATP Montreal") == ("M001", "Hard")
    assert ref.tournament("ATP Nowhere") is None
    assert ref.player("Nadal R.") == "N409"
    assert ref.player("typo") is None
    assert ref.player("Alcaraz Garfia C.") == "A0E2"
    assert ref.player("Someone Unknown X.") is None
    assert ref.candidates("Alcaraz Garfia C.")[0]["player_id"] == "A0E2"

    totals = pnl(book)
    assert totals["n_settled"] == 2 and totals["n_open"] == 0, totals
    # Relative to BANKROLL_START so the check follows the constant rather
    # than pinning a number that changes whenever the scale is retuned.
    assert abs(totals["flat_bankroll"] - (BANKROLL_START + 1.8)) < 1e-9, totals
    assert abs(totals["kelly_bankroll"] - (BANKROLL_START + 2.7)) < 1e-9, totals
    # ROI is over the one resolved bet, not the void: 1.8/2, not 1.8/4. The
    # void's absurd 99% edge must not reach EV either — it would be the
    # loudest number in the Slack post and it would be meaningless.
    assert abs(totals["flat_roi"] - 0.9) < 1e-9, totals
    assert abs(totals["kelly_roi"] - 0.9) < 1e-9, totals
    assert abs(totals["ev_roi"] - 0.05) < 1e-9, totals

    # Board rows are measurement only. Adding them — including settled ones
    # carrying a result — must not move PnL, ROI, bankroll or the open count
    # by any amount. If this ever fires, unstaked projections are leaking
    # into the money numbers.
    with_board = pd.concat([book, pd.DataFrame([
        {"row_type": "board", "stake_flat": 0.0, "stake_kelly": 0.0,
         "result": "", "edge_raw": "0.5", "model_p": "0.7",
         "decimal_odds": "1.9"},
        {"row_type": "board_settle", "stake_flat": 0.0, "stake_kelly": 0.0,
         "pnl_flat": 0.0, "pnl_kelly": 0.0, "result": "win",
         "edge_raw": "0.5", "model_p": "0.7", "decimal_odds": "1.9",
         "market_p_devig": "0.52"},
    ])], ignore_index=True)
    assert pnl(with_board) == totals, pnl(with_board)

    # And the calibration score reads those board rows, not the staked ones.
    sc = calibration_scores(with_board)
    assert sc["n"] == 1, sc
    assert abs(sc["brier"] - (0.7 - 1.0) ** 2) < 1e-9, sc
    # The book is scored on 0.52, its de-vigged number — NOT on 1/1.9 = 0.526,
    # which is the price with its margin still in it.
    assert abs(sc["brier_book"] - (0.52 - 1.0) ** 2) < 1e-9, sc
    assert calibration_scores(book)["n"] == 0, "picks must not be scored"

    # A row written before the de-vig was stored still scores, off the raw
    # price. Old rows must keep parsing, not drop out of the denominator.
    legacy = with_board.copy()
    legacy.loc[legacy["row_type"] == "board_settle", "market_p_devig"] = ""
    assert calibration_scores(legacy)["n"] == 1
    assert abs(calibration_scores(legacy)["brier_book"]
               - (1 / 1.9 - 1.0) ** 2) < 1e-9

    # Closing capture: a pick still on the board on a LATER run gets a second
    # price; the same run must not, or the "close" is just the open again.
    board_rec = {"match_link": "pointsbet:1", "match_date": "2026-08-12T16:00:00Z",
                 "home_team": "A", "away_team": "B",
                 "total_games_market": [
                     {"bookmaker_name": "pointsbet", "submarket_name": "22.5",
                      "odds_over": 1.75, "odds_under": 2.05}]}
    open_pick = {**pick.to_dict(), "row_type": "pick", "run_date": "2026-08-11",
                 "line": "22.5", "book_side": "over", "bookmaker": "pointsbet",
                 "decimal_odds": "1.90", "model_p": "0.60", "market": "total_games"}
    made = closing_rows(pd.DataFrame([open_pick]), {"pointsbet:1": board_rec},
                        date(2026, 8, 12))
    assert len(made) == 1, made
    got = made[0]
    assert got["row_type"] == "close", got
    assert got["open_decimal_odds"] == 1.90 and got["decimal_odds"] == 1.75, got
    # It is a price observation, not a position: it can never reach PnL.
    assert got["stake_flat"] == 0.0 and got["stake_kelly"] == 0.0, got
    assert pnl(pd.DataFrame([got]).reindex(
        columns=LEDGER_COLUMNS))["n_settled"] == 0
    # Same run as the pick: nothing captured.
    assert closing_rows(pd.DataFrame([{**open_pick, "run_date": "2026-08-12"}]),
                        {"pointsbet:1": board_rec}, date(2026, 8, 12)) == []
    # Already captured: never a second close row for the same bet.
    assert closing_rows(
        pd.DataFrame([open_pick, {**got, "row_type": "close"}]),
        {"pointsbet:1": board_rec}, date(2026, 8, 12)) == []
    # The rung is gone: no close rather than the nearest thing to it.
    moved = {**board_rec, "total_games_market": [
        {"bookmaker_name": "pointsbet", "submarket_name": "23.5",
         "odds_over": 1.75, "odds_under": 2.05}]}
    assert closing_rows(pd.DataFrame([open_pick]), {"pointsbet:1": moved},
                        date(2026, 8, 12)) == []

    # The mid-run guard: an unchanged model passes, a moved one stops the run
    # rather than quietly appending rows that measure something else. The
    # first run has no fingerprint yet and must not be blocked by one.
    fp = model_fingerprint()
    check_unchanged({})
    check_unchanged({"start_date": "2026-08-10", "fingerprint": fp})
    try:
        check_unchanged({"start_date": "2026-08-10",
                         "fingerprint": {**fp, "calibration_maps": "stale"}})
    except SystemExit as exc:
        assert "changed mid-run" in str(exc), exc
    else:
        raise AssertionError("a changed calibration map must stop the run")
    print("paper_trade self-check passed")


def preview(dry_run: bool) -> str:
    """Re-render the most recent day's Slack post from the ledger alone.

    For checking how the post READS. It prices nothing, fetches nothing and
    writes nothing — it re-reads rows already committed, so it is safe to run
    against a live run and cannot produce the duplicate ledger rows that
    `deploy/routine_prompt.md` forbids. `--dry-run` prints instead of posting.
    """
    ledger = _load_ledger()
    picks = ledger[ledger["row_type"] == "pick"]
    if picks.empty:
        raise SystemExit("no pick rows in the ledger; nothing to preview")

    day = str(picks["run_date"].max())
    bets = picks[picks["run_date"] == day].to_dict("records")
    for b in bets:
        for k in ("decimal_odds", "model_p", "edge_raw",
                  "stake_flat", "stake_kelly"):
            b[k] = float(b[k] or 0)

    settled = ledger[ledger["row_type"] == "settle"]
    settlements = ([] if settled.empty else
                   settled[settled["run_date"]
                           == settled["run_date"].max()].to_dict("records"))

    board = ledger[(ledger["row_type"] == "board")
                   & (ledger["run_date"] == day)]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    start = state.get("start_date", day)
    run_day = (date.fromisoformat(day) - date.fromisoformat(start)).days + 1

    text = summary(
        date.fromisoformat(day), bets, settlements, {}, pnl(ledger), run_day,
        [{"label": k, "lines": [1]} for k in board["match_key"].unique()],
        calibration_scores(ledger))
    text = f":test_tube: *Format preview — a drill, not today's run*\n{text}"
    post_slack(text, dry_run)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="run the full path without writing the ledger "
                             "or posting to Slack")
    parser.add_argument("--date", type=date.fromisoformat,
                        help="treat this as today (default: today, UTC)")
    parser.add_argument("--self-check", action="store_true",
                        help="run the arithmetic self-check and exit")
    parser.add_argument("--rehearse", action="store_true",
                        help="drive the whole chain on a cached historic "
                             "match; writes nothing, posts nothing")
    parser.add_argument("--preview", action="store_true",
                        help="re-post the latest day's summary from the "
                             "ledger to check formatting; writes nothing")
    parser.add_argument("--variant", default="primary", choices=sorted(VARIANTS),
                        help="which model to price with, and therefore which "
                             "ledger to write (default: primary)")
    args = parser.parse_args()
    select_variant(args.variant)
    if args.self_check:
        demo()
        return
    if args.preview:
        print(preview(args.dry_run))
        return
    if args.rehearse:
        rehearse()
        return
    text = run(args.date or datetime.now(timezone.utc).date(), args.dry_run)
    print("\n" + text)


if __name__ == "__main__":
    main()
