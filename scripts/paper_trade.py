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
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from model import constants as C
from model import price as PZ
from scripts.multi_market_clv import _keys
from scripts.odds_portal_table import _rows

# --- decided, not tunable ---------------------------------------------------
BANKROLL_START = 100.0
FLAT_STAKE = 2.0
KELLY_SCALE = 0.5          # half-Kelly
KELLY_CAP = 5.0            # hard cap, units, per bet
MARKETS = ("total_games", "games_handicap")
RUN_DAYS = 14

PAPER_DIR = C.REPO_ROOT / "paper"
LEDGER = PAPER_DIR / "ledger.csv"
STATE = PAPER_DIR / "run_state.json"
# Curated OddsPortal display name -> model player id. Reviewed and committed,
# so every entry is visible in a diff. Only ever affects FUTURE picks: a past
# ledger row is never revisited because a name was resolved later.
ALIASES = PAPER_DIR / "player_aliases.json"
# Written fresh each run: the names that could not be resolved, with the
# model players that might match. A work list, not a decision.
UNRESOLVED = PAPER_DIR / "unresolved_names.json"

LEDGER_COLUMNS = [
    "row_type", "ts_utc", "run_date", "match_key", "match_link", "league_slug",
    "match_date", "tournament", "player_a", "player_b", "model_id_a",
    "model_id_b", "best_of", "market", "line", "book_side", "model_selection",
    "bookmaker", "decimal_odds", "market_p_devig", "model_p", "edge_raw",
    "edge_devig", "kelly_full", "stake_flat", "stake_kelly", "bankroll_kelly",
    "result", "pnl_flat", "pnl_kelly", "note",
]

# Matches whose model probability sits at the ladder's extremes are priced off
# a handful of pmf cells; a 1% quote is not a claim this harness should stake.
MIN_MODEL_P = 0.05
MAX_MODEL_P = 0.95
# A quote implying a 25pp edge against a market that beats us on discrimination
# is a data error, not an opportunity.
ABSURD_EDGE = 0.25


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def _load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
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
    """Unstarted ATP fixtures on ``days``, with totals and handicap quotes.

    One scrape per date per market family, merged by match link — OddsPortal's
    listing is per-date, so this is a page walk per date rather than a scrape
    per fixture.

    Both today and tomorrow are scanned. OddsPortal does not reliably post a
    full board a day ahead: on 2026-08-10 the next day's tennis page rendered
    zero rows while the same day's rendered three. Scanning only tomorrow
    would silently produce empty days. The listing itself drops matches that
    have already started, so nothing here can bet a match in progress.
    """
    merged: dict[str, dict] = {}
    for day in days:
        for family in MARKETS:
            try:
                # The dated listing is /matches/tennis/YYYYMMDD/ despite the
                # library docstring saying YYYY-MM-DD; the hyphenated form
                # returns an empty document.
                records = _scrape(["--upcoming", "--date",
                                   day.strftime("%Y%m%d"), "--family", family])
            except RuntimeError as exc:
                print(f"WARNING: {family} scrape failed for {day}: {exc}")
                continue
            for rec in records:
                name = str(rec.get("league_name", "")).lower()
                # The date page carries every tour. The model is fitted on ATP
                # main tour; Challenger and WTA fixtures are not ours to price.
                if "atp" not in name or "challenger" in name:
                    continue
                link = rec.get("match_link")
                if not link:
                    continue
                merged.setdefault(link, {}).update(rec)
    return merged


def match_day(record: dict, fallback: date) -> date:
    stamp = pd.to_datetime(record.get("match_date"), errors="coerce", utc=True)
    return fallback if pd.isna(stamp) else stamp.date()


def settle_scrape(link: str) -> dict | None:
    """Re-scrape one played match for its final score."""
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
        return cls(players, tourns, aliases, ids)

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
        """Resolve an OddsPortal league name to (tourney_code, surface)."""
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


def board_lines(priced, quote_frame: pd.DataFrame) -> list[dict]:
    """One representative quote per market, with the model's number beside it.

    This is the reporting counterpart to `pick_bets`: same pairing of a
    priced selection to a real quote, but with none of the edge, staking or
    sanity filters, so a market shows up here whether or not it produced a
    bet. Per market it shows the side the model likes best — the largest
    model-minus-implied gap, negative if that is all there is. Picking by
    price instead would always land on the same side of a two-sided total,
    since both sides sit near even money, and would say nothing about what
    the model actually thinks.
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
                    "model_p": sel.probability,
                    "edge": sel.probability - implied}
            if pick is None or cand["edge"] > pick["edge"]:
                pick = cand
        if pick is not None:
            out.append(pick)
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
            if edge_raw <= 0:
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
            out.append(best)
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


def settle(ledger: pd.DataFrame, today: date, dry_run: bool) -> list[dict]:
    picks = ledger[ledger["row_type"] == "pick"]
    settled = set(ledger[ledger["row_type"] == "settle"]["match_key"] + "|"
                  + ledger[ledger["row_type"] == "settle"]["market"])
    rows = []
    open_picks = [r for _, r in picks.iterrows()
                  if f"{r['match_key']}|{r['market']}" not in settled
                  and str(r["match_date"]) < today.isoformat()]
    by_link: dict[str, list] = {}
    for r in open_picks:
        by_link.setdefault(str(r["match_link"]), []).append(r)

    for link, group in by_link.items():
        record = settle_scrape(link)
        best_of = int(float(group[0].get("best_of") or 3))
        parsed = parse_score(record, best_of) if record else None
        for r in group:
            if parsed is None:
                rows.append({**r.to_dict(), "row_type": "settle",
                             "ts_utc": _now(), "run_date": today.isoformat(),
                             "result": "void", "pnl_flat": 0.0,
                             "pnl_kelly": 0.0,
                             "note": "match did not finish or result "
                                     "unavailable; stake returned"})
                continue
            total, diff, _ = parsed
            result, per_unit = settle_one(r, total, diff)
            rows.append({**r.to_dict(), "row_type": "settle",
                         "ts_utc": _now(), "run_date": today.isoformat(),
                         "result": result,
                         "pnl_flat": per_unit * float(r["stake_flat"]),
                         "pnl_kelly": per_unit * float(r["stake_kelly"]),
                         "note": f"final {total} games, margin {diff:+d}"})
    return rows


# ---------------------------------------------------------------------------
# PnL
# ---------------------------------------------------------------------------


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


def summary(day: date, bets: list[dict], settlements: list[dict],
            skips: dict[str, list[str]], totals: dict, run_day: int,
            board: list[dict] | None = None) -> str:
    lines = [f"*Tennis paper trading — day {run_day} of {RUN_DAYS}* "
             f"(run {day.isoformat()})",
             "_Paper only. Expected outcome is flat-to-negative; this is a "
             "forward measurement, not a strategy._", ""]

    if bets:
        lines.append(f"*{len(bets)} bet(s) placed*")
        for b in bets:
            lines.append(
                f"• {b['player_a']} v {b['player_b']} — {b['market']} "
                f"{b['book_side']} {b['line']} @ {b['decimal_odds']:.2f} "
                f"({b['bookmaker']}) · model {b['model_p']:.1%} vs price "
                f"{1 / b['decimal_odds']:.1%} · edge {b['edge_raw']:+.1%} · "
                f"flat {b['stake_flat']:.2f}u / kelly {b['stake_kelly']:.2f}u")
    else:
        lines.append("*No bets placed.*")
    lines.append("")

    if board:
        lines.append(f"*Board — {len(board)} match(es) priced*")
        for m in board:
            lines.append(f"• {m['label']}  _(bo{m['best_of']}, {m['book']}, "
                         f"{m['quotes']} quotes)_")
            for ln in m["lines"]:
                lines.append(
                    f"    {ln['market']} {ln['side']} {ln['line']} "
                    f"@ {ln['decimal_odds']:.2f} — model {ln['model_p']:.1%} "
                    f"vs {ln['implied']:.1%} ({ln['edge']:+.1%})")
        lines.append("")

    if settlements:
        lines.append(f"*{len(settlements)} position(s) settled*")
        for s in settlements:
            lines.append(
                f"• {s['player_a']} v {s['player_b']} — {s['market']} "
                f"{s['book_side']} {s['line']}: {s['result']} · "
                f"flat {float(s['pnl_flat']):+.2f}u / "
                f"kelly {float(s['pnl_kelly']):+.2f}u")
        lines.append("")

    lines.append(
        f"*Running* — flat: {totals['flat_pnl']:+.2f}u, bankroll "
        f"{totals['flat_bankroll']:.2f}u (ROI {totals['flat_roi']:+.2%}) · "
        f"half-Kelly: {totals['kelly_pnl']:+.2f}u, bankroll "
        f"{totals['kelly_bankroll']:.2f}u (ROI {totals['kelly_roi']:+.2%}) · "
        f"{totals['n_settled']} settled, {totals['n_open']} open")
    lines.append(
        f"_Model claimed {totals['ev_roi']:+.2%} EV on those same bets; "
        f"realised flat is {totals['flat_roi']:+.2%}. The gap is the model "
        f"overrating its own edge, not variance alone — at this sample size "
        f"neither number means much._")

    n_skip = sum(len(v) for v in skips.values())
    if n_skip:
        lines.append(f"\n*{n_skip} match(es) skipped*")
        for reason, names in skips.items():
            shown = ", ".join(names[:4]) + ("…" if len(names) > 4 else "")
            lines.append(f"• {reason}: {len(names)} — {shown}")

    lines.append("\n_Two weeks on two markets is a very small sample. Whatever "
                 "this PnL comes out at, it is dominated by variance and "
                 "cannot show an edge is present or absent._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the daily job
# ---------------------------------------------------------------------------


def run(today: date, dry_run: bool) -> str:
    ledger = _load_ledger()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    start = date.fromisoformat(state.get("start_date", today.isoformat()))
    run_day = (today - start).days + 1

    # 1. settle yesterday first, so today's Kelly bankroll includes it.
    settlements = settle(ledger, today, dry_run)
    _append(settlements, dry_run)
    if settlements and not dry_run:
        ledger = _load_ledger()
    elif settlements:
        ledger = pd.concat([ledger, pd.DataFrame(settlements)
                            .reindex(columns=LEDGER_COLUMNS)],
                           ignore_index=True)
    totals = pnl(ledger)

    # 2. fixtures still to be played, today and tomorrow.
    tomorrow = today + timedelta(days=1)
    found = fixtures([today, tomorrow])
    print(f"{len(found)} unstarted ATP fixture(s) for {today}..{tomorrow}")

    ref = Reference.build(today)
    pricer = PZ.Pricer(today, allow_holdout=True,
                       history_parquet="matches_with_holdout.parquet")

    skips: dict[str, list[str]] = {}
    unresolved: dict[str, dict] = {}
    bets: list[dict] = []
    board: list[dict] = []
    bankroll_kelly = totals["kelly_bankroll"]
    already = set(ledger[ledger["row_type"] == "pick"]["match_key"] + "|"
                  + ledger[ledger["row_type"] == "pick"]["market"])

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
        # shopping and is fiction (`reports/totals_roi.md`).
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
        board.append({
            "label": label, "book": book, "best_of":
                priced.metadata["format"]["best_of"],
            "quotes": len(q), "lines": board_lines(priced, q)})

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

    _append(bets, dry_run)
    if dry_run:
        print(f"[dry-run] would write {len(unresolved)} unresolved name(s) "
              f"to {UNRESOLVED}")
    else:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        UNRESOLVED.write_text(json.dumps(
            {"run_date": today.isoformat(),
             "names": sorted(unresolved.values(), key=lambda r: r["odds_name"])},
            indent=2, ensure_ascii=False) + "\n")
    text = summary(today, bets, settlements, skips, totals, run_day, board)
    post_slack(text, dry_run)

    if not dry_run and not STATE.exists():
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"start_date": today.isoformat(),
                                     "run_days": RUN_DAYS,
                                     "bankroll_start": BANKROLL_START,
                                     "flat_stake": FLAT_STAKE,
                                     "kelly_scale": KELLY_SCALE,
                                     "kelly_cap": KELLY_CAP,
                                     "markets": list(MARKETS)}, indent=2))
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

    priced = PZ.Pricer(as_of).price(id_a, id_b, code, played.year, surface)
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
        "note"} if bets else set()
    assert not missing, missing
    print("rehearsal complete — nothing written, nothing posted")


def demo() -> None:
    """Self-check on the arithmetic that decides money (ground rule 10)."""
    assert abs(kelly(0.6, 2.0) - 0.2) < 1e-9, kelly(0.6, 2.0)
    assert kelly(0.4, 2.0) == 0.0
    assert min(KELLY_SCALE * kelly(0.99, 10.0) * 100.0, KELLY_CAP) == KELLY_CAP

    assert model_selection("total_games", "over", 21.5) == "over 21.5"
    assert model_selection("games_handicap", "home", -4.5) == "A +4.5"
    assert model_selection("games_handicap", "away", -4.5) == "B -4.5"

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
    ref = Reference({"c": [(frozenset({"alcaraz", "garfia"}), "A0E2")]}, {},
                    {"nadal r": "N409", "typo": "NOT_AN_ID"}, {"A0E2", "N409"})
    assert ref.player("Nadal R.") == "N409"
    assert ref.player("typo") is None
    assert ref.player("Alcaraz Garfia C.") == "A0E2"
    assert ref.player("Someone Unknown X.") is None
    assert ref.candidates("Alcaraz Garfia C.")[0]["player_id"] == "A0E2"

    totals = pnl(book)
    assert totals["n_settled"] == 2 and totals["n_open"] == 0, totals
    assert abs(totals["flat_bankroll"] - 101.8) < 1e-9, totals
    assert abs(totals["kelly_bankroll"] - 102.7) < 1e-9, totals
    # ROI is over the one resolved bet, not the void: 1.8/2, not 1.8/4. The
    # void's absurd 99% edge must not reach EV either — it would be the
    # loudest number in the Slack post and it would be meaningless.
    assert abs(totals["flat_roi"] - 0.9) < 1e-9, totals
    assert abs(totals["kelly_roi"] - 0.9) < 1e-9, totals
    assert abs(totals["ev_roi"] - 0.05) < 1e-9, totals
    print("paper_trade self-check passed")


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
    args = parser.parse_args()
    if args.self_check:
        demo()
        return
    if args.rehearse:
        rehearse()
        return
    text = run(args.date or datetime.now(timezone.utc).date(), args.dry_run)
    print("\n" + text)


if __name__ == "__main__":
    main()
