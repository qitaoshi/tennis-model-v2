"""PointsBet AU odds source, shaped as OddsPortal records.

Why this exists: OddsPortal sits behind Cloudflare and blocks headless
Chromium from a datacentre IP, which is exactly where the paper-trading
routine runs. On 2026-08-10 the forward test's day 1 collected nothing —
`net::ERR_CONNECTION_RESET` on every navigation, twice. PointsBet AU
publishes an unofficial JSON API that answers plain `urllib` from the same
host in ~0.14s, with no browser and no fingerprint to defeat.

Ported from the v1 model's `data/fetch_odds.py` (endpoints documented by
declanwalpole/sportsbook-odds-scraper; referenced, not vendored). What is new
here is the `Handicap Games` market, which v1 never used, and the OddsPortal
record shape.

**This is one AU book, not a consensus.** OddsPortal quoted many bookmakers
and the harness picked Bet365; every price here is PointsBet's. That changes
what "the line" means relative to the 2024 data every report in this repo is
written against — it is a different measurement, not a drop-in equal. Say so
when reporting a number that came through this source.

Records come out in the shape `scripts.odds_portal_table._rows` already
parses, so de-vig, name resolution, pricing, staking and the ledger are
untouched by the source swap. Two fields are deliberately absent:

* `partial_results` — the pre-match endpoints carry no finished score, so
  this source cannot settle. Settlement stays on its own path.
* any bookmaker but PointsBet — `bookmaker_name` is constant, so the
  harness's "best book" selection collapses to the only book there is.

Run:  python -m scripts.fetch_pointsbet --date 2026-08-11
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

COMPETITIONS_URL = "https://api.pointsbet.com/api/v2/sports/tennis/competitions"
EVENTS_URL = ("https://api.pointsbet.com/api/v2/competitions/{key}/events/"
              "featured?includeLive=false")
EVENT_URL = "https://api.pointsbet.com/api/mes/v3/events/{key}"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
BOOKMAKER = "pointsbet"

# PointsBet's market names for the two families this project prices. Set,
# player-total and correct-score markets carry different eventNames and are
# ignored: the paper harness stakes total_games and games_handicap only.
TOTALS_MARKET = "total games over/under"
HANDICAP_MARKET = "handicap games"

# Politeness. The v1 module used the same pacing under PROVENANCE terms and
# was never rate-limited; there is no reason to scrape this harder.
PAUSE_S = 1.0


def fetch_json(url: str, retries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 429/5xx are transient; back off hard rather than hammering.
            if e.code not in (429, 500, 502, 503, 504) or i == retries - 1:
                raise
            time.sleep(30 * 2 ** i)
    raise RuntimeError(f"unreachable: {url}")


def odds_portal_name(name: str) -> str:
    """PointsBet's 'Last, First' -> OddsPortal's 'Last F.'.

    The target is deliberately OddsPortal's display form, not 'First Last'.
    `Reference.player` resolves through `_keys(..., odds_style=True)`, which
    reads the FIRST token as the surname and a trailing 'A.' as the given
    initial. Handing it 'Arthur Fils' makes 'arthur' the surname and resolves
    nothing; handing it 'Fils, Arthur' is worse, because 'Fils,' still parses
    and the initial comes out of the wrong word entirely.

    Keeping this format also means the committed `paper/player_aliases.json`
    keys — written against OddsPortal spellings — stay valid across the source
    swap, rather than every alias silently ceasing to match.
    """
    parts = [p.strip() for p in str(name).split(",", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return str(name).strip()
    return f"{parts[0]} {parts[1][0].upper()}."


def _outcomes(market: dict) -> list[dict]:
    return [o for o in market.get("outcomes", [])
            if not o.get("isHidden") and o.get("isOpenForBetting")]


def _price(outcome: dict) -> float | None:
    try:
        price = float(outcome["price"])
    except (KeyError, TypeError, ValueError):
        return None
    # A quote of 1.0 or below pays nothing and is a data error, not a price.
    return price if price > 1.0 else None


def parse_totals(market: dict) -> list[dict]:
    """`Total Games Over/Under` -> one OddsPortal over/under entry per line."""
    by_line: dict[float, dict] = {}
    for o in _outcomes(market):
        side = str(o.get("name", "")).split()[:1]
        side = side[0].casefold() if side else ""
        if side not in ("over", "under"):
            continue
        price = _price(o)
        if price is None or o.get("points") is None:
            continue
        line = float(o["points"])
        by_line.setdefault(line, {})[f"odds_{side}"] = price
    return [{"bookmaker_name": BOOKMAKER, "submarket_name": f"{line:g}",
             "odds_over": prices.get("odds_over"),
             "odds_under": prices.get("odds_under")}
            for line, prices in sorted(by_line.items())]


def parse_handicap(market: dict, home_team: str,
                   away_team: str) -> list[dict]:
    """`Handicap Games` -> one OddsPortal handicap entry per line.

    PointsBet names each outcome by PLAYER ('Rafael Jodar +0.5'), while
    OddsPortal keys the line to the HOME player's handicap and gives the away
    price beside it. So the two sides of one bet appear under opposite
    `points` values (+0.5 home / -0.5 away), and they must be paired by the
    home player's number or the ladder comes out with every rung one-sided.

    The player is matched on the full name PointsBet prints for the event's
    own home/away teams. An outcome naming neither is dropped rather than
    guessed at — an unrecognised name here would otherwise land on whichever
    side was tested first and stake the wrong player.
    """
    home_key, away_key = _name_tokens(home_team), _name_tokens(away_team)
    by_line: dict[float, dict] = {}
    for o in _outcomes(market):
        price = _price(o)
        if price is None or o.get("points") is None:
            continue
        name = str(o.get("name", ""))
        # Strip the trailing signed handicap to leave the player's name.
        who = _name_tokens(re.sub(r"[+-]?\d+(?:\.\d+)?\s*$", "", name))
        points = float(o["points"])
        if who == home_key:
            by_line.setdefault(points, {})["games_handicap_player_1"] = price
        elif who == away_key:
            # Quoted from the home player's side of the same line.
            by_line.setdefault(-points, {})["games_handicap_player_2"] = price
    return [{"bookmaker_name": BOOKMAKER, "submarket_name": f"{line:g}",
             **prices} for line, prices in sorted(by_line.items())]


def _name_tokens(text: object) -> frozenset[str]:
    """Name as an unordered token set.

    PointsBet writes the same player two ways in one payload: 'Jodar, Rafael'
    as the team, 'Rafael Jodar' inside a handicap outcome. Comparing the
    strings — or any order-preserving normalisation of them — matches neither,
    which silently drops every handicap line rather than failing loudly.
    """
    return frozenset(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def event_record(detail: dict, competition: str) -> dict:
    """One PointsBet event as an OddsPortal-shaped record."""
    home = str(detail.get("homeTeam", ""))
    away = str(detail.get("awayTeam", ""))
    totals: list[dict] = []
    handicaps: list[dict] = []
    for market in detail.get("fixedOddsMarkets", []):
        if not market.get("isOpenForBetting"):
            continue
        name = str(market.get("eventName", "")).casefold()
        if name == TOTALS_MARKET:
            totals += parse_totals(market)
        elif name == HANDICAP_MARKET:
            handicaps += parse_handicap(market, home, away)
    return {
        "match_link": f"pointsbet:{detail.get('key')}",
        "match_date": detail.get("startsAt"),
        # The harness filters on 'atp' in the league name and excludes
        # Challengers by the same string, so the competition name is passed
        # through as PointsBet writes it rather than being normalised here.
        "league_name": competition,
        "home_team": odds_portal_name(home),
        "away_team": odds_portal_name(away),
        "over_under_games_market": totals,
        "asian_handicap_market": handicaps,
    }


# Competition names that are ATP-branded but are not main-tour singles
# matches. The harness's own filter is `"atp" in name and "challenger" not in
# name`, which OddsPortal's naming happened to satisfy — PointsBet's does not.
# 'ATP Montreal Doubles' and 'ATP Montreal Futures' both pass that test and
# would be priced by a singles model as though two players had walked on
# court. Outrights are not a match at all.
NON_SINGLES = ("challenger", "doubles", "futures", "outright", "wta", "itf")


def is_main_tour_singles(name: object) -> bool:
    lowered = str(name).casefold()
    return "atp" in lowered and not any(w in lowered for w in NON_SINGLES)


def is_doubles(event_name: object) -> bool:
    """True for 'Arribage T / Olivetti A v Marozsan F / Medvedev D'.

    A slash on either side of the ' v ' is PointsBet's pairing notation. No
    singles player's printed name contains one.
    """
    return "/" in str(event_name)


def competitions() -> list[dict]:
    payload = fetch_json(COMPETITIONS_URL)
    out, seen = [], set()
    for locale in payload.get("locales", []):
        for comp in locale.get("competitions", []):
            # A competition appears under more than one locale; the events
            # behind the same key are the same events.
            if comp.get("key") in seen or not comp.get("name"):
                continue
            seen.add(comp["key"])
            out.append(comp)
    return out


def fetch_records(days: set[date] | None = None,
                  verbose: bool = True) -> list[dict]:
    """Every open ATP event starting on ``days``, as OddsPortal records.

    ``days`` is matched against the event's UTC start date. A match already
    under way is excluded by the endpoint itself (`includeLive=false`), so
    nothing here can price a match in progress.
    """
    records = []
    seen_events: set[str] = set()
    for comp in competitions():
        if not is_main_tour_singles(comp["name"]):
            continue
        name = str(comp["name"])
        time.sleep(PAUSE_S)
        try:
            payload = fetch_json(EVENTS_URL.format(key=comp["key"]))
        except urllib.error.URLError as exc:
            print(f"WARNING: events fetch failed for {name}: {exc}")
            continue
        for event in payload.get("events", []):
            key = str(event.get("key", ""))
            if not key or key in seen_events:
                continue
            starts = _start_date(event.get("startsAt"))
            if days is not None and starts not in days:
                continue
            # A doubles pairing and an outright can appear inside a singles
            # competition, where the competition-level filter never sees them.
            # 'A / B v C / D' is two players a side; the pricer takes one.
            if is_doubles(event.get("name")):
                continue
            seen_events.add(key)
            time.sleep(PAUSE_S)
            try:
                detail = fetch_json(EVENT_URL.format(key=key))
            except urllib.error.URLError as exc:
                print(f"WARNING: event fetch failed for {key}: {exc}")
                continue
            record = event_record(detail, name)
            if record["over_under_games_market"] or record["asian_handicap_market"]:
                records.append(record)
            elif verbose:
                print(f"  no open totals/handicap markets: {event.get('name')}")
    return records


def _start_date(stamp: object) -> date | None:
    try:
        return datetime.fromisoformat(
            str(stamp).replace("Z", "+00:00")).astimezone(timezone.utc).date()
    except (TypeError, ValueError):
        return None


def demo() -> None:
    """Self-check on the parsing that decides which side gets staked."""
    assert odds_portal_name("Fils, Arthur") == "Fils A."
    assert odds_portal_name("Merida Aguilar, Daniel") == "Merida Aguilar D."
    # Already-flat names and empty halves pass through rather than crashing.
    assert odds_portal_name("Carlos Alcaraz") == "Carlos Alcaraz"
    assert odds_portal_name("Fils,") == "Fils,"

    totals = parse_totals({"outcomes": [
        {"name": "Over 22.5", "points": 22.5, "price": 1.85,
         "isOpenForBetting": True},
        {"name": "Under 22.5", "points": 22.5, "price": 1.9,
         "isOpenForBetting": True},
        # Hidden, suspended and unpayable quotes must never reach a ladder.
        {"name": "Over 21.5", "points": 21.5, "price": 1.65,
         "isOpenForBetting": True, "isHidden": True},
        {"name": "Under 23.5", "points": 23.5, "price": 1.0,
         "isOpenForBetting": True},
        {"name": "Over 24.5", "points": 24.5, "price": 2.1,
         "isOpenForBetting": False},
    ]})
    assert totals == [{"bookmaker_name": BOOKMAKER, "submarket_name": "22.5",
                       "odds_over": 1.85, "odds_under": 1.9}], totals

    # The two sides of one handicap arrive under opposite `points` and must
    # pair onto the home player's line, not split into two one-sided rungs.
    hc = parse_handicap({"outcomes": [
        {"name": "Rafael Jodar +0.5", "points": 0.5, "price": 1.71,
         "isOpenForBetting": True},
        {"name": "Arthur Fils -0.5", "points": -0.5, "price": 2.0,
         "isOpenForBetting": True},
        {"name": "Rafael Jodar -1.5", "points": -1.5, "price": 1.9,
         "isOpenForBetting": True},
        {"name": "Arthur Fils +1.5", "points": 1.5, "price": 1.77,
         "isOpenForBetting": True},
        # A name matching neither player is dropped, never guessed onto a side.
        {"name": "Someone Else -2.5", "points": -2.5, "price": 1.5,
         "isOpenForBetting": True},
    ]}, "Jodar, Rafael", "Fils, Arthur")
    assert hc == [
        {"bookmaker_name": BOOKMAKER, "submarket_name": "-1.5",
         "games_handicap_player_1": 1.9, "games_handicap_player_2": 1.77},
        {"bookmaker_name": BOOKMAKER, "submarket_name": "0.5",
         "games_handicap_player_1": 1.71, "games_handicap_player_2": 2.0},
    ], hc

    # End to end: a record must flatten into two-sided quotes the harness's
    # own de-vig accepts, under the market names it prices.
    from scripts.odds_portal_table import _rows
    record = event_record({
        "key": "1", "startsAt": "2026-08-11T20:00:00Z",
        "homeTeam": "Jodar, Rafael", "awayTeam": "Fils, Arthur",
        "fixedOddsMarkets": [
            {"eventName": "Total Games Over/Under", "isOpenForBetting": True,
             "outcomes": [
                 {"name": "Over 22.5", "points": 22.5, "price": 1.85,
                  "isOpenForBetting": True},
                 {"name": "Under 22.5", "points": 22.5, "price": 1.9,
                  "isOpenForBetting": True}]},
            {"eventName": "Handicap Games", "isOpenForBetting": True,
             "outcomes": [
                 {"name": "Rafael Jodar +0.5", "points": 0.5, "price": 1.71,
                  "isOpenForBetting": True},
                 {"name": "Arthur Fils -0.5", "points": -0.5, "price": 2.0,
                  "isOpenForBetting": True}]},
            # A market this project does not price must not leak through.
            {"eventName": "Set Betting", "isOpenForBetting": True,
             "outcomes": [{"name": "2-0", "points": None, "price": 2.5,
                           "isOpenForBetting": True}]},
        ]}, "ATP Montreal")
    assert record["home_team"] == "Jodar R.", record["home_team"]
    flat = _rows(record)
    assert {r["market"] for r in flat} == {"total_games", "games_handicap"}, flat
    # Both sides of both lines survived, so each vig group can be normalised.
    groups = {}
    for r in flat:
        groups.setdefault(r["vig_group"], set()).add(r["side"])
    assert all(len(s) == 2 for s in groups.values()), groups
    assert {r["line"] for r in flat} == {"22.5", "0.5"}, flat

    # Only main-tour singles is ours to price. Every one of these names was
    # live on PointsBet on 2026-08-10 and all but the first would have
    # reached a singles pricer under the harness's own 'atp not challenger'
    # filter.
    assert is_main_tour_singles("ATP Montreal")
    assert not is_main_tour_singles("ATP Montreal Doubles")
    assert not is_main_tour_singles("ATP Montreal Futures")
    assert not is_main_tour_singles("ATP Challenger Hamburg")
    assert not is_main_tour_singles("WTA Cincinnati")
    assert is_doubles("Arribage T / Olivetti A v Marozsan F / Medvedev D")
    assert not is_doubles("Jodar, Rafael v Fils, Arthur")
    print("fetch_pointsbet self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, action="append",
                        help="UTC start date to keep; repeatable "
                             "(default: every open ATP event)")
    parser.add_argument("--output", type=Path,
                        help="write the records as JSON here")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        demo()
        return
    records = fetch_records(set(args.date) if args.date else None)
    print(f"{len(records)} record(s)")
    for r in records:
        print(f"  {r['home_team']} v {r['away_team']}  ({r['league_name']}, "
              f"{r['match_date']})  "
              f"{len(r['over_under_games_market'])} totals / "
              f"{len(r['asian_handicap_market'])} handicap lines")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
