"""Final scores for played ATP singles matches, from ESPN's JSON API.

Why this exists: the odds source cannot settle. PointsBet's endpoints carry
`score: null` and the event disappears once the match ends
(`scripts/fetch_pointsbet`), so a pick made through it has no result behind
it. Without a feed like this one the forward test produces timestamped
projections and no PnL at all.

The v1 model used ESPN for the same job under the same terms
(`data/PROVENANCE.md` there: unofficial, undocumented, no key, ToS-gray,
quarantined and used for reconciliation only). Its host has moved since —
`site.api.espn.com` now answers 403 and `site.web.api.espn.com` answers 200.

Scores here settle bets. They are never joined back into the training data,
and nothing in this module writes to `data/processed/`.

What the feed gives that matters:

* per-set `linescores`, so total games and the game margin are exact rather
  than inferred from a set score;
* an explicit `STATUS_RETIRED` / `STATUS_WALKOVER`, so an unfinished match is
  identified by status rather than by parsing "ret" out of a prose note;
* `mens-singles` as its own grouping, so WTA and doubles never reach a
  singles pricer.

Run:  python -m scripts.fetch_results --date 2026-08-10
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import date
from typing import Iterator

SCOREBOARD_URL = ("https://site.web.api.espn.com/apis/site/v2/sports/tennis/"
                  "{tour}/scoreboard?dates={date}")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Only main-tour men's singles. The scoreboard carries WTA and both doubles
# draws under sibling groupings, and a doubles line settled against a singles
# pick would be silently wrong rather than loud.
SINGLES_SLUG = "mens-singles"

# A match that did not finish must void, not settle: a retirement leaves a
# partial score that would settle a totals under as a win for the wrong
# reason. ESPN says so in the status, so this never depends on prose.
FINISHED = "STATUS_FINAL"
UNFINISHED = ("STATUS_RETIRED", "STATUS_WALKOVER", "STATUS_ABANDONED",
              "STATUS_CANCELED", "STATUS_POSTPONED", "STATUS_SUSPENDED")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _competitions(payload: dict) -> Iterator[dict]:
    for event in payload.get("events", []):
        for grouping in event.get("groupings", []):
            slug = str(grouping.get("grouping", {}).get("slug", ""))
            if slug != SINGLES_SLUG:
                continue
            for competition in grouping.get("competitions", []):
                yield competition


def _sets(competitor: dict) -> list[int]:
    """Games won per set by one player.

    `value` is the games total for that set; `tiebreak` is the points in a
    tiebreak and is deliberately ignored — a 7-6 set is seven games to six
    regardless of whether the breaker went 7-5 or 15-13.
    """
    out = []
    for entry in competitor.get("linescores", []):
        try:
            out.append(int(float(entry["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_competition(competition: dict) -> dict | None:
    """One ESPN competition as a settlement record, or None if not a result.

    Returns `finished=False` for a retirement or walkover rather than
    dropping it: the caller needs to tell "this match did not finish, void
    the bet" apart from "this match is not in the feed yet, leave it open".
    """
    status = competition.get("status", {}).get("type", {}) or {}
    name = str(status.get("name", ""))
    if name != FINISHED and name not in UNFINISHED:
        return None  # scheduled or in progress; not a result yet

    players = {}
    for competitor in competition.get("competitors", []):
        side = str(competitor.get("homeAway", ""))
        athlete = competitor.get("athlete", {}) or {}
        display = str(athlete.get("displayName", "")).strip()
        if side not in ("home", "away") or not display:
            continue
        players[side] = {
            "name": display,
            "sets": _sets(competitor),
            "winner": bool(competitor.get("winner")),
        }
    if set(players) != {"home", "away"}:
        return None

    home, away = players["home"], players["away"]
    finished = name == FINISHED
    # Both sides must report the same number of sets for a margin to mean
    # anything. A truncated linescore is treated as unfinished rather than
    # settled on partial games.
    if finished and (not home["sets"]
                     or len(home["sets"]) != len(away["sets"])):
        finished = False

    return {
        "espn_id": str(competition.get("id", "")),
        "status": name,
        "finished": finished,
        "date": str(competition.get("date", ""))[:10],
        "home_name": home["name"],
        "away_name": away["name"],
        "home_sets": home["sets"],
        "away_sets": away["sets"],
        "total_games": sum(home["sets"]) + sum(away["sets"]) if finished else None,
        "game_margin": sum(home["sets"]) - sum(away["sets"]) if finished else None,
        "sets_played": len(home["sets"]) if finished else None,
        "winner": ("home" if home["winner"] else
                   "away" if away["winner"] else ""),
    }


def name_key(name: object) -> frozenset[str]:
    """Surname tokens for joining ESPN names to bookmaker names.

    ESPN prints 'Nicolas Mejia'; the paper harness carries OddsPortal's
    'Mejia N.'. Neither string matches the other, and the given name is an
    initial on one side, so the join is on surname tokens with any single
    letter dropped.

    Two players in the same draw sharing a surname would collide here. The
    caller resolves that by requiring a unique hit and skipping otherwise —
    an ambiguous settlement must never guess which player it settled.
    """
    tokens = re.sub(r"[^a-z]+", " ", str(name).lower()).split()
    return frozenset(t for t in tokens if len(t) > 1)


def results_for(day: date, tour: str = "atp") -> list[dict]:
    """Every finished-or-abandoned ATP singles match on ``day``."""
    url = SCOREBOARD_URL.format(tour=tour, date=day.strftime("%Y%m%d"))
    try:
        payload = fetch_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"ESPN results fetch failed for {day}: {exc}") from exc
    out = []
    for competition in _competitions(payload):
        record = parse_competition(competition)
        if record is not None:
            out.append(record)
    return out


def find_result(results: list[dict], player_a: str,
                player_b: str) -> dict | None:
    """The single result matching both players, or None.

    Requires exactly one match on both surnames. Anything else — no hit, or
    two plausible hits — returns None and the caller leaves the pick open.
    Settling the wrong match is worse than settling nothing.

    Orientation matters and is not assumed: the harness's player A is the
    bookmaker's home player, which need not be ESPN's. `flipped` says whether
    the game margin has to be negated before it is applied to a handicap.
    """
    key_a, key_b = name_key(player_a), name_key(player_b)
    if not key_a or not key_b:
        return None
    hits = []
    for record in results:
        home, away = name_key(record["home_name"]), name_key(record["away_name"])
        if (key_a & home) and (key_b & away):
            hits.append({**record, "flipped": False})
        elif (key_a & away) and (key_b & home):
            hits.append({**record, "flipped": True})
    return hits[0] if len(hits) == 1 else None


def demo() -> None:
    """Self-check on the parsing that decides whether a bet is paid."""
    def competitor(name, sets, winner, tiebreaks=None):
        lines = [{"value": float(v)} for v in sets]
        for i, tb in (tiebreaks or {}).items():
            lines[i]["tiebreak"] = tb
        return {"homeAway": "home" if winner is None else "x",
                "athlete": {"displayName": name},
                "linescores": lines, "winner": bool(winner)}

    final = {
        "id": "1", "date": "2026-08-10T14:40Z",
        "status": {"type": {"name": "STATUS_FINAL"}},
        "competitors": [
            {**competitor("Rafael Jodar", [6, 3, 6], True), "homeAway": "home"},
            {**competitor("Arthur Fils", [4, 6, 1], False), "homeAway": "away"},
        ]}
    got = parse_competition(final)
    assert got["finished"] and got["total_games"] == 26, got
    assert got["game_margin"] == 15 - 11 == 4, got
    assert got["sets_played"] == 3 and got["winner"] == "home", got

    # A tiebreak is six or seven GAMES, never the points in the breaker. If
    # `tiebreak` ever leaked into the total, a 7-6 set would score 13 games
    # as 20 and settle every over on the ladder.
    tb = {**final, "competitors": [
        {**competitor("A Player", [7, 6], True, {0: 7}), "homeAway": "home"},
        {**competitor("B Player", [6, 4], False, {0: 5}), "homeAway": "away"}]}
    assert parse_competition(tb)["total_games"] == 23, parse_competition(tb)

    # A retirement is reported, but never as a settleable score.
    ret = {**final, "status": {"type": {"name": "STATUS_RETIRED"}}}
    got = parse_competition(ret)
    assert got is not None and not got["finished"], got
    assert got["total_games"] is None, got
    walk = {**final, "status": {"type": {"name": "STATUS_WALKOVER"}}}
    assert parse_competition(walk)["finished"] is False

    # Not played yet is not a result at all — the pick stays open.
    sched = {**final, "status": {"type": {"name": "STATUS_SCHEDULED"}}}
    assert parse_competition(sched) is None

    # A truncated linescore must not settle on partial games.
    ragged = {**final, "competitors": [
        {**competitor("A Player", [6, 3], True), "homeAway": "home"},
        {**competitor("B Player", [4], False), "homeAway": "away"}]}
    assert parse_competition(ragged)["finished"] is False

    # Only men's singles is ours. A doubles or WTA grouping in the same
    # payload must not produce a settlement record.
    payload = {"events": [{"groupings": [
        {"grouping": {"slug": "mens-singles"}, "competitions": [final]},
        {"grouping": {"slug": "womens-singles"}, "competitions": [final]},
        {"grouping": {"slug": "mens-doubles"}, "competitions": [final]}]}]}
    assert len(list(_competitions(payload))) == 1

    # The join runs ESPN's 'Nicolas Mejia' against the harness's 'Mejia N.',
    # in both orientations, and refuses anything ambiguous.
    results = [got | {"finished": True, "total_games": 26, "game_margin": 4,
                      "home_name": "Rafael Jodar", "away_name": "Arthur Fils"}]
    hit = find_result(results, "Jodar R.", "Fils A.")
    assert hit and not hit["flipped"], hit
    # Reversed: the same match, but A and B are the other way round, so the
    # margin must be negated by the caller.
    hit = find_result(results, "Fils A.", "Jodar R.")
    assert hit and hit["flipped"], hit
    assert find_result(results, "Jodar R.", "Nobody X.") is None
    # Two players sharing a surname in one draw: ambiguous, so no settlement.
    twins = results + [{**results[0], "espn_id": "2",
                        "home_name": "Ricardo Jodar",
                        "away_name": "Andre Fils"}]
    assert find_result(twins, "Jodar R.", "Fils A.") is None
    print("fetch_results self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, action="append",
                        help="UTC date to fetch; repeatable")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        demo()
        return
    for day in args.date or [date.today()]:
        rows = results_for(day)
        done = [r for r in rows if r["finished"]]
        print(f"{day}: {len(rows)} result(s), {len(done)} finished")
        for r in rows:
            if r["finished"]:
                print(f"  {r['home_name']} v {r['away_name']}: "
                      f"{r['total_games']} games, margin "
                      f"{r['game_margin']:+d}, {r['sets_played']} sets")
            else:
                print(f"  {r['home_name']} v {r['away_name']}: "
                      f"{r['status']} — voids, does not settle")


if __name__ == "__main__":
    main()
