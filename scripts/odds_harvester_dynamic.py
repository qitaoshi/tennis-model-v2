"""Run OddsHarvester after discovering the lines rendered for each match.

OddsHarvester's CLI accepts only concrete tennis line tokens. Its internal
extractor can discover the visible submarkets first; this wrapper uses that
full-scrape path, then asks the extractor for bookmaker rows on those lines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from oddsharvester.core.odds_portal_market_extractor import (
    OddsPortalMarketExtractor,
)
from oddsharvester.core.scraper_app import run_scraper
from oddsharvester.utils.command_enum import CommandEnum

# ponytail: library default (5000ms) times out on H2H fragment resolution for
# players with a long season history (deep-round AO 2024 matches saw ~50%
# failure); 15000ms clears those without a fork. Revisit if upstream fixes
# oddsharvester#60.
import oddsharvester.core.base_scraper as _base_scraper
_base_scraper.H2H_FRAGMENT_RESOLVE_TIMEOUT_MS = 15000

_ORIGINAL = OddsPortalMarketExtractor.scrape_markets


def _line_token(name: str, family: str) -> str | None:
    if family == "set_betting":
        m = re.search(r"(\d+)\s*:\s*(\d+)", name)
        return f"correct_score_{m.group(1)}_{m.group(2)}" if m else None
    m = re.search(r"([+-]?\d+(?:[.,]\d+)?)\s*(Games|Sets)", name,
                  flags=re.IGNORECASE)
    if not m:
        return None
    # OddsHarvester tennis markets are half-lines only (39_5, not 39).
    raw = m.group(1).lstrip("+").replace(",", ".")
    if "." not in raw:
        return None
    number = raw.replace(".", "_")
    axis = m.group(2).lower()
    if family == "total_games" and axis == "games":
        return f"over_under_games_{number}"
    if family == "total_sets" and axis == "sets":
        return f"over_under_sets_{number}"
    if family == "games_handicap" and axis == "games":
        return f"asian_handicap_{number}_games"
    return None


async def _dynamic_scrape(self, page, sport, markets, period="FullTime",
                          scrape_odds_history=False, target_bookmaker=None,
                          preview_submarkets_only=False):
    period = period or "FullTime"
    if sport != "tennis":
        return await _ORIGINAL(
            self, page, sport, markets, period, scrape_odds_history,
            target_bookmaker, preview_submarkets_only)
    expanded: list[str] = []
    main_market = {
        "total_games": "Over/Under",
        "total_sets": "Over/Under",
        "games_handicap": "Asian Handicap",
        "set_betting": "Correct Score",
    }
    for family in markets:
        if family == "match_winner":
            expanded.append(family)
            continue
        if family not in main_market:
            continue
        names = await self._discover_line_names(
            page, main_market[family], sport, period)
        for name in names:
            token = _line_token(name, family)
            if token and token not in expanded:
                expanded.append(token)
    return await _ORIGINAL(
        self, page, sport, expanded, period, scrape_odds_history,
        target_bookmaker, preview_submarkets_only)


OddsPortalMarketExtractor.scrape_markets = _dynamic_scrape


async def _run(args: argparse.Namespace) -> None:
    common = dict(
        sport="tennis",
        markets=[args.family],
        target_bookmaker=args.target_bookmaker,
        headless=True,
        preview_submarkets_only=False,
        bookies_filter="all",
        request_delay=1.0,
        concurrency_tasks=1,
    )
    if args.upcoming:
        # Fixtures that have not been played. Same market discovery, different
        # OddsPortal page family; `date` is YYYYMMDD in OddsPortal's own terms.
        result = await run_scraper(
            command=CommandEnum.UPCOMING_MATCHES,
            date=args.date,
            leagues=[args.league] if args.league else None,
            links_only=args.links_only,
            **common,
        )
    else:
        # A match_links scrape short-circuits before the league/season path in
        # scraper_app, so settlement re-scrapes need the link alone.
        if not (args.match_link or (args.league and args.season)):
            raise SystemExit("historic mode needs --match-link, or --league and --season")
        result = await run_scraper(
            command=CommandEnum.HISTORIC,
            match_links=[args.match_link] if args.match_link else None,
            leagues=[args.league] if args.league else None,
            seasons=[args.season] if args.season else None,
            **common,
        )
    Path(args.output).write_text(json.dumps(result.success if result else [],
                                            indent=2))
    if not result or not result.success:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league")
    parser.add_argument("--season")
    parser.add_argument("--match-link")
    parser.add_argument("--family", required=True)
    parser.add_argument("--target-bookmaker")
    parser.add_argument("--output", required=True)
    parser.add_argument("--upcoming", action="store_true",
                        help="scrape unplayed fixtures instead of history")
    parser.add_argument("--date", help="YYYYMMDD, upcoming mode only")
    parser.add_argument("--links-only", action="store_true",
                        help="discover fixture links without odds")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
