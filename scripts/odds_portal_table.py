"""Flatten raw OddsHarvester JSON and de-vig within each bookmaker."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from model import constants as C
from scripts.fetch_odds_portal import MARKETS, _cache_path

BET365_NAME = "bet365"  # confirmed by reports/odds_portal_recon.md


def _number(text: object) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(text))
    return float(m.group()) if m else None


def _market_name(key: str) -> str:
    prefixes = {
        "match_winner": "match_winner",
        "over_under_games": "total_games",
        "over_under_sets": "total_sets",
        "asian_handicap": "games_handicap",
        "correct_score": "set_betting",
    }
    for prefix, family in prefixes.items():
        if key.startswith(prefix):
            return family
    return key.removesuffix("_market")


def _rows(record: dict) -> list[dict]:
    common = {
        "match_link": record.get("match_link", ""),
        "date": pd.to_datetime(record.get("match_date"), errors="coerce"),
        "tournament": record.get("league_name", ""),
        "player_1": record.get("home_team", ""),
        "player_2": record.get("away_team", ""),
    }
    out: list[dict] = []
    for key, values in record.items():
        if not key.endswith("_market") or not isinstance(values, list):
            continue
        market = _market_name(key)
        for item in values:
            bookmaker = item.get("bookmaker_name")
            if not bookmaker:
                continue
            if market == "match_winner":
                pairs = [("home", item.get("player_1")),
                         ("away", item.get("player_2"))]
                line = ""
            elif market == "total_games" or market == "total_sets":
                pairs = [("over", item.get("odds_over")),
                         ("under", item.get("odds_under"))]
                line = (_number(item.get("submarket_name"))
                        or _number(key))
            elif market == "games_handicap":
                pairs = [("home", item.get("games_handicap_player_1")),
                         ("away", item.get("games_handicap_player_2"))]
                line = (_number(item.get("submarket_name"))
                        or _number(key))
            elif market == "set_betting":
                score = str(item.get("submarket_name", ""))
                if ":" in score:
                    score = score.split()[-1]
                else:
                    m = re.search(r"correct_score_(\d+_\d+)", key)
                    score = m.group(1).replace("_", "-") if m else score
                pairs = [(score, item.get("correct_score"))]
                line = score
            else:
                continue
            for side, raw in pairs:
                odds = pd.to_numeric(raw, errors="coerce")
                if pd.isna(odds) or float(odds) <= 1:
                    continue
                out.append({**common, "market": market,
                            "line": "" if line is None else str(line),
                            "side": side, "bookmaker": bookmaker,
                            "decimal_odds": float(odds),
                            "vig_group": (
                                f"{common['match_link']}|{market}|{line}|"
                                f"{bookmaker}"
                                if market != "set_betting"
                                else f"{common['match_link']}|{market}|"
                                f"{bookmaker}")})
    return out


def flatten(raw_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in raw_paths:
        blob = json.loads(path.read_text())
        rows.extend(_rows(r) for r in blob.get("records", []))
    flat = pd.DataFrame([row for group in rows for row in group])
    if flat.empty:
        return flat
    flat["implied_p"] = 1.0 / flat["decimal_odds"]
    denom = flat.groupby("vig_group")["implied_p"].transform("sum")
    flat["market_p"] = flat["implied_p"] / denom
    flat["bookmaker_tag"] = np.where(
        flat["bookmaker"].eq(BET365_NAME), "bet365",
        np.where(flat["bookmaker"].str.casefold().eq("pinnacle"),
                 "sharp", "other"))
    flat["date"] = flat["date"].dt.date
    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--league")
    parser.add_argument("--season")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = args.input or [
        _cache_path(args.league, args.season, family)
        for family in MARKETS
    ]
    frame = flatten(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        frame.to_parquet(args.output, index=False)
    else:
        frame.to_csv(args.output, index=False)
    if not frame.empty:
        coverage = (frame[frame["bookmaker_tag"] == "bet365"]
                    .groupby("market")["match_link"].nunique())
        print(frame.groupby("market")["match_link"].nunique().to_string())
        print("Bet365 quote counts:")
        print(coverage.to_string())
    print(f"wrote {args.output} ({len(frame):,} rows)")


if __name__ == "__main__":
    main()
