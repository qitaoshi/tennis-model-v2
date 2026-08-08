"""Fetch raw OddsHarvester records with per-match checkpoints.

This is an evaluation-data acquisition script. It deliberately keeps the
scraper's JSON untouched and never passes a bookmaker filter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from model import constants as C

MARKETS = {
    "match_winner": ["match_winner"],
    "total_games": [f"over_under_games_{n}_{half}" for n in range(6, 51)
                    for half in (("5",) if n == 6 else ("0", "5"))],
    "total_sets": [f"over_under_sets_{n}_5" for n in range(2, 11)],
    "games_handicap": (
        [f"asian_handicap_{sign}{n}_5_games"
         for sign in ("-", "+") for n in range(2, 9)]
        + ["asian_handicap_-2_5_sets", "asian_handicap_-2_0_sets",
           "asian_handicap_-1_5_sets", "asian_handicap_-1_0_sets",
           "asian_handicap_-0_5_sets", "asian_handicap_0_sets",
           "asian_handicap_+0_5_sets", "asian_handicap_+1_0_sets",
           "asian_handicap_+1_5_sets", "asian_handicap_+2_0_sets",
           "asian_handicap_+2_5_sets"]
    ),
    "set_betting": [
        "correct_score_2_0", "correct_score_2_1", "correct_score_0_2",
        "correct_score_1_2", "correct_score_3_0", "correct_score_0_3",
        "correct_score_1_3", "correct_score_2_3", "correct_score_3_1",
        "correct_score_3_2", "correct_score_6_0", "correct_score_6_1",
        "correct_score_6_2", "correct_score_6_3", "correct_score_6_4",
        "correct_score_7_5", "correct_score_7_6", "correct_score_0_6",
        "correct_score_1_6", "correct_score_2_6", "correct_score_3_6",
        "correct_score_4_6", "correct_score_5_7", "correct_score_6_7",
    ],
}


def _cache_path(league: str, season: str, family: str) -> Path:
    safe = league.replace("/", "_")
    return C.PROCESSED_DIR / "odds_portal" / f"tennis_{safe}_{season}_{family}.json"


def _run_links(league: str, season: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        cmd = ["oddsharvester", "historic", "-s", "tennis", "-l", league,
               "--season", season, "--links-only", "--format", "json",
               "-o", tmp.name, "--request-delay", "1", "--concurrency", "1",
               "--headless"]
        subprocess.run(cmd, check=True)
        return json.loads(Path(tmp.name).read_text())


# ponytail: H2H fragment ID resolution is an intermittent OddsPortal race
# (see odds_harvester_dynamic.py:22-27) - same link succeeds or fails across
# runs. Retry a few times with backoff rather than chase a root cause in a
# page we don't control.
_SCRAPE_RETRIES = 3
_SCRAPE_BACKOFF_S = 5.0


def _scrape_one(link: str, league: str, season: str, family: str,
                target_bookmaker: str | None = None) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_SCRAPE_RETRIES):
        if attempt:
            time.sleep(_SCRAPE_BACKOFF_S * attempt)
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            cmd = [sys.executable, "-m", "scripts.odds_harvester_dynamic",
                   "--league", league, "--season", season, "--match-link", link,
                   "--family", family, "--output", tmp.name]
            if target_bookmaker:
                cmd.extend(["--target-bookmaker", target_bookmaker])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                last_exc = RuntimeError(
                    f"exit {result.returncode}\nstdout: {result.stdout[-2000:]}\n"
                    f"stderr: {result.stderr[-2000:]}")
                continue
            rows = json.loads(Path(tmp.name).read_text())
            if len(rows) != 1:
                last_exc = ValueError(
                    f"expected one record for {link}, got {len(rows)}")
                continue
            return rows[0]
    assert last_exc is not None
    raise last_exc


def _load_cache(path: Path) -> tuple[dict, dict[str, dict]]:
    if not path.exists():
        return {}, {}
    blob = json.loads(path.read_text())
    return blob.get("metadata", {}), {
        r["match_link"]: r for r in blob.get("records", [])
        if r.get("match_link")
    }


def _write_cache(path: Path, metadata: dict, records: dict[str, dict],
                 failed: dict[str, str]) -> None:
    payload = {"metadata": metadata, "records": list(records.values()),
               "failed": failed}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)


def fetch(league: str, season: str, families: list[str],
          retry_failed: bool = False, match_links: list[str] | None = None,
          target_bookmaker: str | None = None, concurrency: int = 1) -> None:
    links = ([{"match_link": link, "sport": "tennis", "league": league,
               "season": season} for link in match_links]
             if match_links else _run_links(league, season))
    base = {
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "oddsharvester_version": version("oddsharvester"),
        "sport": "tennis", "league": league, "season": season,
        "full_scrape": True, "bookies_filter": "all",
        "target_bookmaker": target_bookmaker,
    }
    for family in families:
        path = _cache_path(league, season, family)
        path.parent.mkdir(parents=True, exist_ok=True)
        old_meta, records = _load_cache(path)
        old_target = old_meta.get("target_bookmaker")
        if old_meta and old_target != target_bookmaker:
            records = {}
            failed = {}
        metadata = {**old_meta, **base, "market_family": family,
                    "market_args": MARKETS[family]}
        failed: dict[str, str] = {}
        if path.exists():
            if old_target == target_bookmaker:
                failed = json.loads(path.read_text()).get("failed", {})
        pending = [r for r in links if r["match_link"] not in records
                   or (retry_failed and r["match_link"] in failed)]
        started = time.monotonic()
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_scrape_one, link_row["match_link"], league,
                            season, family, target_bookmaker):
                    link_row["match_link"]
                for link_row in pending
            }
            for i, future in enumerate(as_completed(futures), 1):
                link = futures[future]
                try:
                    row = future.result()
                    with lock:
                        records[link] = row
                        failed.pop(link, None)
                    status = "ok"
                except Exception as exc:  # retain failure for selective retry
                    with lock:
                        failed[link] = repr(exc)
                    status = "failed"
                with lock:
                    _write_cache(path, metadata, records, failed)
                elapsed = time.monotonic() - started
                rate = elapsed / i
                eta = rate * (len(pending) - i)
                print(f"{family}: {i}/{len(pending)} {status}; "
                      f"eta {eta / 3600:.2f}h", flush=True)
        print(f"cached {path} ({len(records)} records, "
              f"{len(failed)} failed)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--family", choices=sorted(MARKETS), action="append")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--bet365-only", action="store_true",
        help="scope scrape to exact bookmaker bet365; deviates from the "
             "default all-bookmaker recon/CLV plan")
    parser.add_argument("--match-link", action="append",
                        help="scrape only these links (useful for recon)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="parallel match scrapes; each spawns its own "
                             "headless browser subprocess")
    args = parser.parse_args()
    fetch(args.league, args.season, args.family or list(MARKETS),
          args.retry_failed, args.match_link,
          "bet365" if args.bet365_only else None, args.concurrency)


if __name__ == "__main__":
    main()
