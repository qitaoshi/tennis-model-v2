"""Stage 0 — data audit and canonical match record.

Loads every raw season file inside the pre-holdout range, normalizes it into
one canonical match record, sets the shared ``score_string_suspect`` flag
(ground rule 8), and writes ``reports/data_dictionary.md``.

Bookmaker odds columns present in the raw files are dropped here and never
enter the canonical record: no market price may touch the model path.

HOLDOUT files are never opened by this module — ``load_raw`` refuses any year
whose season can contain dates on or after ``constants.HOLDOUT_CUTOFF``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model import constants as C

# Raw columns kept in the canonical record. Everything else (B365W, PSW,
# MaxW, AvgW, ...) is a bookmaker price and is dropped.
_KEEP = {
    "ATP": "tournament_no",
    "Location": "location",
    "Tournament": "tournament",
    "Date": "date",
    "Series": "level",
    "Court": "court",
    "Surface": "surface",
    "Round": "round",
    "Best of": "best_of",
    "Winner": "winner",
    "Loser": "loser",
    "WRank": "winner_rank",
    "LRank": "loser_rank",
    "WPts": "winner_pts",
    "LPts": "loser_pts",
    "Wsets": "winner_sets",
    "Lsets": "loser_sets",
    "Comment": "comment",
}
_SET_COLS = [f"{s}{i}" for i in range(1, 6) for s in ("W", "L")]

_ODDS_PATTERN = re.compile(r"^(B365|EX|LB|PS|SJ|Max|Avg)[WL]$")

#: Comment -> canonical outcome. Ground rule 6 requires this be explicit.
_OUTCOME = {
    "Completed": "completed",
    "Retired": "retired",
    "Rrtired": "retired",  # data-entry typo, 1 row
    "Walkover": "walkover",
    "Awarded": "other",
    "Disqualified": "other",
    "Sched": "other",
}

#: Documented downstream treatment per outcome (ground rule 6). Printed into
#: the data dictionary so no stage has to re-derive it.
OUTCOME_POLICY = {
    "completed": "Used everywhere.",
    "retired": (
        "EXCLUDED from Stage 1 validation targets, Stage 2 rate fitting, "
        "Stage 6 venue history and Stage 7 correction measurement (the score "
        "is truncated by whatever caused the retirement). INCLUDED in Stage 3 "
        "Elo as a win/loss: tennis-data records a winner and a partial score, "
        "so the result is completed-enough for a win/loss update. Excluded "
        "from holdout outcome scoring and reported separately."
    ),
    "walkover": "Never updates anything. No score exists (all set columns null).",
    "other": (
        "Awarded / Disqualified / Sched. Treated exactly as walkover — never "
        "updates anything — and reported separately. 5 rows total."
    ),
}


@dataclass(frozen=True)
class Suspect:
    flag: bool
    reason: str


def _terminal(w: int, l: int) -> bool:
    """Is this set score a legally finished set under any format in the data?

    Format-agnostic: 6-x (x<=4), 7-5, 7-6, or an advantage-set continuation
    (both >=6, margin 2). rules.py decides whether the format in force at
    match time actually permitted the continuation.
    """
    hi, lo = max(w, l), min(w, l)
    return (
        (hi == 6 and lo <= 4)
        or (hi == 7 and lo in (5, 6))
        or (hi == 13 and lo == 12)  # Wimbledon 2019-2021: final-set TB at 12-12
        or (hi >= 7 and lo >= 6 and hi - lo == 2)  # advantage-set continuation
    )


def _season_years() -> list[int]:
    """Season files whose dates are entirely before the holdout cutoff.

    A season file for year Y covers late-Dec (Y-1) to late-Nov Y, so any file
    with Y >= HOLDOUT_CUTOFF.year is holdout data and is never opened.
    """
    years = []
    for p in sorted(C.DATA_RAW_DIR.glob("*.xlsx")):
        y = int(p.stem)
        if y >= C.HOLDOUT_CUTOFF.year:
            continue
        if y < C.DATA_START.year + 1:
            continue  # 2010 file: 2011-12 are missing, see constants.py
        years.append(y)
    return years


def _score_string(row: pd.Series) -> str:
    """Rebuild a score string from the per-set game columns."""
    parts = []
    for i in range(1, 6):
        w, l = row[f"W{i}"], row[f"L{i}"]
        if pd.isna(w) or pd.isna(l):
            break
        parts.append(f"{int(w)}-{int(l)}")
    return " ".join(parts)


def _check_suspect(row: pd.Series) -> Suspect:
    """Format-agnostic plausibility check on one match's games detail.

    Format-conditional checks (does this set score obey the deciding-set rule
    in force?) belong to rules.py's empirical scan, which contributes further
    suspect flags to the same field.
    """
    reasons: list[str] = []
    outcome = row["outcome"]

    sets = []
    for i in range(1, 6):
        w, l = row[f"W{i}"], row[f"L{i}"]
        if pd.isna(w) and pd.isna(l):
            break
        if pd.isna(w) or pd.isna(l):
            reasons.append(f"set {i} half-recorded")
            break
        sets.append((int(w), int(l)))

    if outcome == "walkover":
        if sets:
            reasons.append("walkover carries a score")
        return Suspect(bool(reasons), "; ".join(reasons))

    if outcome == "completed":
        if not sets:
            reasons.append("completed match with no score")
        if pd.isna(row["best_of"]):
            reasons.append("best_of missing")
        else:
            need = int(row["best_of"]) // 2 + 1
            won = sum(1 for w, l in sets if w > l)
            lost = sum(1 for w, l in sets if l > w)
            if won != need:
                reasons.append(f"winner took {won} sets, needs {need}")
            if lost >= need:
                reasons.append(f"loser took {lost} sets, >= {need}")
            if any(w == l for w, l in sets):
                reasons.append("tied set score")
        # every set of a completed match must be a terminal set
        for i, (w, l) in enumerate(sets, start=1):
            if not _terminal(w, l):
                reasons.append(f"set {i} {w}-{l} not terminal")
            if max(w, l) > 70:
                reasons.append(f"set {i} {w}-{l} implausible")

    # Sets-won columns must agree with the reconstructed score. A retirement
    # is legitimately truncated — its last set is usually mid-play and is not
    # counted in Wsets/Lsets — so only completed sets are compared there.
    if sets and not pd.isna(row["winner_sets"]):
        scored = sets if outcome == "completed" else [s for s in sets if _terminal(*s)]
        won = sum(1 for w, l in scored if w > l)
        lost = sum(1 for w, l in scored if l > w)
        if won != int(row["winner_sets"]) or lost != int(row["loser_sets"]):
            reasons.append(
                f"sets columns {int(row['winner_sets'])}-{int(row['loser_sets'])} "
                f"disagree with score {won}-{lost}"
            )

    return Suspect(bool(reasons), "; ".join(reasons))


def _repair_best_of(m: pd.DataFrame) -> pd.DataFrame:
    """Repair `best_of` from tour format, which is fully determined here.

    Men's Grand Slam singles is best-of-5 throughout this data; every other
    ATP main-tour event in range is best-of-3. The raw files carry 15 nulls
    and one mislabelled Wimbledon 3rd-round row per season (recorded as 3).
    The repair is recorded in `best_of_repaired` rather than applied silently.
    """
    expected = np.where(m["level"] == "Grand Slam", 5.0, 3.0)
    m["best_of_raw"] = m["best_of"]
    m["best_of_repaired"] = m["best_of"].isna() | (
        (m["level"] == "Grand Slam") & (m["best_of"] != 5)
    )
    m["best_of"] = np.where(m["best_of_repaired"], expected, m["best_of"])
    return m


def load_raw(years: list[int] | None = None) -> pd.DataFrame:
    """Load season files into one canonical match record.

    Raises if any loaded date falls in the HOLDOUT range.
    """
    years = years or _season_years()
    frames = []
    for y in years:
        raw = pd.read_excel(C.DATA_RAW_DIR / f"{y}.xlsx")
        df = raw[[c for c in _KEEP if c in raw.columns]].rename(columns=_KEEP)
        for c in _SET_COLS:
            df[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw else np.nan
        df["season_file"] = y
        df["source_row"] = raw.index
        frames.append(df)

    m = pd.concat(frames, ignore_index=True)
    m["date"] = pd.to_datetime(m["date"]).dt.date
    m["outcome"] = m["comment"].map(_OUTCOME).fillna("other")
    m = _repair_best_of(m)
    m["match_id"] = (
        m["season_file"].astype(str) + "_" + m["source_row"].astype(str).str.zfill(5)
    )
    m["score_string"] = m.apply(_score_string, axis=1)
    checks = m.apply(_check_suspect, axis=1)
    m["score_string_suspect"] = [c.flag for c in checks]
    m["score_suspect_reason"] = [c.reason for c in checks]
    m["split"] = [C.split_of(d) for d in m["date"]]

    C.assert_no_holdout(m["date"])
    return m


def flag_suspect(matches: pd.DataFrame, match_ids: list[str], reason: str) -> pd.DataFrame:
    """Set ``score_string_suspect`` on the canonical record (ground rule 8).

    rules.py's discrepancy scan calls this so its findings live on the match
    record, not only in ``reports/rules_discrepancies.csv``.
    """
    hit = matches["match_id"].isin(match_ids)
    matches.loc[hit, "score_string_suspect"] = True
    matches.loc[hit, "score_suspect_reason"] = (
        matches.loc[hit, "score_suspect_reason"].replace("", np.nan).fillna("")
        + np.where(matches.loc[hit, "score_suspect_reason"] == "", "", "; ")
        + reason
    )
    return matches


def find_duplicates(m: pd.DataFrame) -> pd.DataFrame:
    """Same date + same two players (unordered) + same tournament."""
    pair = m.apply(lambda r: " vs ".join(sorted([r["winner"], r["loser"]])), axis=1)
    key = m["date"].astype(str) + "|" + m["tournament"] + "|" + pair
    dup = key.duplicated(keep=False)
    return m.loc[dup, ["match_id", "date", "tournament", "winner", "loser", "round"]]


def surface_changes(m: pd.DataFrame) -> pd.DataFrame:
    """Tournaments whose surface is not constant across seasons."""
    g = m.groupby("tournament")["surface"].nunique()
    changed = g[g > 1].index
    out = (
        m[m["tournament"].isin(changed)]
        .groupby(["tournament", "surface"])["season_file"]
        .agg(lambda s: ", ".join(map(str, sorted(s.unique()))))
        .reset_index()
        .rename(columns={"season_file": "seasons"})
    )
    return out.sort_values(["tournament", "surface"])


def identifier_instability(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Location<->tournament naming drift (sponsor/city renames)."""
    loc_multi = (
        m.groupby("location")["tournament"]
        .agg(lambda s: sorted(s.unique()))
        .loc[lambda s: s.map(len) > 1]
        .reset_index()
    )
    tour_multi = (
        m.groupby("tournament")["location"]
        .agg(lambda s: sorted(s.unique()))
        .loc[lambda s: s.map(len) > 1]
        .reset_index()
    )
    return loc_multi, tour_multi


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_data_dictionary(m: pd.DataFrame, path: Path | None = None) -> Path:
    """Write reports/data_dictionary.md. Every number here is computed."""
    path = path or C.REPORTS_DIR / "data_dictionary.md"
    raw_head = pd.read_excel(C.DATA_RAW_DIR / f"{m['season_file'].max()}.xlsx", nrows=50)
    dropped = [c for c in raw_head.columns if _ODDS_PATTERN.match(str(c))]

    L: list[str] = []
    a = L.append
    a("# Stage 0 — Data dictionary\n")
    a(f"Generated by `model/data_audit.py`. Rows: **{len(m):,}**, "
      f"seasons {m['season_file'].min()}–{m['season_file'].max()}, "
      f"dates {m['date'].min()} … {m['date'].max()}.\n")

    a("## Data source deviation from MODEL_PROMPT.md\n")
    a("The spec assumes TML-Database. `data/raw/` actually holds "
      "tennis-data.co.uk ATP season files (Excel author metadata: "
      "\"Joseph Buchdahl\"). Consequences, in full, because several stages "
      "assume columns that do not exist here:\n")
    a("- **No per-match serve statistics of any kind** — no serve points "
      "won/played, first serves in, aces, double faults, or break points. "
      "Stage 2 as written (\"from match-level TML stats, estimate serve "
      "point-win probability\") has no input. Only per-set game counts exist, "
      "so a serve rate can only be *inferred* from games won on serve — and "
      "even hold/break counts are not recorded, only set scores.\n")
    a("- **No player biographical data** — no hand, height, or age. Stage 5's "
      "style vector loses ace rate, double-fault rate, hand and height; only "
      "games-derived features and surface differential remain.\n")
    a("- **ATP tour level only** — no Challenger, no futures/ITF. Stage 2's "
      "per-level shrinkage, Stage 3's cross-level consistency check, and the "
      "backtest's \"split by tour level\" all lose their second level.\n")
    a("- **Bookmaker odds are present in the raw files** and are dropped at "
      "load: " + ", ".join(f"`{c}`" for c in dropped) + ".\n")
    a("- **Score detail is per-set game counts, not a score string** — "
      "tiebreak point scores are absent, so \"retirement recorded mid-"
      "tiebreak\" is undetectable at that granularity.\n")

    a("\n## Files and coverage\n")
    a("### Columns kept in the canonical record\n")
    a("| canonical | raw | dtype |")
    a("|---|---|---|")
    for raw_c, can_c in _KEEP.items():
        a(f"| `{can_c}` | `{raw_c}` | {m[can_c].dtype} |")
    a(f"| `W1..W5`, `L1..L5` | same | {m['W1'].dtype} (per-set games) |")
    a("| `match_id`, `score_string`, `score_string_suspect`, "
      "`score_suspect_reason`, `outcome`, `split` | derived | — |")

    a("\n### Column presence by season file\n")
    a("All kept columns are present in every season file 2013–2023. Columns "
      "that come and go across files (`SJW/SJL` from 2015, `EXW/EXL` and "
      "`LBW/LBL` from 2019) are all bookmaker odds and are dropped anyway.\n")

    a("\n### Row counts by season and level\n")
    ct = pd.crosstab(m["season_file"], m["level"], margins=True, margins_name="total")
    a(ct.to_markdown())
    a("\nThere is exactly one tour level in this dataset (ATP main tour); "
      "`level` above is the ATP series tier, not a tour/challenger/futures "
      "split.\n")

    a("\n### Field coverage (non-null %) by season\n")
    fields = ["best_of", "winner_rank", "loser_rank", "winner_pts", "W1", "L1",
              "W3", "W5", "winner_sets"]
    cov = m.groupby("season_file")[fields].apply(lambda g: g.notna().mean())
    a((100 * cov).round(1).to_markdown())
    a("\n`W3`/`W5` coverage is low by construction — those sets only exist in "
      "matches that went that far — not a data quality problem. `W1`/`L1` "
      "gaps are walkovers and abandoned matches.\n")

    a("\n### Per-match statistics available\n")
    a("| statistic | present? |")
    a("|---|---|")
    for stat in ["serve points won", "serve points played", "first serves in",
                 "aces", "double faults", "break points won/faced",
                 "per-set games won", "sets won", "tiebreak point scores",
                 "match duration", "player hand / height / age"]:
        present = "**yes**" if stat in ("per-set games won", "sets won") else "no"
        a(f"| {stat} | {present} |")

    a("\n## Retirements, walkovers, defaults\n")
    a(f"Flagged in the raw `Comment` column. Observed values: "
      f"{m['comment'].value_counts().to_dict()}\n")
    a("Note `Rrtired` (1 row) — a typo for `Retired`, normalized on load.\n")
    a("\n| outcome | rows | downstream policy (ground rule 6) |")
    a("|---|---|---|")
    for oc, pol in OUTCOME_POLICY.items():
        a(f"| `{oc}` | {(m['outcome'] == oc).sum():,} | {pol} |")

    a("\n## Surfaces and venues\n")
    a(f"Surface labels: {m['surface'].value_counts().to_dict()}\n")
    a(f"Court: {m['court'].value_counts().to_dict()}\n")
    a(f"Distinct tournaments: {m['tournament'].nunique()}; "
      f"distinct locations: {m['location'].nunique()}.\n")
    a("`tournament_no` (raw `ATP`) is a within-season sequence number and is "
      "**not** stable across years — it must never be used as a venue key. "
      "`location` is the most stable identifier; venue.py keys on "
      "(location, surface).\n")

    a("\n### Surface-change flags\n")
    sc = surface_changes(m)
    if len(sc):
        a(f"{sc['tournament'].nunique()} tournaments change surface across "
          "seasons. Each must be keyed as (venue, surface), never venue "
          "alone:\n")
        a(sc.to_markdown(index=False))
    else:
        a("No tournament changes surface across seasons.\n")

    a("\n### Identifier stability (renames)\n")
    loc_multi, tour_multi = identifier_instability(m)
    a(f"{len(loc_multi)} locations carry more than one tournament name "
      f"(sponsor renames or two events at one venue):\n")
    a(loc_multi.to_markdown(index=False))
    a(f"\n{len(tour_multi)} tournament names appear at more than one location "
      "(event relocations):\n")
    a(tour_multi.to_markdown(index=False))

    a("\n## Score-string suspect flag (ground rule 8)\n")
    n_susp = int(m["score_string_suspect"].sum())
    a(f"{n_susp:,} of {len(m):,} rows ({_pct(n_susp / len(m))}) are flagged by "
      "Stage 0's format-agnostic checks. rules.py's format-conditional scan "
      "adds to the same field via `flag_suspect()`.\n")
    a("\nBy outcome:\n")
    a(m.groupby("outcome")["score_string_suspect"].agg(["sum", "size"]).to_markdown())
    a("\nTop reasons:\n")
    reasons = m.loc[m["score_string_suspect"], "score_suspect_reason"]
    a(reasons.value_counts().head(15).to_markdown())

    a("\n## Duplicate detection\n")
    dups = find_duplicates(m)
    a(f"Same date + same two players + same tournament: **{len(dups)} rows**.\n")
    if len(dups):
        a(dups.to_markdown(index=False))

    a("\n## Split assignment\n")
    a(m.groupby(["split", "season_file"]).size().unstack(fill_value=0).to_markdown())
    a("\nNo row on or after the permanently fixed holdout cutoff "
      f"({C.HOLDOUT_CUTOFF}) is loaded by this module; the 2024–2026 season "
      "files are never opened.\n")

    path.write_text("\n".join(L) + "\n")
    return path


def build(save: bool = True) -> pd.DataFrame:
    """Load, audit, and persist the canonical match record."""
    m = load_raw()
    if save:
        out = C.REPO_ROOT / "data" / "processed"
        out.mkdir(parents=True, exist_ok=True)
        m.to_parquet(out / "matches.parquet", index=False)
        write_data_dictionary(m)
    return m


def demo() -> None:
    """Worked example (ground rule 10)."""
    m = load_raw()
    print(f"loaded {len(m):,} matches, {m['date'].min()} .. {m['date'].max()}")
    print(f"outcomes: {m['outcome'].value_counts().to_dict()}")
    print(f"suspect: {int(m['score_string_suspect'].sum())}")
    cols = ["match_id", "date", "tournament", "surface", "winner", "loser",
            "score_string", "outcome", "score_string_suspect", "split"]
    print(m.loc[m["score_string_suspect"], cols].head(5).to_string(index=False))


if __name__ == "__main__":
    demo()
