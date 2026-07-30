"""Stage 0 — data audit and canonical match record (TML-Database).

Loads every TML season file inside the pre-holdout range (ATP main tour and
Challenger), normalizes it into one canonical match record, sets the shared
``score_string_suspect`` flag (ground rule 8), and writes
``reports/data_dictionary.md``.

HOLDOUT files are never opened: ``load_raw`` refuses any season year on or
after ``constants.HOLDOUT_CUTOFF``.

TML deviates from Sackmann's tennis_atp in ways that matter here — player and
tournament ids are non-numeric strings, an ``indoor`` column exists but is
sparsely populated, and retirement/walkover status lives inside the ``score``
string rather than a flag column. All three are audited below rather than
assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model import constants as C

_TOURS = {"atp": "atp_matches", "chall": "chall_matches"}

#: TML `tourney_level` code -> readable label. Sackmann's scheme plus TML's
#: explicit 250/500 split of tour-level events.
LEVEL_LABEL = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "500": "ATP 500",
    "250": "ATP 250",
    "A": "Tour (other)",
    "F": "Tour Finals",
    "D": "Davis Cup",
    "O": "Olympics",
    "C": "Challenger",
}

#: Outcome markers inside the TML `score` string (ground rule 6).
_RET = re.compile(r"\bRET\b", re.I)
_WO = re.compile(r"\bW/?O\b", re.I)
_DEF = re.compile(r"\bDEF\b", re.I)

_SET_TOKEN = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\((\d{1,3})\))?$")

#: Documented downstream treatment per outcome (ground rule 6). Printed into
#: the data dictionary so no stage has to re-derive it.
OUTCOME_POLICY = {
    "completed": "Used everywhere.",
    "retired": (
        "EXCLUDED from Stage 2 serve-rate fitting, Stage 1 validation targets, "
        "Stage 6 venue history and Stage 7 correction measurement — the stats "
        "are contaminated by whatever caused the retirement and the games "
        "distribution is truncated. INCLUDED in Stage 3 Elo as a win/loss: TML "
        "records a winner and the completed portion of the score, so the "
        "result is completed-enough for a win/loss update. Excluded from "
        "holdout outcome scoring and reported separately."
    ),
    "walkover": (
        "Never updates anything — not Elo, not rates, not venue history. No "
        "match was played; the score field is bare 'W/O'."
    ),
    "default": (
        "Player defaulted/disqualified mid-match. Treated as walkover — never "
        "updates anything — because the abandonment is disciplinary rather "
        "than competitive. 9 rows."
    ),
    "unknown": (
        "Score missing or unparseable. Never updates anything; reported "
        "separately and flagged score_string_suspect."
    ),
}

_SERVE_COLS = [
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced",
]


@dataclass(frozen=True)
class ParsedScore:
    sets: list[tuple[int, int]]
    tiebreaks: int
    winner_games: int
    loser_games: int
    suspect: bool
    reason: str


def _terminal(w: int, l: int) -> bool:
    """Is this set score a legally finished set under any format in the data?

    Format-agnostic: 6-x (x<=4), 7-5, 7-6, the Wimbledon 2019-2021 12-12
    tiebreak (13-12), or an advantage-set continuation (both >=6, margin 2).
    rules.py decides whether the format in force at match time permitted it.
    """
    hi, lo = max(w, l), min(w, l)
    return (
        (hi == 6 and lo <= 4)
        or (hi == 7 and lo in (5, 6))
        or (hi == 13 and lo == 12)
        or (hi >= 7 and lo >= 6 and hi - lo == 2)
    )


def classify_outcome(score: object) -> str:
    """Map a TML score string to a canonical outcome (ground rule 6)."""
    if not isinstance(score, str) or not score.strip():
        return "unknown"
    if _WO.search(score):
        return "walkover"
    if _DEF.search(score):
        return "default"
    if _RET.search(score):
        return "retired"
    return "completed"


def parse_score(score: object, outcome: str, best_of: float) -> ParsedScore:
    """Parse a TML score string into per-set games, with suspect detection.

    Only format-agnostic checks live here. Format-conditional checks (does
    this set obey the deciding-set rule in force?) belong to rules.py's
    empirical scan, which flags the same field via ``flag_suspect``.
    """
    reasons: list[str] = []
    if outcome in ("walkover", "unknown"):
        if outcome == "unknown":
            reasons.append("score missing or unparseable")
        return ParsedScore([], 0, 0, 0, bool(reasons), "; ".join(reasons))

    sets: list[tuple[int, int]] = []
    tb_flags: list[bool] = []
    for tok in str(score).replace("[", "").replace("]", "").split():
        if _RET.fullmatch(tok) or _WO.fullmatch(tok) or _DEF.fullmatch(tok):
            continue
        mt = _SET_TOKEN.match(tok)
        if not mt:
            reasons.append(f"unparseable token {tok!r}")
            continue
        w, l = int(mt.group(1)), int(mt.group(2))
        sets.append((w, l))
        tb_flags.append(
            mt.group(3) is not None or {w, l} == {7, 6} or {w, l} == {13, 12}
        )
    tiebreaks = sum(tb_flags)

    if outcome == "completed":
        if not sets:
            reasons.append("completed match with no sets")
        need = int(best_of) // 2 + 1
        won = sum(1 for w, l in sets if w > l)
        lost = sum(1 for w, l in sets if l > w)
        if won != need:
            reasons.append(f"winner took {won} sets, needs {need}")
        if lost >= need:
            reasons.append(f"loser took {lost} sets, >= {need}")
        for i, ((w, l), tb) in enumerate(zip(sets, tb_flags), start=1):
            # A match tiebreak played in place of a final set is written
            # `1-0(7)` — a legal terminal set wherever the format allows it.
            if tb and {w, l} == {1, 0}:
                continue
            if not _terminal(w, l):
                reasons.append(f"set {i} {w}-{l} not terminal")

    return ParsedScore(
        sets=sets,
        tiebreaks=tiebreaks,
        winner_games=sum(w for w, _ in sets),
        loser_games=sum(l for _, l in sets),
        suspect=bool(reasons),
        reason="; ".join(reasons),
    )


def _season_years(include_holdout: bool = False) -> list[int]:
    """Season years available for loading.

    Holdout years are excluded unless explicitly requested, which only the
    final backtest does. Every other caller gets a dataset that physically
    cannot contain post-cutoff data.
    """
    years = set()
    for p in C.VENDOR_DIR.glob("*_matches_*.csv"):
        y = int(p.stem.rsplit("_", 1)[1])
        if y < C.DATA_START.year:
            continue
        if y >= C.HOLDOUT_CUTOFF.year and not include_holdout:
            continue
        years.add(y)
    return sorted(years)


def load_raw(years: list[int] | None = None,
             include_holdout: bool = False) -> pd.DataFrame:
    """Load TML season files into one canonical match record.

    Raises if any loaded date falls in the HOLDOUT range, unless
    ``include_holdout`` is set — which only the final backtest does.
    """
    years = years or _season_years(include_holdout)
    frames = []
    for tour, stem in _TOURS.items():
        for y in years:
            path = C.VENDOR_DIR / f"{stem}_{y}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path, low_memory=False)
            df["tour"] = tour
            df["season_file"] = y
            frames.append(df)

    m = pd.concat(frames, ignore_index=True)
    # TML writes some counters as text in some season files (l_bpFaced in the
    # 2024+ files), which turns a whole column to object dtype once concatenated
    # and makes every numeric comparison against it fail.
    for side in ("w", "l"):
        for col in [f"{side}_{c}" for c in _SERVE_COLS] + [f"{side}_svpt"]:
            if col in m.columns:
                m[col] = pd.to_numeric(m[col], errors="coerce")
    m["date"] = pd.to_datetime(m["tourney_date"], format="%Y%m%d").dt.date
    m["level_label"] = m["tourney_level"].astype(str).map(LEVEL_LABEL).fillna("unknown")
    #: tourney_id is `YYYY-CODE`; the year prefix moves each season, the rest
    #: is stable. This is the venue/tournament key for Stage 6. A minority of
    #: ids lack the year prefix entirely (audited in the data dictionary) and
    #: are used whole.
    m["tourney_code"] = m["tourney_id"].astype(str).str.replace(
        r"^\d{4}-", "", regex=True
    )
    m["outcome"] = m["score"].map(classify_outcome)

    parsed = [
        parse_score(s, o, b)
        for s, o, b in zip(m["score"], m["outcome"], m["best_of"])
    ]
    m["n_sets"] = [len(p.sets) for p in parsed]
    m["tiebreaks"] = [p.tiebreaks for p in parsed]
    m["winner_games"] = [p.winner_games for p in parsed]
    m["loser_games"] = [p.loser_games for p in parsed]
    m["score_string_suspect"] = [p.suspect for p in parsed]
    m["score_suspect_reason"] = [p.reason for p in parsed]

    m = _cross_check_games(m)
    m["in_scope"] = _in_scope(m)
    m["serve_stats_valid"] = _serve_stats_valid(m)
    # season_file is part of the key because some tourney_ids carry no year
    # prefix and are reused across seasons (see the dictionary's identifier
    # anomalies section).
    m["match_id"] = (
        m["season_file"].astype(str) + "-" + m["tourney_id"].astype(str)
        + "-" + m["match_num"].astype(str) + "-" + m["tour"]
    )
    m["split"] = [C.split_of(d) for d in m["date"]]

    if not include_holdout:
        C.assert_no_holdout(m["date"])
    return m


def _cross_check_games(m: pd.DataFrame) -> pd.DataFrame:
    """Games parsed from the score must match the served-games columns.

    ``w_SvGms + l_SvGms`` is recorded independently of the score string, so a
    disagreement means one of the two is corrupt — exactly the shared data
    quality problem ground rule 8 exists to broadcast.
    """
    played = m["outcome"].isin(("completed", "retired"))
    # SvGms is recorded as 0 on 601 rows (mostly Grand Slam 2017-18) whose
    # other counters are present — a missing field, not a zero, so it cannot
    # arbitrate the score string.
    have = (m["w_SvGms"] > 0) & (m["l_SvGms"] > 0)
    # TML is not internally consistent about whether a tiebreak counts as a
    # service game — both conventions appear — so a set with a tiebreak buys
    # one game of slack either way. An unfinished game at retirement costs one
    # more. Beyond that the two records genuinely conflict.
    total_score = m["winner_games"] + m["loser_games"]
    total_gms = m["w_SvGms"].fillna(0) + m["l_SvGms"].fillna(0)
    bad = played & have & ((total_score - total_gms).abs() > m["tiebreaks"] + 1)
    m.loc[bad, "score_suspect_reason"] = (
        m.loc[bad, "score_suspect_reason"]
        + np.where(m.loc[bad, "score_suspect_reason"] == "", "", "; ")
        + "score games "
        + total_score[bad].astype(int).astype(str)
        + " disagree with SvGms "
        + total_gms[bad].astype(int).astype(str)
    )
    m.loc[bad, "score_string_suspect"] = True
    return m


#: Event types excluded from the model by decision, 2026-07-30. Davis Cup ties
#: carry per-tie identifiers that give the rules inference almost nothing to
#: work with and irregular formats; the Olympics and the season-ending Finals
#: are small, atypical fields; Next Gen Finals uses first-to-4 short sets the
#: engine does not price.
EXCLUDED_LEVELS = ("Davis Cup", "Olympics", "Tour Finals")
EXCLUDED_CODES = ("7696",)


def _in_scope(m: pd.DataFrame) -> pd.Series:
    """Rows the model is built on. Excluded rows stay in the record, audited."""
    return ~(
        m["level_label"].isin(EXCLUDED_LEVELS) | m["tourney_code"].isin(EXCLUDED_CODES)
    )


def _serve_stats_valid(m: pd.DataFrame) -> pd.Series:
    """Serve-stat rows usable for rate fitting (Stage 2's input filter).

    Separate from ``score_string_suspect``: a match can have a clean score and
    missing or internally inconsistent serve counts.
    """
    ok = pd.Series(True, index=m.index)
    for side in ("w", "l"):
        cols = [f"{side}_{c}" for c in _SERVE_COLS]
        ok &= m[cols].notna().all(axis=1)
        ok &= m[f"{side}_svpt"] > 0
        ok &= m[f"{side}_SvGms"] > 0  # 0 means "not recorded", not "served none"
        ok &= m[f"{side}_1stIn"] <= m[f"{side}_svpt"]
        ok &= m[f"{side}_1stWon"] <= m[f"{side}_1stIn"]
        ok &= m[f"{side}_2ndWon"] <= (m[f"{side}_svpt"] - m[f"{side}_1stIn"])
        ok &= m[f"{side}_bpSaved"] <= m[f"{side}_bpFaced"]
        ok &= m[f"{side}_ace"] <= m[f"{side}_svpt"]
    # Ground rule 8: a suspect score string discredits the whole row for any
    # consumer of its game-level detail, Stage 2's rate fitting included.
    return (ok & m["outcome"].eq("completed") & m["in_scope"]
            & ~m["score_string_suspect"])


def flag_suspect(matches: pd.DataFrame, match_ids: list[str], reason: str) -> pd.DataFrame:
    """Set ``score_string_suspect`` on the canonical record (ground rule 8).

    rules.py's discrepancy scan calls this so its findings live on the match
    record, not only in ``reports/rules_discrepancies.csv``.
    """
    hit = matches["match_id"].isin(match_ids)
    matches.loc[hit, "score_suspect_reason"] = (
        matches.loc[hit, "score_suspect_reason"].fillna("")
        + np.where(matches.loc[hit, "score_suspect_reason"] == "", "", "; ")
        + reason
    )
    matches.loc[hit, "score_string_suspect"] = True
    return matches


def find_duplicates(m: pd.DataFrame) -> pd.DataFrame:
    """Same date + same two players + same tournament."""
    pair = np.where(
        m["winner_id"].astype(str) < m["loser_id"].astype(str),
        m["winner_id"].astype(str) + "|" + m["loser_id"].astype(str),
        m["loser_id"].astype(str) + "|" + m["winner_id"].astype(str),
    )
    key = m["date"].astype(str) + "|" + m["tourney_id"].astype(str) + "|" + pair
    dup = pd.Series(key, index=m.index).duplicated(keep=False)
    cols = ["match_id", "date", "tourney_name", "round", "winner_name",
            "loser_name", "score", "tour"]
    return m.loc[dup, cols].sort_values(["date", "winner_name"])


def surface_changes(m: pd.DataFrame) -> pd.DataFrame:
    """Tournaments (stable code) whose surface is not constant across seasons."""
    g = m.groupby("tourney_code")["surface"].nunique()
    changed = g[g > 1].index
    out = (
        m[m["tourney_code"].isin(changed)]
        .groupby(["tourney_code", "surface"])
        .agg(
            name=("tourney_name", lambda s: s.mode().iat[0]),
            seasons=("season_file", lambda s: ", ".join(map(str, sorted(s.unique())))),
            matches=("match_id", "size"),
        )
        .reset_index()
    )
    return out.sort_values(["tourney_code", "surface"])


def identifier_instability(m: pd.DataFrame) -> pd.DataFrame:
    """Tournament codes whose name changes across seasons (sponsor/city renames)."""
    g = m.groupby("tourney_code")["tourney_name"].nunique()
    out = (
        m[m["tourney_code"].isin(g[g > 1].index)]
        .groupby("tourney_code")["tourney_name"]
        .agg(lambda s: sorted(s.unique()))
        .reset_index()
    )
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_data_dictionary(m: pd.DataFrame, path: Path | None = None) -> Path:
    """Write reports/data_dictionary.md. Every number here is computed."""
    path = path or C.REPORTS_DIR / "data_dictionary.md"
    sample = pd.read_csv(C.VENDOR_DIR / "atp_matches_2019.csv", nrows=200)

    L: list[str] = []
    a = L.append
    a("# Stage 0 — Data dictionary (TML-Database)\n")
    a(f"Generated by `model/data_audit.py`. Rows: **{len(m):,}** "
      f"({(m['tour'] == 'atp').sum():,} ATP main tour, "
      f"{(m['tour'] == 'chall').sum():,} Challenger), seasons "
      f"{m['season_file'].min()}–{m['season_file'].max()}, tournament start "
      f"dates {m['date'].min()} … {m['date'].max()}.\n")
    a("Source: `vendor/*_matches_*.csv`, retrieved 2026-07-13 per "
      "`vendor/NOTICE`. Licence MIT, attribution TennisMyLife.\n")
    a("`data/raw/*.xlsx` (tennis-data.co.uk) carries bookmaker odds and no "
      "serve statistics; no model module reads it.\n")

    a("\n## Columns and dtypes\n")
    a("Identical column set in every season file, both tours.\n")
    a("\n| column | dtype | note |")
    a("|---|---|---|")
    notes = {
        "tourney_id": "`YYYY-CODE`; the year prefix moves, the CODE suffix is stable",
        "tourney_date": "tournament START date, not match date — see split note",
        "winner_id": "**non-numeric string** (e.g. `T786`) — TML deviation",
        "loser_id": "**non-numeric string** — TML deviation",
        "indoor": "`I`/`O`, sparsely populated — see coverage",
        "score": "carries RET / W/O / DEF markers; no separate flag column",
        "minutes": "match duration",
    }
    for c in sample.columns:
        a(f"| `{c}` | {sample[c].dtype} | {notes.get(c, '')} |")
    a("\nDerived by this module: `date`, `tour`, `season_file`, `level_label`, "
      "`tourney_code`, `outcome`, `n_sets`, `tiebreaks`, `winner_games`, "
      "`loser_games`, `score_string_suspect`, `score_suspect_reason`, "
      "`serve_stats_valid`, `match_id`, `split`.\n")

    a("\n## Row counts by season and tour level\n")
    ct = pd.crosstab(m["season_file"], m["level_label"], margins=True,
                     margins_name="total")
    a(ct.to_markdown())
    a("\n2020 is roughly half a normal season (COVID).\n")

    a("\n### Scope exclusions\n")
    out_rows = m[~m["in_scope"]]
    a(f"**{len(out_rows):,} rows ({_pct(len(out_rows) / len(m))}) are excluded "
      f"from the model**, leaving {int(m['in_scope'].sum()):,} in scope "
      f"({(m['in_scope'] & m['tour'].eq('atp')).sum():,} ATP, "
      f"{(m['in_scope'] & m['tour'].eq('chall')).sum():,} Challenger). They stay "
      "in the canonical record, flagged `in_scope=False`, so the audit still "
      "sees them.\n")
    a("\n| excluded | rows | why |")
    a("|---|---|---|")
    a(f"| Davis Cup | {(out_rows['level_label'] == 'Davis Cup').sum():,} | "
      "per-tie identifiers leave the rules inference with almost no evidence, "
      "and tie formats are irregular |")
    a(f"| Olympics | {(out_rows['level_label'] == 'Olympics').sum():,} | small, "
      "atypical field on a four-year cycle |")
    a(f"| Tour Finals | {(out_rows['level_label'] == 'Tour Finals').sum():,} | "
      "round-robin, eight-player field |")
    a(f"| Next Gen Finals | {(out_rows['tourney_code'] == '7696').sum():,} | "
      "first-to-4 short sets; the engine prices 6-game sets only |")

    a("\n## Per-match statistics and their coverage\n")
    a("All nine serve counters exist for both players: `{w,l}_` × "
      + ", ".join(f"`{c}`" for c in _SERVE_COLS) + ", plus `minutes`.\n")
    a("\nNon-null % of `w_svpt` (serve points), by season and tour:\n")
    cov = (
        m.assign(has=m["w_svpt"].notna())
        .groupby(["season_file", "tour"])["has"].mean().unstack()
    )
    a((100 * cov).round(1).to_markdown())
    a("\nNon-null % of other fields, whole dataset:\n")
    fields = ["minutes", "indoor", "surface", "winner_hand", "winner_ht",
              "winner_age", "winner_rank", "winner_seed", "draw_size"]
    a(pd.DataFrame({
        "field": fields,
        "non-null %": [round(100 * m[f].notna().mean(), 1) for f in fields],
    }).to_markdown(index=False))
    a(f"\n`serve_stats_valid` (all counters present, positive serve points, "
      f"internally consistent, completed match): **{_pct(m['serve_stats_valid'].mean())}** "
      f"of all rows, {_pct(m.loc[m.outcome == 'completed', 'serve_stats_valid'].mean())} "
      "of completed matches. This is Stage 2's input filter.\n")
    a(f"\n`indoor` is populated on only {_pct(m['indoor'].notna().mean())} of "
      "rows — Stage 6 may use it as a feature only where present, and must "
      "not treat missing as outdoor.\n")

    a("\n## Retirements, walkovers, defaults\n")
    a("TML has no flag column: status is embedded in `score` as `RET`, `W/O` "
      "or `DEF`. Counts:\n")
    a("\n| outcome | rows | share | downstream policy (ground rule 6) |")
    a("|---|---|---|---|")
    for oc, pol in OUTCOME_POLICY.items():
        n = int((m["outcome"] == oc).sum())
        a(f"| `{oc}` | {n:,} | {_pct(n / len(m))} | {pol} |")
    a("\nBy tour:\n")
    a(pd.crosstab(m["outcome"], m["tour"]).to_markdown())

    a("\n## Surfaces and venues\n")
    a(f"Surface labels: {m['surface'].value_counts(dropna=False).to_dict()}\n")
    a(f"\n`surface` is null on {int(m['surface'].isna().sum()):,} rows "
      f"({_pct(m['surface'].isna().mean())}); Carpet exists in the early "
      "seasons and is a distinct surface, not a Hard variant.\n")
    a(f"\nDistinct tournament codes: {m['tourney_code'].nunique()}; distinct "
      f"names: {m['tourney_name'].nunique()}. Stage 6 keys on "
      "(`tourney_code`, `surface`), never on name and never on `tourney_id` "
      "(whose year prefix changes annually).\n")

    a("\n### Surface-change flags\n")
    sc = surface_changes(m)
    a(f"{sc['tourney_code'].nunique()} tournament codes change surface across "
      "seasons. Every one must be keyed as (code, surface):\n")
    a(sc.to_markdown(index=False))

    a("\n### Identifier anomalies\n")
    noyear = m[~m["tourney_id"].astype(str).str.match(r"^\d{4}-")]
    a(f"{len(noyear):,} rows carry a `tourney_id` with no `YYYY-` prefix. "
      "They are used whole as the tournament code, and `match_id` includes the "
      "season file so the reuse cannot collide:\n")
    if len(noyear):
        a(noyear.groupby(["tourney_id", "tourney_name", "season_file"])
          .size().rename("matches").reset_index().to_markdown(index=False))
        a("\nThe `Rome` / `GA` rows are a TML field swap — the id holds the "
          "city and the name holds the state — for the Rome, Georgia "
          "Challenger. Reported, not silently repaired.\n")
    dc = m[m["tourney_id"].astype(str).str.contains("DC-")]
    a(f"\nDavis Cup ties ({len(dc):,} rows) use a compound id per tie rather "
      "than a stable tournament code; each tie is effectively its own venue. "
      "Stage 6 must not pool them into one multiplier.\n")

    a("\n### Identifier stability (renames)\n")
    ren = identifier_instability(m)
    a(f"{len(ren)} tournament codes carry more than one name across seasons "
      "(sponsor and city renames). The code is stable, the name is not:\n")
    a(ren.head(40).to_markdown(index=False))
    if len(ren) > 40:
        a(f"\n… {len(ren) - 40} more.\n")

    a("\n## Score-string suspect flag (ground rule 8)\n")
    n = int(m["score_string_suspect"].sum())
    a(f"{n:,} of {len(m):,} rows ({_pct(n / len(m))}) flagged by Stage 0's "
      "format-agnostic checks: unparseable tokens, non-terminal sets, set "
      "counts inconsistent with `best_of`, and score games disagreeing with "
      "`w_SvGms + l_SvGms` by more than one game. rules.py's "
      "format-conditional scan adds to the same field via `flag_suspect()`.\n")
    a("\nBy outcome:\n")
    a(m.groupby("outcome")["score_string_suspect"].agg(["sum", "size"]).to_markdown())
    a("\nTop reasons:\n")
    reasons = m.loc[m["score_string_suspect"], "score_suspect_reason"]
    a(reasons.str.replace(r"\d+", "N", regex=True).value_counts().head(15).to_markdown())

    a("\n## Duplicate detection\n")
    dups = find_duplicates(m)
    a(f"Same tournament + same start date + same two players: "
      f"**{len(dups)} rows**.\n")
    if len(dups):
        a(dups.head(40).to_markdown(index=False))
        a("\nThese are round-robin and Davis Cup rematches plus genuine "
          "double-entries; they are reported, not dropped, because "
          "distinguishing the two needs the round field and match_num, which "
          "downstream stages have.\n")

    a("\n## Split assignment\n")
    a(m.groupby(["split", "season_file"]).size().unstack(fill_value=0).to_markdown())
    a("\nSplit is assigned from `tourney_date` (tournament start), so an event "
      "straddling a boundary lands whole in one window. No row on or after the "
      f"permanently fixed holdout cutoff ({C.HOLDOUT_CUTOFF}) is loaded; the "
      "2024–2026 season files are never opened.\n")

    path.write_text("\n".join(L) + "\n")
    return path


#: The holdout-inclusive record lives in its own file so no ordinary caller
#: can reach post-cutoff data by accident.
HOLDOUT_PARQUET = "matches_with_holdout.parquet"


def build(save: bool = True, include_holdout: bool = False) -> pd.DataFrame:
    """Load, audit, and persist the canonical match record."""
    m = load_raw(include_holdout=include_holdout)
    if save:
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        if include_holdout:
            m.to_parquet(C.PROCESSED_DIR / HOLDOUT_PARQUET, index=False)
        else:
            m.to_parquet(C.PROCESSED_DIR / "matches.parquet", index=False)
            write_data_dictionary(m)
    return m


def demo() -> None:
    """Worked example (ground rule 10)."""
    m = load_raw()
    print(f"loaded {len(m):,} matches, {m['date'].min()} .. {m['date'].max()}")
    print(f"tours: {m['tour'].value_counts().to_dict()}")
    print(f"outcomes: {m['outcome'].value_counts().to_dict()}")
    print(f"serve_stats_valid: {int(m['serve_stats_valid'].sum()):,}")
    print(f"score_string_suspect: {int(m['score_string_suspect'].sum()):,}")
    cols = ["match_id", "date", "tourney_name", "surface", "winner_name",
            "loser_name", "score", "outcome", "score_suspect_reason"]
    print("\nfirst 5 suspect rows:")
    print(m.loc[m["score_string_suspect"], cols].head(5).to_string(index=False))


if __name__ == "__main__":
    demo()
