"""Stage 0 validation gate.

Per MODEL_PROMPT.md: the data dictionary exists, ingestion of every year runs
without error, coverage numbers are printed not assumed, and the surface-change
and retirement-handling sections are complete.
"""

from __future__ import annotations

import pandas as pd
import pytest

from model import constants as C
from model import data_audit as A


@pytest.fixture(scope="module")
def m() -> pd.DataFrame:
    return A.load_raw()


def test_every_year_in_range_ingests(m: pd.DataFrame) -> None:
    years = A._season_years()
    assert years == list(range(2013, 2024)), years
    counts = m.groupby("season_file").size()
    assert set(counts.index) == set(years)
    assert (counts > 1000).all(), counts.to_dict()


def test_holdout_files_never_opened() -> None:
    """The 2024+ season files exist on disk and must not be in the load set."""
    on_disk = {int(p.stem) for p in C.DATA_RAW_DIR.glob("*.xlsx")}
    assert on_disk & {2024, 2025, 2026}, "expected holdout files to exist"
    assert not set(A._season_years()) & {y for y in on_disk if y >= 2024}


def test_no_holdout_dates(m: pd.DataFrame) -> None:
    assert max(m["date"]) < C.HOLDOUT_CUTOFF
    assert set(m["split"]) == {"fit", "tune", "test", "reserve"}


def test_no_bookmaker_odds_in_canonical_record(m: pd.DataFrame) -> None:
    assert not [c for c in m.columns if A._ODDS_PATTERN.match(c)]


def test_every_comment_value_is_mapped(m: pd.DataFrame) -> None:
    assert set(m["comment"].dropna()) <= set(A._OUTCOME)
    assert set(m["outcome"]) <= set(A.OUTCOME_POLICY)


def test_best_of_repair(m: pd.DataFrame) -> None:
    assert m["best_of"].notna().all()
    slam = m["level"] == "Grand Slam"
    assert (m.loc[slam, "best_of"] == 5).all()
    assert (m.loc[~slam, "best_of"] == 3).all()
    assert m["best_of_repaired"].sum() == 25


def test_terminal_set_rules() -> None:
    for w, l in [(6, 0), (6, 4), (7, 5), (7, 6), (13, 12), (70, 68), (8, 6)]:
        assert A._terminal(w, l), (w, l)
    for w, l in [(5, 3), (6, 5), (7, 4), (2, 0), (12, 12), (9, 6)]:
        assert not A._terminal(w, l), (w, l)


def test_suspect_flag_is_rare_and_reasoned(m: pd.DataFrame) -> None:
    rate = m["score_string_suspect"].mean()
    assert rate < 0.01, rate
    flagged = m[m["score_string_suspect"]]
    assert (flagged["score_suspect_reason"].str.len() > 0).all()
    assert (m.loc[~m["score_string_suspect"], "score_suspect_reason"] == "").all()


def test_retirements_and_walkovers_not_suspect_by_construction(m: pd.DataFrame) -> None:
    """Truncation is an outcome, not a garbled score."""
    ret = m[m["outcome"] == "retired"]
    assert ret["score_string_suspect"].mean() < 0.01
    wo = m[m["outcome"] == "walkover"]
    assert wo["score_string"].eq("").all()
    assert not wo["score_string_suspect"].any()


def test_flag_suspect_propagates(m: pd.DataFrame) -> None:
    df = m.head(50).copy()
    clean = df.loc[~df["score_string_suspect"], "match_id"].iloc[0]
    A.flag_suspect(df, [clean], "rules.py scan: format mismatch")
    row = df.loc[df["match_id"] == clean].iloc[0]
    assert row["score_string_suspect"]
    assert "rules.py scan" in row["score_suspect_reason"]


def test_duplicates_and_surface_changes_computed(m: pd.DataFrame) -> None:
    assert len(A.find_duplicates(m)) == 0
    sc = A.surface_changes(m)
    assert sc["tournament"].nunique() >= 4
    loc_multi, tour_multi = A.identifier_instability(m)
    assert len(loc_multi) > 0 and len(tour_multi) > 0


def test_data_dictionary_sections_complete() -> None:
    path = C.REPORTS_DIR / "data_dictionary.md"
    assert path.exists(), "run `python -m model.data_audit` / A.build() first"
    text = path.read_text()
    for section in [
        "## Files and coverage",
        "### Row counts by season and level",
        "### Field coverage (non-null %) by season",
        "### Per-match statistics available",
        "## Retirements, walkovers, defaults",
        "### Surface-change flags",
        "### Identifier stability (renames)",
        "## Score-string suspect flag",
        "## Duplicate detection",
        "## Split assignment",
    ]:
        assert section in text, section
    # coverage numbers printed, not asserted in prose
    assert "27,458" in text
    for outcome, policy in A.OUTCOME_POLICY.items():
        assert f"`{outcome}`" in text
        assert policy.split(".")[0] in text
