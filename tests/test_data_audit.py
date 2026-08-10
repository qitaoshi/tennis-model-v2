"""Stage 0 validation gate (TML-Database).

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
    assert years == list(range(C.DATA_START.year, C.HOLDOUT_CUTOFF.year)), years
    counts = m.groupby(["season_file", "tour"]).size()
    assert set(counts.index.get_level_values(0)) == set(years)
    assert (counts > 1000).all(), counts.to_dict()


def test_holdout_files_never_opened() -> None:
    on_disk = {int(p.stem.rsplit("_", 1)[1]) for p in C.VENDOR_DIR.glob("*_matches_*.csv")}
    held = {y for y in on_disk if y >= C.HOLDOUT_CUTOFF.year}
    assert held, "expected holdout season files to exist on disk"
    assert not set(A._season_years()) & held


def test_no_holdout_dates(m: pd.DataFrame) -> None:
    assert max(m["date"]) < C.HOLDOUT_CUTOFF
    assert set(m["split"]) <= {"fit", "tune", "burned_test", "test", "reserve"}
    assert {"fit", "tune", "test"} <= set(m["split"])


def test_pre_data_rows_dropped(m: pd.DataFrame) -> None:
    """A misfiled event can put pre-DATA_START matches in a modern file."""
    assert min(m["date"]) >= C.DATA_START
    assert "pre_data" not in set(m["split"])


def test_both_tours_present(m: pd.DataFrame) -> None:
    assert set(m["tour"]) == {"atp", "chall"}
    assert (m["level_label"] == "Challenger").sum() > 50_000
    assert m["level_label"].ne("unknown").all()


def test_outcome_classification() -> None:
    assert A.classify_outcome("6-4 6-3") == "completed"
    assert A.classify_outcome("6-4 3-0 RET") == "retired"
    assert A.classify_outcome("W/O") == "walkover"
    assert A.classify_outcome("6-1 DEF") == "default"
    assert A.classify_outcome(float("nan")) == "unknown"
    assert A.classify_outcome("") == "unknown"


def test_outcomes_are_policied(m: pd.DataFrame) -> None:
    assert set(m["outcome"]) <= set(A.OUTCOME_POLICY)
    assert (m["outcome"] == "walkover").sum() > 0
    assert (m["outcome"] == "retired").sum() > 0


def test_terminal_set_rules() -> None:
    for w, l in [(6, 0), (6, 4), (7, 5), (7, 6), (13, 12), (70, 68), (8, 6)]:
        assert A._terminal(w, l), (w, l)
    for w, l in [(5, 3), (6, 5), (7, 4), (2, 0), (12, 12), (9, 6), (4, 1)]:
        assert not A._terminal(w, l), (w, l)


def test_parse_score() -> None:
    p = A.parse_score("7-6(6) 6-4", "completed", 3)
    assert p.sets == [(7, 6), (6, 4)] and p.tiebreaks == 1
    assert p.winner_games == 13 and p.loser_games == 10 and not p.suspect

    p = A.parse_score("6-4 3-0 RET", "retired", 3)
    assert p.sets == [(6, 4), (3, 0)] and not p.suspect  # truncation is not garble

    p = A.parse_score("W/O", "walkover", 3)
    assert p.sets == [] and not p.suspect

    # match tiebreak in place of a final set (Laver Cup)
    assert not A.parse_score("6-7(3) 7-5 1-0(7)", "completed", 3).suspect
    # genuine garble: a non-terminal set in a completed match
    assert A.parse_score("6-4 3-1", "completed", 3).suspect
    # too few sets for the format
    assert A.parse_score("6-2 4-6 7-6(1)", "completed", 5).suspect


def test_suspect_flag_is_rare_and_reasoned(m: pd.DataFrame) -> None:
    assert m["score_string_suspect"].mean() < 0.01
    flagged = m[m["score_string_suspect"]]
    assert (flagged["score_suspect_reason"].str.len() > 0).all()
    assert (m.loc[~m["score_string_suspect"], "score_suspect_reason"] == "").all()


def test_truncated_outcomes_not_suspect_by_construction(m: pd.DataFrame) -> None:
    assert m.loc[m["outcome"] == "retired", "score_string_suspect"].mean() < 0.02
    assert not m.loc[m["outcome"] == "walkover", "score_string_suspect"].any()


def test_scope_exclusions(m: pd.DataFrame) -> None:
    """Excluded events stay in the record, flagged, not deleted."""
    out = m[~m["in_scope"]]
    assert set(out["level_label"]) <= set(A.EXCLUDED_LEVELS) | {"Tour (other)"}
    assert (out["level_label"].isin(A.EXCLUDED_LEVELS)
            | out["tourney_code"].isin(A.EXCLUDED_CODES)).all()
    assert not m.loc[m["in_scope"], "level_label"].isin(A.EXCLUDED_LEVELS).any()
    assert not m.loc[m["in_scope"], "tourney_code"].isin(A.EXCLUDED_CODES).any()
    # Drift tripwire, not a derivation. Update deliberately when the split
    # boundaries or the vendor files move, never to make a red test green.
    # 96,617 under the pre-2026-08-08 split (FIT..TEST ending 2023-12-31).
    assert m["in_scope"].sum() == 114_353


def test_serve_stats_valid_filter(m: pd.DataFrame) -> None:
    ok = m["serve_stats_valid"]
    assert 0.85 < ok.mean() < 1.0
    assert (m.loc[ok, "outcome"] == "completed").all()
    assert (m.loc[ok, "w_svpt"] > 0).all()
    assert (m.loc[ok, "w_SvGms"] > 0).all()
    assert (m.loc[ok, "w_1stWon"] <= m.loc[ok, "w_1stIn"]).all()


def test_flag_suspect_propagates(m: pd.DataFrame) -> None:
    df = m.head(200).copy()
    clean = df.loc[~df["score_string_suspect"], "match_id"].iloc[0]
    A.flag_suspect(df, [clean], "rules.py scan: format mismatch")
    row = df.loc[df["match_id"] == clean].iloc[0]
    assert row["score_string_suspect"]
    assert "rules.py scan" in row["score_suspect_reason"]


def test_match_id_unique(m: pd.DataFrame) -> None:
    assert m["match_id"].is_unique


def test_tourney_code_is_stable_across_seasons(m: pd.DataFrame) -> None:
    """The venue key Stage 6 depends on: code persists, year prefix does not."""
    seasons = m.groupby("tourney_code")["season_file"].nunique()
    assert (seasons > 5).sum() > 100
    brisbane = m[m["tourney_code"] == "339"]
    assert brisbane["season_file"].nunique() > 5


def test_duplicates_and_surface_changes_computed(m: pd.DataFrame) -> None:
    A.find_duplicates(m)  # reported, not asserted away
    sc = A.surface_changes(m)
    assert sc["tourney_code"].nunique() > 0
    assert len(A.identifier_instability(m)) > 0


def test_data_dictionary_sections_complete(m: pd.DataFrame) -> None:
    path = C.REPORTS_DIR / "data_dictionary.md"
    assert path.exists(), "run `python -m model.data_audit` / A.build() first"
    text = path.read_text()
    for section in [
        "## Columns and dtypes",
        "## Row counts by season and tour level",
        "## Per-match statistics and their coverage",
        "## Retirements, walkovers, defaults",
        "### Surface-change flags",
        "### Identifier stability (renames)",
        "## Score-string suspect flag",
        "## Duplicate detection",
        "## Split assignment",
    ]:
        assert section in text, section
    # The dictionary must state the row count of the record it describes.
    # Derived, not hardcoded: the count moves whenever the split boundaries do,
    # and a literal here would just be re-typed to whatever made it green.
    assert f"{len(m):,}" in text
    for outcome, policy in A.OUTCOME_POLICY.items():
        assert f"`{outcome}`" in text
        assert policy.split("—")[0].split(".")[0].strip() in text
