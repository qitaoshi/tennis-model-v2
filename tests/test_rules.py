"""rules.py validation gate.

Per MODEL_PROMPT.md: spot-check against known matches, then scan the whole
dataset without hard-failing on mismatch, logging discrepancies and flagging
tournaments above the manual-review threshold.
"""

from __future__ import annotations

import pandas as pd
import pytest

from model import constants as C
from model import rules as R


@pytest.fixture(scope="module")
def m() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")


# --- documented rule histories -------------------------------------------

def test_majors_documented() -> None:
    for code, year in [("520", 2015), ("540", 2015), ("560", 2015), ("580", 2015)]:
        assert R.rules_for(code, year).provenance == "documented"


def test_wimbledon_rule_history() -> None:
    assert R.rules_for("540", 2010).final_set == "advantage"
    assert R.rules_for("540", 2018).final_set == "advantage"
    s2019 = R.rules_for("540", 2019)
    assert (s2019.final_set, s2019.final_tb_at, s2019.final_tb_to) == ("tiebreak", 12, 7)
    s2023 = R.rules_for("540", 2023)
    assert (s2023.final_set, s2023.final_tb_at, s2023.final_tb_to) == ("tiebreak", 6, 10)


def test_other_major_histories() -> None:
    assert R.rules_for("580", 2018).final_set == "advantage"
    assert R.rules_for("580", 2019).final_tb_to == 10
    assert R.rules_for("520", 2021).final_set == "advantage"
    assert R.rules_for("520", 2022).final_tb_to == 10
    assert R.rules_for("560", 2015).final_tb_to == 7  # US Open TB throughout
    assert R.rules_for("560", 2022).final_tb_to == 10


def test_isner_mahut_is_legal_under_its_assigned_rules(m: pd.DataFrame) -> None:
    """Wimbledon 2010 70-68: the canonical advantage-set match."""
    row = m[(m["tourney_code"] == "540") & (m["season_file"] == 2010)
            & m["score"].str.contains("70-68", na=False)]
    assert len(row) == 1, "Isner-Mahut not found in the data"
    r = row.iloc[0]
    spec = R.rules_for(r["tourney_code"], int(r["season_file"]))
    assert spec.final_set == "advantage"
    assert R.score_conflicts(r["score"], spec, int(r["n_sets"])) == ""
    assert not r["score_string_suspect"]


def test_modern_wimbledon_final_set_resolves_by_tiebreak() -> None:
    spec = R.rules_for("540", 2023)
    # 6-6 must resolve as a 10-point tiebreak: a 7-6 final set is legal,
    # an advantage continuation is not.
    assert R.score_conflicts("6-4 3-6 6-3 6-7(4) 7-6(8)", spec, 5) == ""
    assert R.score_conflicts("6-4 3-6 6-3 6-7(4) 12-10", spec, 5) != ""


def test_advantage_format_rejects_deciding_tiebreak() -> None:
    spec = R.rules_for("540", 2010)
    assert R.score_conflicts("6-4 3-6 6-3 6-7(4) 7-6(3)", spec, 5) != ""
    assert R.score_conflicts("6-4 3-6 6-3 6-7(4) 12-10", spec, 5) == ""


def test_match_tiebreak_notation_is_not_a_conflict() -> None:
    spec = R.rules_for("605", 2019)
    assert R.score_conflicts("6-7(3) 7-5 [10-8]", spec, 3) == ""


def test_short_set_format_is_marked_unsupported() -> None:
    spec = R.rules_for("7696", 2019)
    assert spec.games_to_win_set == 4 and not spec.supported
    assert spec.provenance == "documented"


# --- inference ------------------------------------------------------------

def test_inference_recovers_known_advantage_eras() -> None:
    """The inferred table must independently see what the majors document."""
    table = R._INFERRED_CACHE or R._inferred("520", 2015) and R._INFERRED_CACHE
    assert table is not None, "run rules.build() first"
    for code, year in [("520", 2015), ("540", 2015), ("580", 2015)]:
        assert table[f"{code}|{year}"]["final_set"] == "advantage"
    for code, year in [("560", 2015), ("339", 2019)]:
        assert table[f"{code}|{year}"]["final_set"] == "tiebreak"


def test_challenger_entries_are_inferred() -> None:
    spec = R.rules_for("3967", 2012)
    assert spec.provenance == "inferred"
    assert spec.best_of == 3
    assert spec.source


def test_unknown_tournament_falls_back_to_tour_default() -> None:
    spec = R.rules_for("no-such-code", 2015)
    assert spec.provenance == "inferred" and spec.best_of == 3
    assert spec.final_set == "tiebreak"


def test_provenance_is_never_merged() -> None:
    """documented and inferred stay distinguishable (ground rule 7)."""
    assert R.rules_for("540", 2015).provenance != R.rules_for("3967", 2012).provenance


# --- empirical scan -------------------------------------------------------

def test_scan_does_not_hard_fail_and_logs(m: pd.DataFrame) -> None:
    disc = R.empirical_scan(m, save=False)
    assert isinstance(disc, pd.DataFrame)
    assert len(disc) / m["outcome"].eq("completed").sum() < 0.01
    assert C.RULES_DISCREPANCIES_PATH.exists()
    logged = pd.read_csv(C.RULES_DISCREPANCIES_PATH)
    assert set(logged.columns) >= {"match_id", "assigned_format", "score",
                                   "provenance", "conflict"}


def test_no_tournament_crosses_review_threshold(m: pd.DataFrame) -> None:
    """If this fails, the table entry is implicated — a human judgment call."""
    disc = R.empirical_scan(m, save=False)
    flagged = R.review_candidates(m, disc)
    assert len(flagged) == 0, flagged.to_string()


def test_conflicts_are_flagged_on_the_match_record(m: pd.DataFrame) -> None:
    disc = pd.read_csv(C.RULES_DISCREPANCIES_PATH)
    flagged = m[m["match_id"].isin(disc["match_id"])]
    assert len(flagged) == len(disc)
    assert flagged["score_string_suspect"].all()
    assert flagged["score_suspect_reason"].str.contains("rules.py").all()
