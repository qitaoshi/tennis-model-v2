"""Stage 3 tests — rating mechanics and the exclusions ground rule 6 fixes.

The gate itself (calibration + Brier vs a rankings baseline + the cross-level
check) is run by ``scripts/fit_stage3.py`` and recorded in
reports/stage_validations/stage_3.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import constants as C
from model import elo as E


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    return m[m["split"].isin(("fit", "tune"))]


@pytest.fixture(scope="module")
def small(matches: pd.DataFrame) -> pd.DataFrame:
    return matches[matches["season_file"].isin((2010, 2011, 2012))]


@pytest.fixture(scope="module")
def res(small: pd.DataFrame) -> pd.DataFrame:
    return E.run_elo(small, E.EloParams())


def test_expected_score_is_standard_elo() -> None:
    assert E.expected_score(1500, 1500) == pytest.approx(0.5)
    assert E.expected_score(1900, 1500) == pytest.approx(1 / (1 + 10 ** -1))
    assert E.expected_score(1500, 1900) + E.expected_score(1900, 1500) == pytest.approx(1.0)


def test_only_rating_outcomes_update(small: pd.DataFrame, res: pd.DataFrame) -> None:
    """Retirements count; walkovers, defaults and unknowns never do."""
    used = small[small["match_id"].isin(res["match_id"])]
    assert set(used["outcome"]) == {"completed", "retired"}
    assert used["in_scope"].all()
    assert (used["outcome"] == "retired").sum() > 0


def test_suspect_scores_still_update_ratings(small: pd.DataFrame, res: pd.DataFrame) -> None:
    """The win/loss outcome is not in doubt even when the score string is."""
    suspect = small[small["score_string_suspect"] & small["in_scope"]
                    & small["outcome"].isin(E.RATING_OUTCOMES)]
    assert len(suspect) > 0
    assert suspect["match_id"].isin(res["match_id"]).all()


def test_ratings_are_as_of(small: pd.DataFrame) -> None:
    """Truncating the future cannot change a rating computed in the past."""
    params = E.EloParams()
    full = E.run_elo(small, params)
    cut = pd.Timestamp("2012-01-01").date()
    part = E.run_elo(small[small["date"] < cut], params)
    merged = full.merge(part, on="match_id", suffixes=("_f", "_p"))
    assert len(merged) > 5000
    np.testing.assert_allclose(merged["p_winner_f"], merged["p_winner_p"], atol=1e-12)


def test_first_match_is_the_level_default(res: pd.DataFrame) -> None:
    debut = res[(res["w_n"] == 0) & (res["l_n"] == 0)]
    assert len(debut) > 50
    assert debut["p_winner"].round(6).eq(0.5).all()


def test_new_challenger_players_start_lower(small: pd.DataFrame) -> None:
    gap = E.EloParams(level_gap=100.0)
    none = E.EloParams(level_gap=0.0)
    a = E.run_elo(small, gap)
    b = E.run_elo(small, none)
    assert not np.allclose(a["p_winner"], b["p_winner"])


def test_inactivity_regresses_toward_the_reference() -> None:
    p = E.EloParams(inactivity_half_life=365.0)
    r = E._Ratings(p)
    r.overall["x"] = 1900.0
    r.last["x"] = 0.0
    r.n["x"] = 5
    fresh, _ = r.get("x", "Hard", 0.0, "atp")
    after_year, _ = r.get("x", "Hard", 365.0, "atp")
    after_decade, _ = r.get("x", "Hard", 3650.0, "atp")
    assert fresh == pytest.approx(1900.0)
    assert after_year == pytest.approx(1700.0)          # halfway to 1500
    assert 1500.0 < after_decade < 1510.0
    off = E._Ratings(E.EloParams(inactivity_half_life=None))
    off.overall["x"], off.last["x"], off.n["x"] = 1900.0, 0.0, 5
    assert off.get("x", "Hard", 3650.0, "atp")[0] == pytest.approx(1900.0)


def test_surface_blend_weight_matters(small: pd.DataFrame) -> None:
    a = E.run_elo(small, E.EloParams(surface_weight=0.0))
    b = E.run_elo(small, E.EloParams(surface_weight=1.0))
    assert not np.allclose(a["p_winner"], b["p_winner"])


def test_challenger_k_multiplier_matters(small: pd.DataFrame) -> None:
    a = E.run_elo(small, E.EloParams(k_chall_mult=0.5))
    b = E.run_elo(small, E.EloParams(k_chall_mult=1.0))
    assert not np.allclose(a["p_winner"], b["p_winner"])


def test_both_tours_update_ratings(res: pd.DataFrame) -> None:
    assert set(res["level_group"]) == {"atp", "chall"}
    assert (res["w_chall_share"] > 0).any()


def test_decile_calibration_is_symmetrized(res: pd.DataFrame) -> None:
    """Without symmetrizing, every observed rate would be 1 by construction."""
    calib = E.decile_calibration(res["p_winner"].to_numpy())
    assert len(calib) == 10
    assert calib["observed"].min() < 0.4 and calib["observed"].max() > 0.6
    assert calib["n"].sum() == 2 * len(res)


def test_metrics_orientation() -> None:
    perfect = np.ones(100)
    assert E.log_loss(perfect) == pytest.approx(0.0, abs=1e-8)  # clipped at 1-1e-9
    assert E.brier(perfect) == pytest.approx(0.0)
    coin = np.full(100, 0.5)
    assert E.log_loss(coin) == pytest.approx(np.log(2))
    assert E.brier(coin) == pytest.approx(0.25)


def test_ranking_baseline_is_fitted_on_fit_only(res: pd.DataFrame) -> None:
    fit_mask = (res["split"] == "fit").to_numpy()
    p = E.ranking_baseline(res, fit_mask)
    assert np.isfinite(p).all()
    assert (p > 0).all() and (p < 1).all()
    # a baseline fitted on FIT must not change when TUNE rows are perturbed
    other = res.copy()
    other.loc[~fit_mask, "winner_rank"] = 1.0
    p2 = E.ranking_baseline(other, fit_mask)
    np.testing.assert_allclose(p[fit_mask], p2[fit_mask], atol=1e-12)


def test_cross_level_check_reports(res: pd.DataFrame) -> None:
    out = E.cross_level_check(res)
    assert out["n"] > 0
    assert 0.0 < out["predicted_chall_winrate"] < 1.0
    assert "gap" in out
