"""Stage 2 tests — as-of discipline first, everything else second.

The gate itself (beat career-average and last-10 baselines on TUNE) is run by
``scripts/fit_stage2.py`` and recorded in reports/stage_validations/stage_2.md;
these tests pin the properties that make that number trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import constants as C
from model import player_rates as PR


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    return m[m["in_scope"] & m["split"].eq("fit")]


@pytest.fixture(scope="module")
def small(matches: pd.DataFrame) -> pd.DataFrame:
    """Two seasons is enough to exercise the machinery quickly."""
    return matches[matches["season_file"].isin((2010, 2011))]


@pytest.fixture(scope="module")
def asof(small: pd.DataFrame) -> pd.DataFrame:
    return PR.build_asof(small, PR.RateParams())


def test_only_usable_matches_contribute(small: pd.DataFrame, asof: pd.DataFrame) -> None:
    """Retirements, walkovers, suspect scores and out-of-scope events are out."""
    assert len(asof) == int(small["serve_stats_valid"].sum())
    used = small[small["match_id"].isin(asof["match_id"])]
    assert used["outcome"].eq("completed").all()
    assert not used["score_string_suspect"].any()
    assert used["in_scope"].all()


def test_as_of_discipline_no_lookahead(small: pd.DataFrame) -> None:
    """Truncating the future cannot change a rate computed in the past.

    This is the check that catches lookahead reintroduced through a secondary
    path — the accumulators, the opponent adjustment, or the league anchor.
    """
    params = PR.RateParams()
    full = PR.serve_rates(PR.build_asof(small, params), params)
    cut = pd.Timestamp("2011-01-01").date()
    truncated = small[small["date"] < cut]
    part = PR.serve_rates(PR.build_asof(truncated, params), params)

    merged = full.merge(part, on="match_id", suffixes=("_full", "_part"))
    assert len(merged) > 1000
    for side in ("w", "l"):
        np.testing.assert_allclose(
            merged[f"{side}_rate_full"], merged[f"{side}_rate_part"], atol=1e-12
        )


def test_every_rate_carries_its_n(asof: pd.DataFrame) -> None:
    rates = PR.serve_rates(asof, PR.RateParams())
    for side in ("w", "l"):
        assert (rates[f"{side}_n"] > 0).all()
        assert rates[f"{side}_n"].notna().all()
        assert rates[f"{side}_nm"].notna().all()
    assert rates["w_rate"].between(0, 1).all()


def test_first_match_of_a_player_falls_back_to_the_league_prior(asof: pd.DataFrame) -> None:
    """A true debut — no serve points anywhere — prices at the level prior.

    Note ``w_nm`` counts matches on THIS surface; a player with a hard-court
    record but no clay match has w_nm == 0 and is not a debutant.
    """
    rates = PR.serve_rates(asof, PR.RateParams())
    rates["sa_pl"] = asof["w_sa_pl"].to_numpy()
    debutants = rates[rates["sa_pl"] == 0]
    assert len(debutants) > 100
    lg = debutants["league"].fillna(0.62)
    np.testing.assert_allclose(debutants["w_rate"], lg, atol=1e-9)


def test_vectorized_matches_scalar_reference(asof: pd.DataFrame) -> None:
    params = PR.RateParams()
    vec = PR.serve_rates(asof, params)
    sample = asof.sample(200, random_state=0)
    for i, row in sample.iterrows():
        ref = PR.rate_from_state(row["w_ss_won"], row["w_ss_pl"], row["w_sa_won"],
                                 row["w_sa_pl"], row["w_nm"], row["league"], params)
        got = vec.loc[vec["match_id"] == row["match_id"], "w_rate"].iat[0]
        assert ref.value == pytest.approx(got, abs=1e-12)
        assert ref.n > 0


def test_rate_requires_its_sample_size() -> None:
    with pytest.raises(AssertionError):
        PR.Rate(value=0.6, n=-1.0, n_matches=3, level="atp", surface="Clay")


def test_shrinkage_is_toward_the_players_own_level(small: pd.DataFrame) -> None:
    """A Challenger player never shrinks toward the ATP mean."""
    params = PR.RateParams(shrink_n0=1e6)  # shrink almost entirely to the prior
    rates = PR.serve_rates(PR.build_asof(small, params), params)
    by_level = rates.groupby("level_group")["w_rate"].mean()
    assert set(by_level.index) == {"atp", "chall"}
    assert abs(by_level["atp"] - by_level["chall"]) > 1e-4


def test_recency_toggle_changes_nothing_else(small: pd.DataFrame) -> None:
    off = PR.RateParams(recency=False)
    rates = PR.serve_rates(PR.build_asof(small, off), off)
    assert rates["w_rate"].between(0, 1).all()
    assert (rates["w_n"] > 0).all()


def test_surface_split_toggle(small: pd.DataFrame) -> None:
    on = PR.RateParams(surface_split=True)
    off = PR.RateParams(surface_split=False)
    a = PR.serve_rates(PR.build_asof(small, on), on)
    b = PR.serve_rates(PR.build_asof(small, off), off)
    assert not np.allclose(a["w_rate"], b["w_rate"])


def test_opponent_adjustment_converges(small: pd.DataFrame) -> None:
    """Unconverged is an error, never a silent partial fit."""
    PR.build_asof(small, PR.RateParams(opponent_adjust=True))  # must not raise
    off = PR.build_asof(small, PR.RateParams(opponent_adjust=False))
    on = PR.build_asof(small, PR.RateParams(opponent_adjust=True))
    assert not np.allclose(on["w_ss_won"], off["w_ss_won"])


def test_league_anchor_is_stable(asof: pd.DataFrame) -> None:
    """The identifiability anchor must not drift with the adjustment.

    Judged once the anchor has real evidence behind it — the first few
    matches of the dataset are a handful of serve points, not a tour average.
    """
    lg = asof.loc[asof["league_n"] > 5000, "league"].dropna()
    assert len(lg) > 1000
    assert lg.between(0.55, 0.70).all(), (lg.min(), lg.max())


def test_expected_spw_puts_the_opponent_back_in() -> None:
    league = 0.62
    neutral = PR.expected_spw(0.65, 1 - league, league)
    assert neutral == pytest.approx(0.65, abs=1e-9)
    vs_good_returner = PR.expected_spw(0.65, 1 - league + 0.03, league)
    assert vs_good_returner < neutral
    assert PR.expected_spw(0.65, np.nan, league) == pytest.approx(0.65)


def test_metrics_are_weighted_by_serve_points() -> None:
    """A match with more serve points carries more weight in the log-loss."""
    actual = np.array([0.6, 0.6])
    pred = np.array([0.6, 0.3])  # second prediction is badly wrong
    good_heavy = PR.spw_metrics(actual, pred, np.array([100.0, 1.0]))
    bad_heavy = PR.spw_metrics(actual, pred, np.array([1.0, 100.0]))
    assert good_heavy["logloss"] < bad_heavy["logloss"]
    assert good_heavy["mae"] == pytest.approx(bad_heavy["mae"])


def test_baselines_are_as_of(small: pd.DataFrame) -> None:
    b = PR.baseline_rates(small, last_k=10)
    assert b["w_pred"].isna().sum() > 0  # a debut has no history to average
    assert b["w_pred"].dropna().between(0, 1).all()
    career = PR.baseline_rates(small, last_k=None)
    assert not np.allclose(
        career["w_pred"].fillna(0.62), b["w_pred"].fillna(0.62)
    )
