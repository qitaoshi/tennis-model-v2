"""Stage 6 tests — keying, as-of history, provenance, and level-only effect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import constants as C
from model import venue as V


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    return m[m["in_scope"] & m["split"].isin(("fit", "tune"))]


@pytest.fixture(scope="module")
def asof(matches: pd.DataFrame) -> pd.DataFrame:
    return V.build_asof_index(matches, V.VenueParams())


def test_key_is_venue_and_surface_never_venue_alone(asof: pd.DataFrame) -> None:
    keys = pd.Series(asof["venue_key"].tolist())
    assert keys.map(len).eq(2).all()
    # a venue that changed surface must appear under two distinct keys
    codes = keys.map(lambda k: k[0])
    per_code = keys.groupby(codes).nunique()
    assert (per_code > 1).sum() > 15, "surface changes must split the key"
    # Stuttgart (code 321) moved from clay to grass in 2015
    assert per_code.get("321", 0) == 2
    assert {k for k in keys if k[0] == "321"} == {("321", "Clay"), ("321", "Grass")}


def test_multiplier_history_is_as_of(matches: pd.DataFrame) -> None:
    """Truncating the future cannot change a multiplier used in the past."""
    params = V.VenueParams()
    full = V.build_asof_index(matches, params)
    cut = pd.Timestamp("2015-01-01").date()
    part = V.build_asof_index(matches[matches["date"] < cut], params)
    merged = full.merge(part, on="match_id", suffixes=("_f", "_p"))
    assert len(merged) > 20_000
    np.testing.assert_allclose(merged["multiplier_f"], merged["multiplier_p"],
                               atol=1e-12)


def test_suspect_and_non_completed_matches_are_excluded(matches: pd.DataFrame,
                                                        asof: pd.DataFrame) -> None:
    used = matches[matches["match_id"].isin(asof["match_id"])]
    assert used["outcome"].eq("completed").all()
    assert not used["score_string_suspect"].any()


def test_first_edition_venues_are_flagged_unmeasured(asof: pd.DataFrame) -> None:
    first = asof[~asof["measured"]]
    assert len(first) > 100
    assert (first["multiplier"] == 1.0).all(), "unmeasured must get the surface average"
    assert (first["venue_n_points"] == 0).all()


def test_shrinkage_pulls_thin_venues_toward_the_surface_mean(matches: pd.DataFrame) -> None:
    light = V.build_asof_index(matches, V.VenueParams(shrink_n0=1_000.0))
    heavy = V.build_asof_index(matches, V.VenueParams(shrink_n0=500_000.0))
    thin = light["venue_n_matches"].between(1, 15)
    assert thin.sum() > 100
    assert ((heavy.loc[thin, "multiplier"] - 1).abs().mean()
            < (light.loc[thin, "multiplier"] - 1).abs().mean())
    # a venue with 15 matches of history should barely move
    assert (heavy.loc[thin, "multiplier"] - 1).abs().max() < 0.02


def test_multipliers_are_plausible(asof: pd.DataFrame) -> None:
    m = asof.loc[asof["measured"], "multiplier"]
    assert m.between(0.85, 1.15).all(), (m.min(), m.max())
    assert 0.98 < m.mean() < 1.02


def test_multiplier_moves_the_level_not_the_split() -> None:
    for mult in (0.95, 1.0, 1.06):
        pa, pb = V.apply_multiplier(0.64, 0.60, mult)
        assert pa - pb == pytest.approx(0.04, abs=1e-9)
        assert pa + pb == pytest.approx(1.24 * mult, abs=1e-9)


def test_apply_multiplier_stays_in_range() -> None:
    pa, pb = V.apply_multiplier(0.85, 0.84, 1.5)
    assert 0 < pb <= pa < 1
    pa, pb = V.apply_multiplier(0.40, 0.38, 0.5)
    assert 0 < pb <= pa < 1


def test_index_for_returns_provenance(asof: pd.DataFrame) -> None:
    mid = asof.loc[asof["measured"], "match_id"].iloc[-1]
    idx = V.index_for(asof, mid)
    assert idx.measured and idx.n_matches > 0 and idx.n_serve_points > 0
    assert len(idx.key) == 2
    assert idx.apply_to_level(1.24) == pytest.approx(1.24 * idx.multiplier)


def test_indoor_key_option_changes_the_keying(matches: pd.DataFrame) -> None:
    plain = V.build_asof_index(matches, V.VenueParams(use_indoor=False))
    indoor = V.build_asof_index(matches, V.VenueParams(use_indoor=True))
    assert pd.Series(indoor["venue_key"].tolist()).map(len).eq(3).all()
    assert not np.allclose(plain["multiplier"], indoor["multiplier"])


def test_residual_estimate_removes_field_strength(matches: pd.DataFrame) -> None:
    """With an expectation supplied, the venue effect is a residual.

    The raw estimate confounds court speed with who plays there, so it must
    be the wider of the two and only partly correlated with the residual one.
    """
    import json
    from model import player_rates as PR

    s2 = json.loads(C.FITTED_PARAMS_PATH.read_text())["stage_2"]
    rp = PR.RateParams(half_life_days=s2["half_life_days"],
                       surface_pool=s2["surface_pool"], shrink_n0=s2["shrink_n0"])
    rates = PR.serve_rates(PR.build_asof(matches, rp), rp)
    pa = [PR.expected_spw(a, o, l) for a, o, l
          in zip(rates["w_rate"], rates["l_ret"], rates["league"])]
    pb = [PR.expected_spw(a, o, l) for a, o, l
          in zip(rates["l_rate"], rates["w_ret"], rates["league"])]
    expected = pd.Series(
        ((np.array(pa) * rates["w_svpt"] + np.array(pb) * rates["l_svpt"])
         / (rates["w_svpt"] + rates["l_svpt"])).to_numpy(),
        index=rates["match_id"].to_numpy())

    params = V.VenueParams(shrink_n0=20_000.0)
    residual = V.build_asof_index(matches, params, expected=expected)
    raw = V.build_asof_index(matches, params)

    r_measured = residual.loc[residual["measured"], "multiplier"]
    w_measured = raw.loc[raw["measured"], "multiplier"]
    assert r_measured.std() < w_measured.std(), "residual must be the tighter estimate"
    assert r_measured.mean() == pytest.approx(1.0, abs=0.01)
    assert r_measured.between(0.9, 1.1).all()
