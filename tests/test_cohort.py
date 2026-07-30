"""Stage 5 tests.

The centrepiece is the lookahead test MODEL_PROMPT.md asks for by name:
pricing a historical match must use no feature input AND no standardization
statistic that post-dates it. That is a separate code path from Stage 2 and
the classic way lookahead sneaks back in through a kNN prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import cohort as CH
from model import constants as C


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    return m[m["in_scope"] & m["split"].isin(("fit", "tune"))]


@pytest.fixture(scope="module")
def built(matches: pd.DataFrame):
    params = CH.CohortParams()
    snaps, history = CH.build_snapshots(matches, params)
    return snaps, history, params


def test_snapshots_are_ordered_and_populated(built) -> None:
    snaps, _, _ = built
    assert len(snaps) > 20
    assert all(a.date_ord < b.date_ord for a, b in zip(snaps, snaps[1:]))
    assert len(snaps[-1].player_ids) > 100


def test_no_lookahead_in_features_or_standardization(matches: pd.DataFrame) -> None:
    """Truncating everything after a date cannot change earlier snapshots.

    This covers both paths at once: the per-player features and the
    population mean/std used to z-score them.
    """
    params = CH.CohortParams()
    cut = pd.Timestamp("2018-01-01").date()
    full_snaps, full_hist = CH.build_snapshots(matches, params)
    part_snaps, part_hist = CH.build_snapshots(matches[matches["date"] < cut], params)

    cut_ord = pd.Timestamp(cut).toordinal()
    full_by_date = {s.date_ord: s for s in full_snaps}
    checked = 0
    for s in part_snaps:
        if s.date_ord > cut_ord:
            continue
        f = full_by_date[s.date_ord]
        np.testing.assert_allclose(s.mean, f.mean, atol=1e-12)   # standardization
        np.testing.assert_allclose(s.std, f.std, atol=1e-12)
        np.testing.assert_array_equal(s.player_ids, f.player_ids)  # features
        np.testing.assert_allclose(s.serve_pw, f.serve_pw, atol=1e-12)
        checked += 1
    assert checked > 10


def test_pricing_a_historical_match_uses_only_prior_data(matches: pd.DataFrame) -> None:
    """The as-of contract, stated the way the spec states it."""
    params = CH.CohortParams()
    snaps, history = CH.build_snapshots(matches, params)
    mid_2015 = matches[(matches["date"] >= pd.Timestamp("2015-06-01").date())
                       & (matches["date"] < pd.Timestamp("2015-07-01").date())]
    match = mid_2015.iloc[0]
    t = pd.Timestamp(match["date"]).toordinal()
    snap = CH.snapshot_for(snaps, t)
    assert snap is not None
    assert snap.date_ord <= t, "snapshot must not post-date the match"

    # and the same snapshot is reproduced from truncated data
    truncated = matches[matches["date"] < match["date"]]
    snaps_t, _ = CH.build_snapshots(truncated, params)
    snap_t = CH.snapshot_for(snaps_t, t)
    assert snap_t.date_ord == snap.date_ord
    np.testing.assert_allclose(snap_t.mean, snap.mean, atol=1e-12)
    np.testing.assert_allclose(snap_t.std, snap.std, atol=1e-12)


def test_snapshot_lookup_never_returns_a_future_snapshot(built) -> None:
    snaps, _, _ = built
    for probe in (snaps[3].date_ord - 1, snaps[3].date_ord, snaps[3].date_ord + 5):
        got = CH.snapshot_for(snaps, probe)
        assert got.date_ord <= probe
    assert CH.snapshot_for(snaps, snaps[0].date_ord - 1000) is None


def test_style_vector_shape_and_missing_handling(built) -> None:
    snaps, history, params = built
    snap = snaps[-1]
    assert snap.z.shape[1] == len(CH.FEATURES)
    assert np.isfinite(snap.z).all(), "missing features must be imputed, not NaN"
    assert np.isfinite(snap.mean).all() and (snap.std > 0).all()

    # a vector with missing height and hand still produces a prior
    vec = snap.mean.copy()
    vec[CH.FEATURES.index("height")] = np.nan
    vec[CH.FEATURES.index("hand_left")] = np.nan
    s, r, idx = CH.cohort_prior(vec, snap, params)
    assert np.isfinite(s) and np.isfinite(r) and len(idx) == params.k


def test_cohort_prior_differs_from_the_flat_average(built) -> None:
    """If it returned the population mean there would be no stage here."""
    snaps, _, params = built
    snap = snaps[-1]
    big_server = snap.mean.copy()
    big_server[CH.FEATURES.index("ace_rate")] += 2 * snap.std[CH.FEATURES.index("ace_rate")]
    big_server[CH.FEATURES.index("serve_pw")] += 1.5 * snap.std[CH.FEATURES.index("serve_pw")]
    s, _, _ = CH.cohort_prior(big_server, snap, params)
    assert s > snap.mean[CH.FEATURES.index("serve_pw")] + 0.01

    grinder = snap.mean.copy()
    grinder[CH.FEATURES.index("serve_pw")] -= 1.5 * snap.std[CH.FEATURES.index("serve_pw")]
    s2, _, _ = CH.cohort_prior(grinder, snap, params)
    assert s2 < s


def test_k_controls_neighbourhood_size(built) -> None:
    snaps, _, _ = built
    snap = snaps[-1]
    vec = snap.mean.copy()
    for k in (5, 20, 40):
        _, _, idx = CH.cohort_prior(vec, snap, CH.CohortParams(k=k))
        assert len(idx) == k


def test_neighbours_are_well_sampled_only(built) -> None:
    snaps, history, params = built
    snap = snaps[-1]
    assert len(snap.player_ids) == len(snap.z) == len(snap.serve_pw)
    # every neighbour rate is a plausible serve rate, not an artefact
    assert ((snap.serve_pw > 0.4) & (snap.serve_pw < 0.85)).all()


def test_empty_snapshot_is_handled() -> None:
    empty = CH.Snapshot(0, np.array([]), np.zeros((0, len(CH.FEATURES))),
                        np.zeros(0), np.zeros(0), np.zeros(len(CH.FEATURES)),
                        np.ones(len(CH.FEATURES)), {})
    s, r, idx = CH.cohort_prior(np.zeros(len(CH.FEATURES)), empty, CH.CohortParams())
    assert np.isnan(s) and np.isnan(r) and len(idx) == 0


def test_vs_cohort_totals_are_as_of(matches: pd.DataFrame) -> None:
    """The opponent-vs-cohort record must not see the match it prices."""
    small = matches[matches["season_file"].isin((2010, 2011))]
    snaps, _ = CH.build_snapshots(small, CH.CohortParams())
    vs = CH.vs_cohort_totals(small, snaps)
    assert len(vs["match_id"]) == int(small["serve_stats_valid"].sum())
    # the first match of the dataset can have no prior record
    assert vs["w_vs_svpt"][0] == 0 and vs["w_vs_spw"][0] == 0
    assert (vs["w_vs_spw"] <= vs["w_vs_svpt"]).all()
    assert set(np.unique(vs["w_opp_tercile"])) <= {0, 1, 2}


def test_opponent_vs_cohort_rate_shrinks_toward_overall() -> None:
    # no record against the tier: unchanged
    assert CH.opponent_vs_cohort_rate(0, 0, 0.64, 200) == pytest.approx(0.64)
    # disabled: unchanged
    assert CH.opponent_vs_cohort_rate(1000, 700, 0.64, 0) == pytest.approx(0.64)
    # a real record moves it, but not all the way
    got = CH.opponent_vs_cohort_rate(1000, 700, 0.64, 200)
    assert 0.64 < got < 0.70
    # more evidence moves it further
    more = CH.opponent_vs_cohort_rate(5000, 3500, 0.64, 200)
    assert more > got


def test_tercile_boundaries_are_as_of(built) -> None:
    snaps, _, _ = built
    lo, hi = CH.return_terciles(snaps[-1])
    assert 0.3 < lo < hi < 0.45
    assert CH.tercile_of(lo - 0.01, (lo, hi)) == 0
    assert CH.tercile_of((lo + hi) / 2, (lo, hi)) == 1
    assert CH.tercile_of(hi + 0.01, (lo, hi)) == 2
