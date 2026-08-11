"""Scoring and interval machinery in scripts/backtest.py.

These cover the reporting layer, not the model: the ways a backtest can look
better or firmer than the evidence supports. All three defects fixed on
2026-08-11 were of that kind and none of them would have failed a test that
only asked whether the script runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtest import _cluster_ci, _ragged_indices, _score


def test_ragged_indices_matches_the_obvious_loop():
    rng = np.random.default_rng(0)
    for _ in range(100):
        counts = rng.integers(1, 6, 7)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        pick = rng.integers(0, 7, 7)
        want = np.concatenate([np.arange(starts[i], starts[i] + counts[i])
                               for i in pick])
        assert np.array_equal(_ragged_indices(starts[pick], counts[pick]), want)


def test_clustering_widens_the_interval():
    """The whole point: rows from one match are not independent observations.

    Twenty perfectly correlated rows per match carry the information of one.
    Resampling rows would claim an interval roughly sqrt(20) times too narrow,
    which is how a fourth-decimal difference gets read as a real one.
    """
    rng = np.random.default_rng(0)
    n_match, per = 400, 20
    truth = rng.random(n_match)
    p = np.repeat(truth, per)
    y = (rng.random(n_match) < truth).astype(float).repeat(per)

    clustered = pd.DataFrame({"match_id": np.repeat(np.arange(n_match), per),
                              "p": p, "y": y, "family": "x"})
    # Same numbers, but every row called its own match.
    as_rows = pd.DataFrame({"match_id": np.arange(n_match * per),
                            "p": p, "y": y, "family": "x"})

    lo, hi = _cluster_ci(clustered, "brier", n_boot=300)
    ilo, ihi = _cluster_ci(as_rows, "brier", n_boot=300)
    assert hi > lo and ihi > ilo
    assert (hi - lo) > 3 * (ihi - ilo)


def test_cluster_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    n_match, per = 300, 7
    p = rng.uniform(0.2, 0.8, n_match * per)
    y = (rng.random(n_match * per) < p).astype(float)
    df = pd.DataFrame({"match_id": np.repeat(np.arange(n_match), per),
                       "p": p, "y": y, "family": "x"})
    for metric in ("brier", "logloss", "ece"):
        lo, hi = _cluster_ci(df, metric, n_boot=300)
        assert lo <= _score(df)[metric] <= hi, metric


def test_score_counts_matches_not_just_rows():
    """`n` inflates with the ladder; `n_matches` is the real sample size."""
    df = pd.DataFrame({"match_id": ["m1", "m1", "m1", "m2"],
                       "p": [0.5, 0.5, 0.5, 0.5],
                       "y": [1.0, 0.0, 1.0, 0.0],
                       "family": ["totals"] * 4})
    s = _score(df)
    assert s["n"] == 4
    assert s["n_matches"] == 2


def test_crps_rows_are_excluded_from_probability_metrics():
    """_crps rows carry a game count in `p` and NaN in `y`.

    Scoring them as probabilities would be silent nonsense: a CRPS of 3.4 is
    not a probability and would land in the top reliability bin.
    """
    df = pd.DataFrame({"match_id": ["m1", "m1"],
                       "p": [0.5, 3.44],
                       "y": [1.0, np.nan],
                       "family": ["totals", "_crps"]})
    s = _score(df)
    assert s["n"] == 1
    assert s["brier"] == 0.25
    assert s["crps"] == 3.44
