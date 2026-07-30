"""Validation for the four-way split definition (ground rule 1).

Gate: boundaries are ordered, contiguous with no gap and no overlap, every
date maps to exactly one split, and the holdout cutoff is the last boundary.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from model import constants as C

BOUNDARIES = [
    (C.DATA_START, C.FIT_END, "fit"),
    (C.TUNE_START, C.TUNE_END, "tune"),
    (C.TEST_START, C.TEST_END, "test"),
    (C.RESERVE_START, C.RESERVE_END, "reserve"),
]


def test_boundaries_ordered_and_contiguous() -> None:
    prev_end = None
    for start, end, _ in BOUNDARIES:
        assert start <= end
        if prev_end is not None:
            assert start == prev_end + timedelta(days=1), (prev_end, start)
        prev_end = end
    assert C.HOLDOUT_CUTOFF == prev_end + timedelta(days=1)


@pytest.mark.parametrize("start,end,name", BOUNDARIES)
def test_split_of_endpoints(start: date, end: date, name: str) -> None:
    assert C.split_of(start) == name
    assert C.split_of(end) == name


def test_holdout_and_pre_data() -> None:
    assert C.split_of(C.HOLDOUT_CUTOFF) == "holdout"
    assert C.split_of(C.HOLDOUT_CUTOFF + timedelta(days=3650)) == "holdout"
    assert C.split_of(C.DATA_START - timedelta(days=1)) == "pre_data"


def test_assert_no_holdout() -> None:
    C.assert_no_holdout([C.FIT_END, C.TUNE_START, C.TEST_END])
    with pytest.raises(ValueError):
        C.assert_no_holdout([C.FIT_END, C.HOLDOUT_CUTOFF])


def test_nothing_fitted_is_hardcoded_here() -> None:
    """constants.py holds boundaries/paths/seeds only — no fitted values."""
    assert C.FITTED_PARAMS_PATH.name == "fitted_params.json"
    assert C.OVERFIT_SIGNAL_MULTIPLE > 0
    assert 0 < C.TAIL_MASS_THRESHOLD < 1e-6
