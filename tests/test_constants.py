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
]


def test_boundaries_ordered_and_cover_everything_to_the_cutoff() -> None:
    """Windows are ordered, and burned windows fill the gaps left behind."""
    for start, end, _ in BOUNDARIES:
        assert start <= end
    assert C.FIT_END < C.TUNE_START <= C.TUNE_END < C.TEST_START <= C.TEST_END
    # The spent window sits AFTER the replacement TEST, which is the whole
    # reason it cannot be recycled into FIT the way a burned window is.
    assert C.TEST_END < C.SPENT_START <= C.SPENT_END < C.HOLDOUT_CUTOFF
    assert C.TEST_END < C.HOLDOUT_CUTOFF

    # every day from DATA_START to the cutoff maps to exactly one split
    d = C.DATA_START
    seen = set()
    while d < C.HOLDOUT_CUTOFF:
        seen.add(C.split_of(d))
        d += timedelta(days=1)
    assert seen <= {"fit", "tune", "burned_test", "spent", "test", "reserve"}
    assert {"fit", "tune", "test"} <= seen


def test_burned_windows_keep_their_own_label() -> None:
    """A spent TEST window must never quietly become TUNE."""
    assert C.BURNED_TEST_WINDOWS
    for lo, hi in C.BURNED_TEST_WINDOWS:
        assert C.split_of(lo) == "burned_test"
        assert C.split_of(hi) == "burned_test"
        assert hi < C.TEST_START
        assert C.split_of(lo) != "tune"


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
