"""Stage 4 tests — the inversion, the blend, and the disagreement metadata."""

from __future__ import annotations

import numpy as np
import pytest

from model import combine as CB
from model.engine import p_match
from model.rules import rules_for

BO3 = rules_for("339", 2019)
BO5 = rules_for("540", 2023)


def test_inversion_round_trips_through_the_engine() -> None:
    """The gap the grid returns must reproduce the target probability."""
    for spec in (BO3, BO5):
        for level in (1.15, 1.24, 1.36):
            for target in (0.25, 0.4, 0.5, 0.65, 0.85):
                gap = CB.elo_gap_for(target, level, spec)
                pa, pb = (level + gap) / 2, (level - gap) / 2
                got = p_match(round(pa, 6), round(pb, 6), spec)
                assert got == pytest.approx(target, abs=0.01), (spec.best_of, level, target)


def test_inversion_is_monotonic_in_the_target() -> None:
    gaps = [CB.elo_gap_for(p, 1.24, BO3) for p in np.arange(0.1, 0.91, 0.05)]
    assert all(b > a for a, b in zip(gaps, gaps[1:]))


def test_even_match_has_zero_gap() -> None:
    assert CB.elo_gap_for(0.5, 1.24, BO3) == pytest.approx(0.0, abs=1e-3)
    assert CB.elo_gap_for(0.5, 1.30, BO5) == pytest.approx(0.0, abs=1e-3)


def test_grid_probability_matches_the_engine() -> None:
    """p_from_grid is only used for fitting, but it must not be far off."""
    key = CB._spec_key(BO3)
    for level, gap in ((1.24, 0.03), (1.18, -0.05), (1.34, 0.11)):
        grid = float(CB.p_from_grid(np.array([level]), np.array([gap]), key)[0])
        exact = p_match((level + gap) / 2, (level - gap) / 2, BO3)
        assert grid == pytest.approx(exact, abs=0.005)


def test_blend_endpoints() -> None:
    a = CB.combine(0.64, 0.60, 0.45, BO3, CB.CombineParams(w=0.0), 5000, 5000)
    assert a["pa"] == pytest.approx(0.64) and a["pb"] == pytest.approx(0.60)

    b = CB.combine(0.64, 0.60, 0.45, BO3, CB.CombineParams(w=1.0), 5000, 5000)
    assert p_match(round(b["pa"], 3), round(b["pb"], 3), BO3) == pytest.approx(
        0.45, abs=0.01
    )


def test_level_is_preserved_by_the_blend() -> None:
    """Only the split moves; the level is the serve model's business."""
    for w in (0.0, 0.3, 0.7, 1.0):
        out = CB.combine(0.66, 0.58, 0.4, BO5, CB.CombineParams(w=w), 5000, 5000)
        assert out["pa"] + out["pb"] == pytest.approx(0.66 + 0.58, abs=1e-9)


def test_disagreement_is_reported_not_absorbed() -> None:
    out = CB.combine(0.64, 0.60, 0.45, BO3, CB.CombineParams(w=0.5), 5000, 5000)
    assert out["disagreement_pp"] == pytest.approx(
        100 * (out["elo_p"] - out["serve_p"]), abs=1e-9
    )
    assert abs(out["disagreement_pp"]) > 1.0
    # blending does not change what the two components said
    other = CB.combine(0.64, 0.60, 0.45, BO3, CB.CombineParams(w=0.9), 5000, 5000)
    assert other["disagreement_pp"] == pytest.approx(out["disagreement_pp"])


def test_bucketing_and_per_bucket_weights() -> None:
    p = CB.CombineParams(w=0.5, w_both_well=0.3, w_one_thin=0.6, w_both_thin=0.9,
                         thin_n=1000.0)
    assert p.weight_for(5000, 5000) == 0.3
    assert p.weight_for(5000, 100) == 0.6
    assert p.weight_for(100, 100) == 0.9
    np.testing.assert_array_equal(
        CB.bucket_of(np.array([5000, 5000, 100]), np.array([5000, 100, 100]), 1000.0),
        np.array([0, 1, 2]),
    )


def test_missing_bucket_weights_fall_back_to_overall() -> None:
    p = CB.CombineParams(w=0.42)
    assert p.weight_for(5000, 5000) == 0.42
    assert p.weight_for(10, 10) == 0.42


def test_every_output_carries_its_n() -> None:
    out = CB.combine(0.64, 0.60, 0.45, BO3, CB.CombineParams(), 4321, 99)
    assert out["n_a"] == 4321 and out["n_b"] == 99
    assert out["bucket"] == 1


def test_combine_many_matches_scalar() -> None:
    params = CB.CombineParams(w=0.4)
    keys = [CB._spec_key(BO3), CB._spec_key(BO5)]
    many = CB.combine_many(np.array([0.64, 0.66]), np.array([0.60, 0.58]),
                           np.array([0.45, 0.7]), keys, params,
                           np.array([5000.0, 5000.0]), np.array([5000.0, 5000.0]))
    for i, spec in enumerate((BO3, BO5)):
        one = CB.combine(float([0.64, 0.66][i]), float([0.60, 0.58][i]),
                         float([0.45, 0.7][i]), spec, params, 5000, 5000)
        assert many["pa"][i] == pytest.approx(one["pa"], abs=1e-9)
        assert many["pb"][i] == pytest.approx(one["pb"], abs=1e-9)
