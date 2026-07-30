"""Stage 7 tests — the two corrections must be separate and do what they claim."""

from __future__ import annotations

import numpy as np
import pytest

from model import corrections as CR
from model.engine import match_distribution, set_distribution
from model.rules import rules_for

BO3 = rules_for("339", 2019)
BO5 = rules_for("540", 2023)


def _sd(pmf: dict[int, float]) -> float:
    xs = np.array(list(pmf), dtype=float)
    ps = np.array(list(pmf.values()))
    mean = float(np.sum(xs * ps))
    return float(np.sqrt(np.sum(ps * (xs - mean) ** 2)))


def test_no_correction_is_the_plain_engine() -> None:
    plain = match_distribution(0.64, 0.60, BO3)
    same = CR.corrected_distribution(0.64, 0.60, BO3, CR.CorrectionParams())
    assert same.p_a == pytest.approx(plain.p_a, abs=1e-12)
    assert same.tiebreak_any == pytest.approx(plain.tiebreak_any, abs=1e-12)


def test_tiebreak_correction_raises_tiebreak_frequency() -> None:
    base = match_distribution(0.64, 0.60, BO3)
    for infl in (0.005, 0.01):
        d = CR.corrected_distribution(0.64, 0.60, BO3,
                                      CR.CorrectionParams(tiebreak_inflation=infl))
        assert d.tiebreak_any > base.tiebreak_any


def test_tiebreak_correction_barely_moves_the_winner() -> None:
    """It acts at the close-set states symmetrically, so it is not a level shift."""
    base = match_distribution(0.64, 0.60, BO3)
    d = CR.corrected_distribution(0.64, 0.60, BO3,
                                  CR.CorrectionParams(tiebreak_inflation=0.01))
    assert abs(d.p_a - base.p_a) < 0.001
    assert d.tiebreak_any - base.tiebreak_any > 0.004


def test_close_inflation_only_fires_when_the_set_is_close() -> None:
    plain = set_distribution(0.64, 0.60, BO3)
    infl = set_distribution(0.64, 0.60, BO3, close_inflation=0.02)
    # a 6-0 set never reaches the close states, so its probability is unchanged
    assert infl.scores[(6, 0, False)] == pytest.approx(plain.scores[(6, 0, False)],
                                                       abs=1e-12)
    assert infl.p_tiebreak > plain.p_tiebreak


def test_variance_correction_widens_the_games_distribution() -> None:
    base = match_distribution(0.64, 0.60, BO5)
    widths = []
    for sigma in (0.0, 0.02, 0.05):
        d = CR.corrected_distribution(0.64, 0.60, BO5,
                                      CR.CorrectionParams(level_sigma=sigma))
        widths.append(_sd(d.total_games_pmf()))
    assert widths[0] < widths[1] < widths[2]


def test_corrections_are_separable() -> None:
    """Each does its own job; neither substitutes for the other."""
    tb_only = CR.corrected_distribution(
        0.64, 0.60, BO3, CR.CorrectionParams(tiebreak_inflation=0.01))
    var_only = CR.corrected_distribution(
        0.64, 0.60, BO3, CR.CorrectionParams(level_sigma=0.03))
    both = CR.corrected_distribution(
        0.64, 0.60, BO3, CR.CorrectionParams(tiebreak_inflation=0.01, level_sigma=0.03))
    base = match_distribution(0.64, 0.60, BO3)

    # The variance correction alone does not close the tiebreak gap. It is not
    # orthogonal either — a wobbled level produces more close sets, so at
    # these settings it recovers about half of the tiebreak correction's
    # effect. That overlap is why the stage validates the two separately and
    # checks that neither degrades the other.
    assert (var_only.tiebreak_any - base.tiebreak_any) < 0.6 * (
        tb_only.tiebreak_any - base.tiebreak_any)
    # the tiebreak correction alone does not widen the games distribution much
    assert (_sd(tb_only.total_games_pmf()) - _sd(base.total_games_pmf())) < 0.5 * (
        _sd(var_only.total_games_pmf()) - _sd(base.total_games_pmf()))
    # applied together, both effects survive
    assert both.tiebreak_any > base.tiebreak_any
    assert _sd(both.total_games_pmf()) > _sd(base.total_games_pmf())


def test_corrected_distribution_is_still_a_distribution() -> None:
    d = CR.corrected_distribution(0.66, 0.58, BO5,
                                  CR.CorrectionParams(tiebreak_inflation=0.01,
                                                      level_sigma=0.03))
    assert sum(d.games.values()) + d.truncated_mass == pytest.approx(1.0, abs=1e-6)
    assert sum(d.sets.values()) + d.truncated_mass == pytest.approx(1.0, abs=1e-6)
    assert 0 < d.p_a < 1


def test_provenance_weights_implement_all_three_schemes() -> None:
    prov = np.array(["documented", "inferred", "documented", "inferred"])
    doc_only = CR.provenance_weights(prov, CR.CorrectionParams(scheme="documented_only"))
    np.testing.assert_array_equal(doc_only, [1, 0, 1, 0])

    pooled = CR.provenance_weights(prov, CR.CorrectionParams(scheme="pooled"))
    np.testing.assert_array_equal(pooled, [1, 1, 1, 1])

    down = CR.provenance_weights(prov, CR.CorrectionParams(
        scheme="downweight_inferred", inferred_weight=0.25))
    np.testing.assert_array_equal(down, [1, 0.25, 1, 0.25])
    assert set(CR.SCHEMES) == {"documented_only", "pooled", "downweight_inferred"}


def test_tiebreak_gap_measurement() -> None:
    obs = np.array([1.0, 0.0, 1.0, 0.0])
    pred = np.array([0.5, 0.5, 0.5, 0.5])
    assert CR.measure_tiebreak_gap(obs, pred) == pytest.approx(0.0)
    assert CR.measure_tiebreak_gap(np.ones(4), pred) == pytest.approx(0.5)
    # weights select a subset
    w = np.array([1.0, 0.0, 1.0, 0.0])
    assert CR.measure_tiebreak_gap(obs, pred, w) == pytest.approx(0.5)


def test_pit_and_coverage_on_a_known_forecast() -> None:
    rng = np.random.default_rng(0)
    # a perfectly calibrated continuous forecast gives uniform PIT
    u = rng.random(20_000)
    assert CR.pit_deviation(u) < 0.01
    assert CR.coverage(u) == pytest.approx(0.8, abs=0.02)
    # an overconfident forecast piles mass in the tails
    tails = np.concatenate([rng.random(10_000) * 0.1,
                            0.9 + rng.random(10_000) * 0.1])
    assert CR.pit_deviation(tails) > 0.05
    assert CR.coverage(tails) < 0.2
