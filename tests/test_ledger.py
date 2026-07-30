"""Ledger semantics (ground rule 2).

The overfit signal must be reproducible from the file alone: two people
reading the ledger have to reach the same list of implicated stages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model import constants as C
from model import ledger


@pytest.fixture
def tmp_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "tune_ledger.json"
    monkeypatch.setattr(C, "TUNE_LEDGER_PATH", p)
    monkeypatch.setattr(ledger.C, "TUNE_LEDGER_PATH", p)
    return p


def _append(stage: str, tune_gain: float, folds: list[float], frozen: bool,
            selected=None) -> None:
    ledger.append(stage=stage, metric="m", selected=selected or {"k": 1},
                  tune_gain=tune_gain, fit_rolling_origin_gains=folds,
                  baseline="baseline", frozen=frozen)


def test_append_is_append_only(tmp_ledger: Path) -> None:
    _append("stage_2", 0.001, [0.001, 0.0011, 0.0009], False)
    _append("stage_2", 0.002, [0.001, 0.0011, 0.0009], True)
    assert len(ledger.read()) == 2
    assert len(json.loads(tmp_ledger.read_text())) == 2


def test_active_lineage_is_frozen_only_and_latest(tmp_ledger: Path) -> None:
    _append("stage_2", 0.001, [0.001], False)
    _append("stage_2", 0.002, [0.001], True)
    _append("stage_3", 0.003, [0.001], False)
    lineage = ledger.active_lineage()
    assert [e["stage"] for e in lineage] == ["stage_2"]
    assert lineage[0]["tune_gain"] == 0.002

    # a superseding frozen entry replaces it in the lineage but not in the file
    _append("stage_2", 0.004, [0.001], True)
    assert ledger.active_lineage()[0]["tune_gain"] == 0.004
    assert len(ledger.read()) == 4


def test_fold_spread_is_recorded(tmp_ledger: Path) -> None:
    e = ledger.append(stage="s", metric="m", selected={}, tune_gain=0.01,
                      fit_rolling_origin_gains=[0.01, 0.02, 0.03],
                      baseline="b", tune_metric_value=0.5)
    assert e["fit_rolling_origin_gain_mean"] == pytest.approx(0.02)
    assert e["fit_rolling_origin_gain_std"] == pytest.approx(0.01)
    assert e["tune_metric_value"] == 0.5


def test_overfit_flag_uses_the_fixed_multiple(tmp_ledger: Path) -> None:
    """Flag the excess of TUNE gain over FIT gain, not the gain itself."""
    folds = [0.010, 0.012, 0.008, 0.011, 0.009]  # mean 0.010, std ~0.00158
    mean, std = 0.010, 0.0015811388300841892
    envelope = C.OVERFIT_SIGNAL_MULTIPLE * std

    _append("reproduces_fit", mean, folds, True)          # excess 0
    _append("slightly_better", mean + envelope * 0.5, folds, True)
    _append("underperforms", mean * 0.5, folds, True)     # negative excess
    _append("suspicious", mean + envelope * 1.5, folds, True)

    flagged = {e["stage"] for e in ledger.overfit_flags()}
    assert flagged == {"suspicious"}
    entry = ledger.overfit_flags()[0]
    assert entry["excess_over_fit_gain"] == pytest.approx(envelope * 1.5)
    assert entry["envelope"] == pytest.approx(envelope)


def test_a_stage_that_simply_works_is_not_flagged(tmp_ledger: Path) -> None:
    """A consistent, real gain must not look like overfitting."""
    folds = [0.00111, 0.00107, 0.00098, 0.00102, 0.00124]  # Stage 2's actual folds
    _append("stage_2", 0.00071, folds, True)               # actual TUNE gain
    assert ledger.overfit_flags() == []


def test_unfrozen_entries_are_never_flagged(tmp_ledger: Path) -> None:
    folds = [0.010, 0.012, 0.008]
    _append("dev_only", 10.0, folds, False)
    assert ledger.overfit_flags() == []


def test_real_ledger_matches_the_schema() -> None:
    """The committed ledger must be readable by the Stage 8 protocol."""
    entries = ledger.read()
    assert entries, "no ledger entries yet"
    for e in entries:
        assert {"stage", "run_id", "metric", "selected", "tune_gain",
                "fit_rolling_origin_gain_folds", "frozen"} <= set(e)
