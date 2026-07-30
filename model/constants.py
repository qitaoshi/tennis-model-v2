"""Project-wide constants: data split boundaries, paths, seeds, thresholds.

NOTHING IN THIS MODULE IS FITTED. Every value here is either a boundary
decision (the four-way split), a path, a seed, or a threshold derived from
theory / spec text. Fitted quantities live in ``fitted_params.json`` per
ground rule 4 of MODEL_PROMPT.md.

The four-way split (ground rule 1)
----------------------------------
The match data is TML-Database season files (``vendor/``), one per calendar
year per tour (ATP main tour and Challenger), 2010 onward.

    FIT      2010-01-01 .. 2020-12-31   parameter fitting only
    TUNE     2021-01-01 .. 2022-06-30   hyperparameter selection, Stages 2-7
    TEST     2022-07-01 .. 2023-06-30   pre-cutoff test, Stage 8 only, once
    RESERVE  2023-07-01 .. 2023-12-31   untouched; replacement TEST window if
                                        and only if Stage 8 fails and the
                                        TUNE/TEST boundary must move
    HOLDOUT  2024-01-01 ..              final backtest only

``HOLDOUT_CUTOFF`` IS PERMANENTLY FIXED. It never moves, including after a
Stage 8 failure. The only boundary that may move during iteration is the
TUNE/TEST boundary, which is what ``RESERVE`` exists to fund.

The TEST window spans a full 12 months (Jul->Jun) so it contains the whole
surface cycle — fall hard, Australian hard, spring clay, grass — rather than
a single surface season. RESERVE (Jul-Dec) is hard-court-heavy; that is a
known limitation of the contingency window, not of TEST.

A match's split is assigned from ``tourney_date``, the tournament START date,
which is the only date TML records. A tournament straddling a boundary is
therefore assigned whole to the split its first day falls in — deliberate, so
no event is split across two windows.

Data note: ``data/raw/`` holds tennis-data.co.uk season files, which carry
bookmaker odds and no serve statistics. No model module reads them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Literal

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
#: TML-Database match files (vendor/atp_matches_YYYY.csv, chall_matches_YYYY.csv).
#: This is the model's only match-data source.
VENDOR_DIR: Path = REPO_ROOT / "vendor"
#: tennis-data.co.uk season files. Carry bookmaker odds and no serve stats; not
#: read by any model module.
DATA_RAW_DIR: Path = REPO_ROOT / "data" / "raw"
PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed"
WEATHER_CACHE_DIR: Path = REPO_ROOT / "data" / "weather_cache"
REPORTS_DIR: Path = REPO_ROOT / "reports"
STAGE_VALIDATIONS_DIR: Path = REPORTS_DIR / "stage_validations"
TUNE_LEDGER_PATH: Path = REPORTS_DIR / "tune_ledger.json"
RULES_DISCREPANCIES_PATH: Path = REPORTS_DIR / "rules_discrepancies.csv"
FITTED_PARAMS_PATH: Path = REPO_ROOT / "fitted_params.json"

# --------------------------------------------------------------------------
# Four-way split (ground rule 1)
# --------------------------------------------------------------------------

Split = Literal["pre_data", "fit", "tune", "test", "reserve", "holdout"]

DATA_START: date = date(2010, 1, 1)
FIT_END: date = date(2020, 12, 31)
TUNE_START: date = date(2021, 1, 1)
TUNE_END: date = date(2022, 6, 30)
TEST_START: date = date(2022, 7, 1)
TEST_END: date = date(2023, 6, 30)
RESERVE_START: date = date(2023, 7, 1)
RESERVE_END: date = date(2023, 12, 31)

#: PERMANENTLY FIXED. Everything on or after this date is HOLDOUT.
HOLDOUT_CUTOFF: date = date(2024, 1, 1)

# --------------------------------------------------------------------------
# Determinism (ground rule 9)
# --------------------------------------------------------------------------

MC_SEED: int = 20260730
ROLLING_ORIGIN_FOLDS: int = 5  # FIT-internal folds, ground rule 2

# --------------------------------------------------------------------------
# Thresholds fixed by spec, not fitted
# --------------------------------------------------------------------------

#: Ground rule 2: a stage is implicated by the Stage 8 failure protocol when
#: its TUNE-set gain exceeds this multiple of the fold-to-fold std of its
#: FIT-internal rolling-origin gain. Fixed here so the diagnosis is not
#: re-decided per incident.
OVERFIT_SIGNAL_MULTIPLE: float = 2.0

#: rules.py empirical scan: a tournament whose score strings disagree with its
#: assigned format at a rate above this, over at least the minimum match count,
#: is escalated for manual review rather than treated as data-entry noise.
RULES_DISCREPANCY_REVIEW_RATE: float = 0.03
RULES_DISCREPANCY_MIN_MATCHES: int = 30

#: Stage 1: advantage-set total-games distributions are unbounded. Truncate
#: once the remaining tail mass falls below this. This truncation point IS the
#: ladder's upper bound — no line beyond it is priced.
TAIL_MASS_THRESHOLD: float = 1e-9

#: Stage 2 opponent-adjustment iteration (paired comparison). Hard-fail rather
#: than silently accept an unconverged solution.
RATE_ITER_TOL: float = 1e-6
RATE_ITER_MAX: int = 50


# --------------------------------------------------------------------------
# Split helpers
# --------------------------------------------------------------------------


def split_of(d: date) -> Split:
    """Return which split a match date belongs to."""
    if d >= HOLDOUT_CUTOFF:
        return "holdout"
    if d >= RESERVE_START:
        return "reserve"
    if d >= TEST_START:
        return "test"
    if d >= TUNE_START:
        return "tune"
    if d >= DATA_START:
        return "fit"
    return "pre_data"


def assert_no_holdout(dates: Iterable[date]) -> None:
    """Raise if any date is in the HOLDOUT range.

    Call this from every loader outside the final backtest. Advisory rules
    are easy to forget several stages in; this makes the boundary a runtime
    error instead.
    """
    bad = [d for d in dates if d >= HOLDOUT_CUTOFF]
    if bad:
        raise ValueError(
            f"HOLDOUT data touched outside the final backtest: "
            f"{len(bad)} rows on/after {HOLDOUT_CUTOFF} (first {min(bad)})"
        )


def demo() -> None:
    """Worked example (ground rule 10)."""
    print("Four-way split:")
    for label, lo, hi in [
        ("FIT    ", DATA_START, FIT_END),
        ("TUNE   ", TUNE_START, TUNE_END),
        ("TEST   ", TEST_START, TEST_END),
        ("RESERVE", RESERVE_START, RESERVE_END),
        ("HOLDOUT", HOLDOUT_CUTOFF, None),
    ]:
        print(f"  {label}  {lo} .. {hi or 'open (permanently fixed cutoff)'}")

    print("\nsplit_of() spot checks:")
    for d in [
        date(2012, 12, 31),
        date(2020, 12, 31),
        date(2021, 1, 1),
        date(2022, 6, 30),
        date(2022, 7, 1),
        date(2023, 6, 30),
        date(2023, 7, 1),
        date(2024, 1, 1),
    ]:
        print(f"  {d} -> {split_of(d)}")

    print("\nassert_no_holdout() on a holdout date:")
    try:
        assert_no_holdout([date(2019, 5, 1), date(2024, 5, 1)])
    except ValueError as exc:
        print(f"  raised as expected: {exc}")


if __name__ == "__main__":
    demo()
