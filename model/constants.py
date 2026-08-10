"""Project-wide constants: data split boundaries, paths, seeds, thresholds.

NOTHING IN THIS MODULE IS FITTED. Every value here is either a boundary
decision (the four-way split), a path, a seed, or a threshold derived from
theory / spec text. Fitted quantities live in ``fitted_params.json`` per
ground rule 4 of MODEL_PROMPT.md.

RE-SPLIT of 2026-08-08 — read this before trusting any pre-2026 report
---------------------------------------------------------------------
The original build ran to completion: every stage gate passed, and the
HOLDOUT backtest ran once on 2024-01-01 .. 2026-07-20. That spent the
original holdout. The project goal then changed from "find mispriced betting
lines" to "produce accurate match projections", and the accuracy work that
follows (recalibration, fatigue features, recency weighting) needs an
evaluation window that has not been used for selection.

No unused window existed: the vendor data ends 2026-07-20 and the backtest
had read all of it. The human chose, explicitly, to re-split rather than wait
for new seasons to accrue. **The original HOLDOUT_CUTOFF of 2024-01-01 was
moved on 2026-08-08.** The module previously said that date was permanently
fixed; it was, for the original build, and moving it is the cost of the
re-split, paid knowingly. See ``PRIOR_EVALUATION_WINDOWS`` below.

What that costs: the 2024-01-01 .. 2026-07-20 backtest is no longer an
unbiased out-of-sample estimate of anything fitted after 2026-08-08. Its
recorded numbers stand as a description of the frozen original model on data
that model had never seen — that much is still true — but they must not be
quoted as validation of any later change.

The four-way split (ground rule 1), as of 2026-08-08
----------------------------------------------------
The match data is TML-Database season files (``vendor/``), one per calendar
year per tour (ATP main tour and Challenger), 2010 onward.

    FIT      2010-01-01 .. 2023-12-31   parameter fitting only
    TUNE     2024-01-01 .. 2025-06-30   hyperparameter selection
    TEST     2025-07-01 .. 2025-12-31   pre-cutoff test, touched once
    HOLDOUT  2026-01-01 ..              final evaluation only

    burned   2022-07-01 .. 2023-06-30   spent by the original Stage 8's
                                        first run; now inside FIT

FIT absorbs the original FIT, TUNE, burned window and TEST. Data spent as a
*test* set is legitimate *fitting* data once the boundary has moved past it —
what it can never again be is a test set, which is why the burned window
keeps its own ``split_of`` label instead of silently becoming "fit".

CONTAMINATION NOTE, stated plainly: the new TUNE, TEST and HOLDOUT windows
were all read once by the original backtest, at the aggregate level only
(family Brier / ECE / CRPS, ablation totals). No parameter was ever fitted or
selected against them, and no per-match result from them informed a modelling
decision. That is far weaker contamination than a selection set, but it is
not zero, and any 2026 result should be reported with this caveat attached
rather than as a pristine holdout.

The pre-2026 windows keep a full surface cycle each; HOLDOUT 2026 covers
January through July only, so it is hard/clay/grass with no fall hard-court
season. Sample size on the new HOLDOUT is ~6,200 raw matches versus the
original backtest's 24,572 — smaller, and seasonally incomplete. Both are
limitations of the re-split, not of the model.

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

Split = Literal["pre_data", "fit", "tune", "burned_test", "test", "reserve",
                "holdout"]

DATA_START: date = date(2010, 1, 1)
FIT_END: date = date(2023, 12, 31)
TUNE_START: date = date(2024, 1, 1)
TUNE_END: date = date(2025, 6, 30)

#: TEST windows already spent. The original Stage 8's first run consumed
#: 2022-07-01 .. 2023-06-30. Both that window and the original TEST
#: (2023-07-01 .. 2023-12-31) now sit inside FIT; the burned one keeps its own
#: label so it stays visible that a hyperparameter was once selected on it.
BURNED_TEST_WINDOWS: tuple[tuple[date, date], ...] = (
    (date(2022, 7, 1), date(2023, 6, 30)),
)

#: Windows already read by an evaluation run, with the run that read them.
#: Documentation only — ``split_of`` does not consult this. It exists so the
#: contamination described in the module docstring cannot be forgotten: every
#: window listed here has been observed at least once, so a result measured on
#: it carries a caveat even when no parameter was fitted against it.
PRIOR_EVALUATION_WINDOWS: tuple[tuple[date, date, str], ...] = (
    (date(2022, 7, 1), date(2023, 6, 30),
     "original Stage 8, first run — FAILED, window burned for selection"),
    (date(2023, 7, 1), date(2023, 12, 31),
     "original Stage 8, replacement run — isotonic maps selected here"),
    (date(2024, 1, 1), date(2026, 7, 20),
     "original HOLDOUT backtest, 2026-07-31, aggregate metrics only; "
     "now re-split into TUNE + TEST + HOLDOUT"),
)

TEST_START: date = date(2025, 7, 1)
TEST_END: date = date(2025, 12, 31)
#: No reserve remains: the window is empty by construction, so ``split_of``
#: can never return "reserve". A gate failure cannot be answered with another
#: replacement window inside the pre-holdout data — that is a finding to
#: report, never a reason to move the holdout cutoff again.
RESERVE_START: date = date(2026, 1, 1)
RESERVE_END: date = date(2026, 1, 1)

#: Everything on or after this date is HOLDOUT. Moved once, on 2026-08-08,
#: from 2024-01-01, as part of the documented re-split — see the module
#: docstring. Fixed again from that date: it does not move for a failing gate.
HOLDOUT_CUTOFF: date = date(2026, 1, 1)

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
    """Return which split a match date belongs to.

    A burned TEST window keeps its own label rather than falling back into
    TUNE. It is legitimate FITTING data once the boundary has moved past it —
    it was only ever spent as a *test* set — but silently merging it into TUNE
    would hide that a hyperparameter had been selected on it.
    """
    if d >= HOLDOUT_CUTOFF:
        return "holdout"
    if d >= RESERVE_START:
        return "reserve"
    if d >= TEST_START:
        return "test"
    for lo, hi in BURNED_TEST_WINDOWS:
        if lo <= d <= hi:
            return "burned_test"
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
        date(2022, 6, 30),
        date(2022, 7, 1),
        date(2023, 6, 30),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2025, 6, 30),
        date(2025, 7, 1),
        date(2026, 1, 1),
    ]:
        print(f"  {d} -> {split_of(d)}")

    print("\nwindows already observed (documentation, not a split rule):")
    for lo, hi, why in PRIOR_EVALUATION_WINDOWS:
        print(f"  {lo} .. {hi}  {why}")

    print("\nassert_no_holdout() on a holdout date:")
    try:
        assert_no_holdout([date(2019, 5, 1), date(2024, 5, 1)])
    except ValueError as exc:
        print(f"  raised as expected: {exc}")


if __name__ == "__main__":
    demo()
