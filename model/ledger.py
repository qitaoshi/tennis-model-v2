"""Append-only tune ledger (ground rule 2).

Stages 2-7 are each selected against the same TUNE window, using a pipeline
whose earlier stages were also selected against it. No stage looks like it is
overfitting on its own; the composition still can. The four-way split catches
that it happened — this ledger is what makes it *diagnosable*, by recording
for every selection both the TUNE-set value and the FIT-internal
rolling-origin value with its fold-to-fold spread.

``frozen: true`` marks the entry as the version currently feeding price.py.
The Stage 8 failure protocol reads frozen entries only: the live lineage, not
the development history. Superseded entries stay for audit.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np

from model import constants as C


def _run_id() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=C.REPO_ROOT,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        sha = ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{sha}" if sha else stamp


def read() -> list[dict]:
    if not C.TUNE_LEDGER_PATH.exists():
        return []
    return json.loads(C.TUNE_LEDGER_PATH.read_text())


def active_lineage() -> list[dict]:
    """Frozen entries only, latest per (stage, metric) — what price.py runs on."""
    latest: dict[tuple[str, str], dict] = {}
    for e in read():
        if e.get("frozen"):
            latest[(e["stage"], e["metric"])] = e
    return list(latest.values())


def append(stage: str, metric: str, selected: Any, tune_gain: float,
           fit_rolling_origin_gains: list[float], baseline: str,
           tune_metric_value: float | None = None,
           baselines: dict[str, float] | None = None,
           frozen: bool = False, notes: str = "", **extra: Any) -> dict:
    """Append one selection record.

    Everything here is a *gain* over the stage's own baseline, not a raw
    metric level: TUNE and FIT cover different periods, so comparing raw
    levels across them would flag every stage for the calendar rather than
    for overfitting.

    ``fit_rolling_origin_gains`` is the per-fold gain within FIT, measured
    before TUNE is ever consulted. Its fold-to-fold standard deviation is the
    noise envelope that :func:`overfit_flags` compares the TUNE gain against.
    """
    folds = list(map(float, fit_rolling_origin_gains))
    entry = {
        "stage": stage,
        "run_id": _run_id(),
        "metric": metric,
        "baseline": baseline,
        "selected": selected,
        "tune_gain": float(tune_gain),
        "tune_metric_value": None if tune_metric_value is None else float(tune_metric_value),
        "fit_rolling_origin_gain_folds": folds,
        "fit_rolling_origin_gain_mean": float(np.mean(folds)) if folds else None,
        "fit_rolling_origin_gain_std": float(np.std(folds, ddof=1)) if len(folds) > 1 else None,
        "baselines": baselines or {},
        "frozen": bool(frozen),
        "notes": notes,
        **extra,
    }
    entries = read()
    entries.append(entry)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.TUNE_LEDGER_PATH.write_text(json.dumps(entries, indent=1))
    return entry


def overfit_flags() -> list[dict]:
    """Frozen stages whose TUNE gain outran their FIT-internal noise envelope.

    The signature is defined here once, not eyeballed at diagnosis time
    (ground rule 2): the quantity thresholded is the TUNE gain *in excess of*
    the FIT-internal gain, measured in units of the FIT fold-to-fold spread —

        tune_gain - mean(fit_gains) > OVERFIT_SIGNAL_MULTIPLE * std(fit_gains)

    A stage that merely helps consistently (large gain in FIT and a similar
    gain on TUNE) is not implicated; a stage whose TUNE gain outruns what
    FIT-internal evaluation ever showed is. Thresholding the raw TUNE gain
    against the spread instead would flag every stage that works at all, which
    would make the Stage 8 protocol's candidate list useless. Two people
    reading the ledger still reach the same list.
    """
    out = []
    for e in active_lineage():
        std = e.get("fit_rolling_origin_gain_std")
        mean = e.get("fit_rolling_origin_gain_mean")
        if std is None or mean is None or std == 0:
            continue
        excess = e["tune_gain"] - mean
        envelope = C.OVERFIT_SIGNAL_MULTIPLE * std
        if excess > envelope:
            out.append({**e, "excess_over_fit_gain": excess, "envelope": envelope})
    return out


def demo() -> None:
    """Worked example (ground rule 10)."""
    entries = read()
    print(f"{len(entries)} ledger entries, {len(active_lineage())} in the active lineage")
    for e in entries[-6:]:
        std = e.get("fit_rolling_origin_gain_std")
        mean = e.get("fit_rolling_origin_gain_mean")
        print(f"  {e['stage']:8s} {e['metric']:24s} {str(e['selected']):32s} "
              f"tune gain {e['tune_gain']:+.5f}  FIT gain {mean:+.5f} "
              f"+/-{'n/a' if std is None else f'{std:.5f}'}  frozen={e['frozen']}")
    flags = overfit_flags()
    print(f"overfit-signal flags: {[f['stage'] for f in flags] or 'none'}")


if __name__ == "__main__":
    demo()
