"""Did the match_winner cascade actually help, end to end?

``confirm_elo_refit.py`` measured the new Elo with Stage 4, Stage 7 and the
calibration map all still fitted against the OLD Elo, and the Elo-layer gain
did not survive. That was a LOWER bound by construction. This measures the
same thing after all three have been refitted.

Incumbent = the pre-cascade ``fitted_params.json`` and its calibration maps,
both recovered from git. Cascade = whatever is on disk now. Both are scored on
the same TUNE panel, uncalibrated and calibrated, per family.

Nothing is fitted here. TEST and HOLDOUT are not touched.

Run:  python -m scripts.confirm_cascade --incumbent-params PATH --incumbent-maps PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from model import venue as V
from scripts import panel as P
from scripts.fit_stage8 import collect

REPORT = C.REPORTS_DIR / "cascade_confirmation.md"

#: The panel is sampled so a run costs minutes rather than an hour. Both arms
#: draw the SAME rows: same seed, same size, and the sample is taken before
#: either parameter set is applied.
EVAL_SAMPLE = 6000


def _score(fitted: dict, maps_path: Path, pan: pd.DataFrame) -> dict:
    """Per-family metrics under one parameter set, calibrated and not."""
    s7 = fitted["stage_7"]
    cp = CR.CorrectionParams(
        tiebreak_inflation=s7["tiebreak_inflation"], level_sigma=s7["level_sigma"],
        split_sigma=s7["split_sigma"], recenter=s7["recenter"],
        scheme=s7["provenance_scheme"])
    samples = collect(pan, cp)
    maps = RC.CalibrationMaps.load(maps_path)

    out = {}
    for fam, (p, y) in samples.items():
        row = {"n": len(p), "raw_ece": RC.calibration_error(p, y),
               "raw_brier": RC.brier(p, y), "raw_logloss": RC.log_loss(p, y)}
        cal = maps.apply(fam, p)
        row |= {"cal_ece": RC.calibration_error(cal, y),
                "cal_brier": RC.brier(cal, y), "cal_logloss": RC.log_loss(cal, y)}
        out[fam] = row
    return out


def _build_panel(fitted: dict) -> pd.DataFrame:
    """TUNE panel under one parameter set.

    The panel depends on Stage 3 and Stage 4 through (pa, pb), so it has to be
    rebuilt per arm; ``panel.build`` reads fitted_params.json, hence the swap.
    """
    original = C.FITTED_PARAMS_PATH.read_text()
    C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
    try:
        s6 = fitted["stage_6"]
        vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                           enabled=s6["enabled"])
        pan = P.build(splits=("tune",), venue_params=vp)
    finally:
        C.FITTED_PARAMS_PATH.write_text(original)
    return pan.sample(min(len(pan), EVAL_SAMPLE), random_state=C.MC_SEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent-params", required=True, type=Path)
    ap.add_argument("--incumbent-maps", required=True, type=Path)
    a = ap.parse_args()

    incumbent = json.loads(a.incumbent_params.read_text())
    cascade = json.loads(C.FITTED_PARAMS_PATH.read_text())

    print("scoring incumbent ...")
    before = _score(incumbent, a.incumbent_maps, _build_panel(incumbent))
    print("scoring cascade ...")
    after = _score(cascade, RC.MAPS_PATH, _build_panel(cascade))

    fams = sorted(set(before) & set(after))
    lines = [
        "# Did the match_winner cascade help, end to end?\n",
        "`confirm_elo_refit.md` measured the new Elo with Stage 4, Stage 7 and "
        "the calibration map still fitted against the OLD Elo, and the "
        "Elo-layer gain did not survive. That was a lower bound by "
        "construction. This is the same measurement after all three were "
        "refitted.\n",
        "\nIncumbent: Stage 3 half-life 1095 / seed scale 0, and every "
        "downstream stage as shipped before the cascade. Cascade: Stage 3 "
        "half-life 540 / seed scale 160, Stage 4 and Stage 7 refit against it, "
        "calibration map refit on top of those.\n",
        f"\nTUNE, {EVAL_SAMPLE:,} sampled matches, same rows in both arms. "
        "Nothing fitted here; TEST and HOLDOUT untouched.\n",
    ]
    for tag, key in (("Uncalibrated", "raw"), ("Calibrated", "cal")):
        lines += [
            f"\n## {tag}\n",
            "\n| family | n | ECE before | ECE after | Brier before | "
            "Brier after | log-loss before | log-loss after |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for f in fams:
            b, c = before[f], after[f]
            lines.append(
                f"| {f} | {b['n']:,} | {b[key + '_ece']:.5f} | {c[key + '_ece']:.5f} "
                f"| {b[key + '_brier']:.5f} | {c[key + '_brier']:.5f} "
                f"| {b[key + '_logloss']:.5f} | {c[key + '_logloss']:.5f} |")

    mw_b, mw_a = before["match_winner"], after["match_winner"]
    lines += [
        "\n\n## match_winner, the family this was for\n",
        f"\nCalibrated ECE {mw_b['cal_ece']:.5f} -> {mw_a['cal_ece']:.5f} "
        f"({mw_b['cal_ece'] - mw_a['cal_ece']:+.5f}), Brier "
        f"{mw_b['cal_brier']:.5f} -> {mw_a['cal_brier']:.5f} "
        f"({mw_b['cal_brier'] - mw_a['cal_brier']:+.5f}), log-loss "
        f"{mw_b['cal_logloss']:.5f} -> {mw_a['cal_logloss']:.5f} "
        f"({mw_b['cal_logloss'] - mw_a['cal_logloss']:+.5f}).\n",
        "\n## Caveat\n",
        "TUNE-selected and TUNE-confirmed. Stage 4, Stage 7 and the "
        "calibration map were all selected on this same window, so these "
        "numbers are optimistic for the cascade arm and not for the incumbent "
        "one. No untouched window remains to settle it: TEST went to the "
        "calibration refit and HOLDOUT to the 2026 backtest.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
