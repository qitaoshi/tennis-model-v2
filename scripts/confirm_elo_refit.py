"""Does the Elo-layer win survive the full pipeline?

``scripts/fit_unknown_players.py`` selected half-life 540 and rank seed scale
160 at the Elo layer. That is a screen, not a verdict: Stage 4 blends the Elo
probability with a serve-rate probability using weights fitted against the OLD
Elo, and Stage 7's corrections were tuned against the old blend. A gain at the
Elo layer can be diluted, or undone, by either.

This runs the frozen pipeline over TUNE twice — once with the incumbent Stage
3, once with the selected one — and compares match_winner end to end. Nothing
is fitted here; it is a confirmation.

Probabilities are compared UNCALIBRATED. The calibration map was fitted
against the incumbent Elo, so applying it to the new one would measure the
stale map as much as the change.

TEST and HOLDOUT are not touched.

Run:  python -m scripts.confirm_elo_refit
"""

from __future__ import annotations

import json

import numpy as np

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from model import venue as V
from scripts import panel as P
from scripts.fit_stage8 import collect

REPORT = C.REPORTS_DIR / "elo_refit_confirmation.md"


def measure(tag: str, s3_override: dict) -> dict:
    """Build the TUNE panel under a Stage 3 setting and score match_winner."""
    original = json.loads(C.FITTED_PARAMS_PATH.read_text())
    patched = json.loads(json.dumps(original))
    patched["stage_3"].update(s3_override)
    # try/finally: fitted_params.json is the shared source of truth for every
    # other script, so an exception here must not leave it holding a setting
    # nobody chose.
    C.FITTED_PARAMS_PATH.write_text(json.dumps(patched, indent=1))
    try:
        s6 = patched["stage_6"]
        vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                           enabled=s6["enabled"])
        s7 = patched["stage_7"]
        cp = CR.CorrectionParams(
            tiebreak_inflation=s7["tiebreak_inflation"],
            level_sigma=s7["level_sigma"], split_sigma=s7["split_sigma"],
            recenter=s7["recenter"], scheme=s7["provenance_scheme"])
        pan = P.build(splits=("tune",), venue_params=vp)
        pan = pan.sample(min(len(pan), 6_000), random_state=C.MC_SEED)
        samples = collect(pan, cp)
    finally:
        C.FITTED_PARAMS_PATH.write_text(json.dumps(original, indent=1))

    p, y = samples["match_winner"]
    return {"tag": tag, "n": len(p), "ece": RC.calibration_error(p, y),
            "brier": RC.brier(p, y), "logloss": RC.log_loss(p, y)}


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s3 = fitted["stage_3"]
    print(f"incumbent Stage 3: half-life {s3['inactivity_half_life']}, "
          f"seed scale {s3.get('rank_seed_scale', 0.0)}")

    before = measure("incumbent", {"inactivity_half_life": s3["inactivity_half_life"],
                                   "rank_seed_scale": 0.0})
    print(f"  incumbent: ECE {before['ece']:.5f} Brier {before['brier']:.5f} "
          f"log-loss {before['logloss']:.5f} (n={before['n']:,})")

    after = measure("selected", {"inactivity_half_life": 540.0,
                                 "rank_seed_scale": 160.0})
    print(f"  selected:  ECE {after['ece']:.5f} Brier {after['brier']:.5f} "
          f"log-loss {after['logloss']:.5f}")

    d_ece = before["ece"] - after["ece"]
    d_ll = before["logloss"] - after["logloss"]
    d_brier = before["brier"] - after["brier"]
    survives = bool(d_ece > 0 and d_ll >= -1e-4)
    print(f"\ndelta: ECE {d_ece:+.5f}, log-loss {d_ll:+.5f}, Brier {d_brier:+.5f}")
    print(f"\nEnd-to-end gain survives: {survives}")

    lines = [
        "# Does the Elo refit survive the full pipeline?\n",
        "`scripts/fit_unknown_players.py` selected inactivity half-life 540 "
        "and rank seed scale 160 at the Elo layer. Stage 4's blend weights "
        "were fitted against the OLD Elo and Stage 7's corrections against the "
        "old blend, so an Elo-layer gain can be diluted or undone downstream. "
        "This runs the frozen pipeline over TUNE both ways.\n",
        "\nProbabilities are compared **uncalibrated** — the calibration map "
        "was fitted against the incumbent Elo, so applying it would measure "
        "the stale map as much as the change. Nothing is fitted here. TEST "
        "and HOLDOUT are not touched.\n",
        "\n## match_winner on TUNE, through the full pipeline\n",
        f"\n| setting | n | ECE | Brier | log-loss |\n|---|---:|---:|---:|---:|\n"
        f"| incumbent (hl 1095, seed 0) | {before['n']:,} | {before['ece']:.5f} "
        f"| {before['brier']:.5f} | {before['logloss']:.5f} |\n"
        f"| selected (hl 540, seed 160) | {after['n']:,} | {after['ece']:.5f} "
        f"| {after['brier']:.5f} | {after['logloss']:.5f} |\n",
        f"\nDelta: ECE {d_ece:+.5f}, log-loss {d_ll:+.5f}, "
        f"Brier {d_brier:+.5f}.\n",
        f"\n**End-to-end gain survives: {survives}.**\n",
        "\n## If this ships\n",
        "Stage 4's blend weights and Stage 7's corrections were both selected "
        "against the incumbent Elo and would need refitting, and the "
        "calibration map would need refitting after that. This confirmation "
        "measures the change with all of them held at their old values, so it "
        "is a LOWER bound on what a full refit would give — and also a "
        "reminder that shipping this is a cascade, not a one-line edit.\n",
        "\n## Caveat\n",
        "No untouched window remains to confirm this on: TEST went to the "
        "calibration refit and HOLDOUT to the 2026 backtest. TUNE-selected, "
        "TUNE-confirmed, and nothing stronger until new data accrues.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
