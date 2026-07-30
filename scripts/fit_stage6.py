"""Stage 6 fitting and validation — venue court-speed index.

Selects the shrinkage strength (and whether the indoor flag is part of the
key) on TUNE, then checks the gate the spec states: total-games calibration
improves at venues with above-median history, and does not degrade at
unmeasured venues.

Run:  python -m scripts.fit_stage6
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import constants as C
from model import ledger
from model import venue as V
from scripts import panel as P

SHRINK_GRID = [5_000.0, 20_000.0, 50_000.0, 150_000.0]
INDOOR_GRID = [False, True]


def split_eval(pan: pd.DataFrame, median_n: float) -> dict:
    """Evaluate overall, at well-measured venues, and at unmeasured ones."""
    measured = pan["venue_measured"] & (pan["venue_n_matches"] >= median_n)
    unmeasured = ~pan["venue_measured"]
    return {
        "overall": P.evaluate(pan),
        "well_measured": P.evaluate(pan[measured]) if measured.any() else None,
        "unmeasured": P.evaluate(pan[unmeasured]) if unmeasured.any() else None,
    }


def main() -> None:
    base = P.build(splits=("fit", "tune"), venue_params=None)
    tune_base = base[base["split"] == "tune"]
    print(f"baseline (no venue): {len(tune_base):,} TUNE matches")

    # median venue history is defined once, from the baseline panel, so the
    # "above-median history" bucket does not move between candidates
    probe = V.build_asof_index(
        pd.read_parquet(C.PROCESSED_DIR / "matches.parquet").pipe(
            lambda d: d[d["in_scope"] & d["split"].isin(("fit", "tune"))]),
        V.VenueParams())
    median_n = float(probe.loc[probe["split"] == "tune", "venue_n_matches"].median())
    print(f"median venue history at TUNE matches: {median_n:.0f} matches")

    base_panel_with_flags = P.build(splits=("fit", "tune"),
                                    venue_params=V.VenueParams(enabled=False))
    base_flags = V.build_asof_index(
        pd.read_parquet(C.PROCESSED_DIR / "matches.parquet").pipe(
            lambda d: d[d["in_scope"] & d["split"].isin(("fit", "tune"))]),
        V.VenueParams()).set_index("match_id")
    for col in ("measured", "venue_n_matches"):
        base.loc[:, "venue_" + col if col == "measured" else col] = (
            base_flags.loc[base["match_id"], col].to_numpy()
        )
    base_tune = base[base["split"] == "tune"]
    base_eval = split_eval(base_tune, median_n)
    print(f"  baseline CRPS overall {base_eval['overall']['crps']:.4f}, "
          f"well-measured {base_eval['well_measured']['crps']:.4f}, "
          f"unmeasured {base_eval['unmeasured']['crps']:.4f}")

    results, best = [], None
    for shrink in SHRINK_GRID:
        for indoor in INDOOR_GRID:
            vp = V.VenueParams(shrink_n0=shrink, use_indoor=indoor)
            pan = P.build(splits=("fit", "tune"), venue_params=vp)
            tune = pan[pan["split"] == "tune"]
            ev = split_eval(tune, median_n)
            row = {"shrink_n0": shrink, "use_indoor": indoor,
                   "crps": ev["overall"]["crps"],
                   "crps_well_measured": ev["well_measured"]["crps"],
                   "crps_unmeasured": ev["unmeasured"]["crps"],
                   "logloss": ev["overall"]["logloss"]}
            results.append(row)
            print(f"  shrink={shrink:>9.0f} indoor={indoor}: "
                  f"CRPS {row['crps']:.4f} (well-measured "
                  f"{row['crps_well_measured']:.4f})")
            if best is None or row["crps_well_measured"] < best[0]["crps_well_measured"]:
                best = (row, vp, pan)

    best_row, sel, best_panel = best
    grid = pd.DataFrame(results).sort_values("crps_well_measured").reset_index(drop=True)

    # --- FIT-internal rolling origin --------------------------------------
    fit_sel = best_panel[(best_panel["split"] == "fit")
                         & best_panel["venue_measured"]
                         & (best_panel["venue_n_matches"] >= median_n)]
    fit_base = base[(base["split"] == "fit") & base["venue_measured"]
                    & (base["venue_n_matches"] >= median_n)]
    edges = np.quantile(fit_sel["t"], np.linspace(0.5, 1.0, C.ROLLING_ORIGIN_FOLDS + 1))
    folds = []
    for lo, hi in zip(edges, edges[1:]):
        a = P.evaluate(fit_sel[(fit_sel["t"] >= lo) & (fit_sel["t"] < hi)])
        b = P.evaluate(fit_base[(fit_base["t"] >= lo) & (fit_base["t"] < hi)])
        folds.append(b["crps"] - a["crps"])

    tune_gain = base_eval["well_measured"]["crps"] - best_row["crps_well_measured"]
    improves = best_row["crps_well_measured"] < base_eval["well_measured"]["crps"]
    # "no degradation at unmeasured venues": those get exactly the surface
    # average, so any difference is numerical, not a modelling effect
    no_degrade = best_row["crps_unmeasured"] <= base_eval["unmeasured"]["crps"] + 1e-6
    passed = improves and no_degrade

    selected = {"shrink_n0": sel.shrink_n0, "use_indoor": sel.use_indoor,
                "enabled": bool(passed)}
    ledger.append(
        stage="stage_6", metric="total_games_crps_well_measured_venues",
        selected=selected, tune_gain=tune_gain, fit_rolling_origin_gains=folds,
        baseline="no_venue_multiplier",
        tune_metric_value=best_row["crps_well_measured"],
        baselines={"no_venue_crps_well_measured": base_eval["well_measured"]["crps"],
                   "no_venue_crps_unmeasured": base_eval["unmeasured"]["crps"],
                   "no_venue_crps_overall": base_eval["overall"]["crps"]},
        frozen=False, notes=f"grid of {len(grid)} settings; median venue history "
                            f"{median_n:.0f} matches",
    )

    lines = [
        "# Stage 6 validation — venue court-speed index\n",
        "Per-(venue, surface) serve-dominance multiplier relative to the "
        "same-surface tour average, applied to the LEVEL of (pa + pb) and never "
        "to the split. Keyed on the stable `tourney_code`, so sponsor and city "
        "renames do not split a venue's history and a surface change does not "
        "merge two different courts.\n",
        f"\nSelected on TUNE: shrinkage **{sel.shrink_n0:,.0f}** serve points, "
        f"indoor flag in the key **{sel.use_indoor}**.\n",
        "\n## TUNE total-games CRPS\n",
        "| venue group | no venue index | with venue index |",
        "|---|---|---|",
        f"| above-median history (>= {median_n:.0f} matches) | "
        f"{base_eval['well_measured']['crps']:.4f} | "
        f"**{best_row['crps_well_measured']:.4f}** |",
        f"| unmeasured (first edition) | {base_eval['unmeasured']['crps']:.4f} | "
        f"{best_row['crps_unmeasured']:.4f} |",
        f"| all TUNE matches | {base_eval['overall']['crps']:.4f} | "
        f"{best_row['crps']:.4f} |",
        f"\nMatch-winner log-loss is unchanged by construction "
        f"({base_eval['overall']['logloss']:.5f} -> {best_row['logloss']:.5f}); "
        "the multiplier moves the level, and the level barely moves who wins.\n",
        "\n## Provenance\n",
        f"{100 * best_panel['venue_measured'].mean():.1f}% of matches are at a "
        "venue with prior same-surface history. The rest receive exactly the "
        "surface average and are flagged `measured=False`, which price.py "
        "carries into its metadata (ground rule 7).\n",
        "\n## Grid\n",
        grid.to_markdown(index=False),
        "\n\n## FIT-internal rolling origin (well-measured venues)\n",
        "CRPS gain over the no-venue baseline, per fold: "
        + ", ".join(f"{g:+.5f}" for g in folds),
        f"\n\nmean {np.mean(folds):+.5f}, std {np.std(folds, ddof=1):.5f}; "
        f"TUNE gain {tune_gain:+.5f}.\n",
        f"\n## Gate\n\nImproves at venues with above-median history: "
        f"**{improves}**. No degradation at unmeasured venues: "
        f"**{no_degrade}**. Gate **{'PASSED' if passed else 'FAILED'}**.\n",
    ]
    out = C.STAGE_VALIDATIONS_DIR / "stage_6.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]))

    ledger.append(
        stage="stage_6", metric="total_games_crps_well_measured_venues",
        selected=selected, tune_gain=tune_gain, fit_rolling_origin_gains=folds,
        baseline="no_venue_multiplier",
        tune_metric_value=best_row["crps_well_measured"],
        baselines={"no_venue_crps_well_measured": base_eval["well_measured"]["crps"]},
        frozen=True,
        notes="gate passed; frozen" if passed else
              "gate failed; frozen as DISABLED so the lineage records it is unused",
    )
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    fitted["stage_6"] = {**selected, "selected_on": "tune",
                         "median_venue_history": median_n,
                         "tune_crps_well_measured": best_row["crps_well_measured"],
                         "baseline_crps_well_measured": base_eval["well_measured"]["crps"]}
    C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
