"""Stage 7 fitting and validation — iid failure-mode corrections.

Two separate failures, two separate corrections, validated separately:

(a) tiebreak frequency — fit the close-set hold inflation that closes the
    measured gap between observed and predicted tiebreak occurrence;
(b) total-games variance — fit the level wobble that makes the games
    distribution's PIT uniform.

The provenance-weighting question is RESOLVED here, not left open: both
corrections are measured under all three schemes in ``corrections.SCHEMES``
and the one with the better TUNE calibration is written to
fitted_params.json.

Gate: post-correction, tiebreak-occurrence calibration and games-total
PIT/coverage are both acceptable, and neither correction degraded the other's.

Run:  python -m scripts.fit_stage7
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import ledger
from model import venue as V
from scripts import panel as P

#: The correction is "sized by the measured gap", so the grid has to be able
#: to take whichever sign the data demands. On this dataset the engine turns
#: out to OVER-predict tiebreaks — real matches produce more breaks than iid
#: points do — so the fitted value is a deflation, not the inflation the spec
#: anticipated.
INFLATIONS = [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01]
SIGMAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]

#: Gate criteria, fixed before the run.
MAX_TB_GAP = 0.01          # 1 percentage point of tiebreak occurrence
COVERAGE_BAND = (0.75, 0.85)   # empirical coverage of the central 80%
NO_DEGRADE_TOL = 0.002

EVAL_SAMPLE = 6000


@lru_cache(maxsize=400_000)
def _corrected(pa: float, pb: float, key: tuple, infl: float, sigma: float):
    from model.rules import FormatSpec
    spec = FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                      tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                      final_tb_to=key[6], provenance="documented", source="s7")
    return CR.corrected_distribution(pa, pb, spec,
                                     CR.CorrectionParams(tiebreak_inflation=infl,
                                                         level_sigma=sigma))


def predict(pan: pd.DataFrame, infl: float, sigma: float) -> dict[str, np.ndarray]:
    """Predicted tiebreak probability and games CDFs for a panel."""
    p_tb, cdf_at, cdf_before = [], [], []
    for pa, pb, key, actual in zip(pan["pa"], pan["pb"], pan["spec_key"],
                                   pan["total_games"]):
        d = _corrected(round(float(pa), 3), round(float(pb), 3), key, infl, sigma)
        p_tb.append(d.tiebreak_any)
        pmf = d.total_games_pmf()
        xs = np.array(sorted(pmf))
        cum = np.cumsum([pmf[x] for x in xs])
        at = float(cum[xs <= actual][-1]) if (xs <= actual).any() else 0.0
        before = float(cum[xs < actual][-1]) if (xs < actual).any() else 0.0
        cdf_at.append(at)
        cdf_before.append(before)
    return {"p_tb": np.array(p_tb), "cdf_at": np.array(cdf_at),
            "cdf_before": np.array(cdf_before)}


def measure(pan: pd.DataFrame, pred: dict, weights: np.ndarray) -> dict:
    obs_tb = (pan["tiebreaks"].to_numpy() > 0).astype(float)
    gap = CR.measure_tiebreak_gap(obs_tb, pred["p_tb"], weights)
    pit = CR.pit_values(pred["cdf_at"], pred["cdf_before"])
    keep = weights > 0
    return {"tb_gap": gap, "pit_dev": CR.pit_deviation(pit[keep]),
            "coverage80": CR.coverage(pit[keep]),
            "obs_tb": float(np.average(obs_tb, weights=weights)),
            "pred_tb": float(np.average(pred["p_tb"], weights=weights)),
            "n": int(keep.sum())}


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s6 = fitted["stage_6"]
    vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                       enabled=s6["enabled"])
    full = P.build(splits=("fit", "tune"), venue_params=vp)
    # Ground rules 6 and 8 already applied by the panel's serve_stats_valid
    # filter; conditioning on the format in force comes from spec_key.
    tune = full[full["split"] == "tune"]
    if len(tune) > EVAL_SAMPLE:
        tune = tune.sample(EVAL_SAMPLE, random_state=C.MC_SEED)
    fit = full[full["split"] == "fit"]
    print(f"TUNE evaluation rows: {len(tune):,}; FIT rows: {len(fit):,}")
    prov = tune["format_provenance"].to_numpy()
    print(f"provenance: {pd.Series(prov).value_counts().to_dict()}")

    # --- (a) tiebreak frequency, per scheme -------------------------------
    tb_runs = {infl: predict(tune, infl, 0.0) for infl in INFLATIONS}
    print("tiebreak sweep done")

    chosen: dict[str, dict] = {}
    for scheme in CR.SCHEMES:
        w = CR.provenance_weights(prov, CR.CorrectionParams(
            scheme=scheme, inferred_weight=0.5))
        if w.sum() == 0:
            continue
        rows = {infl: measure(tune, tb_runs[infl], w) for infl in INFLATIONS}
        best_infl = min(rows, key=lambda i: abs(rows[i]["tb_gap"]))
        chosen[scheme] = {"inflation": best_infl, "tb": rows[best_infl],
                          "tb_uncorrected": rows[0.0], "weights": w,
                          "tb_rows": rows}
        print(f"  {scheme:20s} inflation {best_infl:.4f} "
              f"gap {rows[0.0]['tb_gap']:+.4f} -> {rows[best_infl]['tb_gap']:+.4f}")

    # --- (b) total-games variance, per scheme -----------------------------
    for scheme, info in chosen.items():
        w = info["weights"]
        rows = {}
        for sigma in SIGMAS:
            pred = predict(tune, info["inflation"], sigma)
            rows[sigma] = (measure(tune, pred, w), pred)
        best_sigma = min(rows, key=lambda s: rows[s][0]["pit_dev"])
        info["sigma"] = best_sigma
        info["both"] = rows[best_sigma][0]
        info["sigma_rows"] = {s: r[0] for s, r in rows.items()}
        # variance correction alone, to check it did not need the other
        alone = measure(tune, predict(tune, 0.0, best_sigma), w)
        info["sigma_alone"] = alone
        print(f"  {scheme:20s} sigma {best_sigma:.3f} "
              f"PIT dev {rows[0.0][0]['pit_dev']:.5f} -> "
              f"{rows[best_sigma][0]['pit_dev']:.5f}, "
              f"coverage {rows[best_sigma][0]['coverage80']:.3f}")

    # --- select the provenance scheme by TUNE calibration -----------------
    def score(info: dict) -> float:
        return abs(info["both"]["tb_gap"]) / MAX_TB_GAP + info["both"]["pit_dev"] / 0.01

    sel_scheme = min(chosen, key=lambda s: score(chosen[s]))
    sel = chosen[sel_scheme]
    params = CR.CorrectionParams(tiebreak_inflation=sel["inflation"],
                                 level_sigma=sel["sigma"],
                                 inferred_weight=0.5 if sel_scheme == "downweight_inferred" else 1.0,
                                 scheme=sel_scheme)
    print(f"\nselected scheme: {sel_scheme} -> {params}")

    # --- FIT-internal rolling origin, both corrections --------------------
    fit_sample = fit.sample(min(len(fit), EVAL_SAMPLE), random_state=C.MC_SEED)
    fit_prov = fit_sample["format_provenance"].to_numpy()
    w_fit = CR.provenance_weights(fit_prov, params)
    edges = np.quantile(fit_sample["t"], np.linspace(0.5, 1.0, C.ROLLING_ORIGIN_FOLDS + 1))
    tb_folds, pit_folds = [], []
    for lo, hi in zip(edges, edges[1:]):
        mask = (fit_sample["t"] >= lo) & (fit_sample["t"] < hi)
        sub = fit_sample[mask]
        ws = w_fit[mask.to_numpy()]
        if ws.sum() == 0:
            continue
        before = measure(sub, predict(sub, 0.0, 0.0), ws)
        after = measure(sub, predict(sub, params.tiebreak_inflation,
                                     params.level_sigma), ws)
        tb_folds.append(abs(before["tb_gap"]) - abs(after["tb_gap"]))
        pit_folds.append(before["pit_dev"] - after["pit_dev"])

    before_tune = sel["tb_uncorrected"]
    after_tune = sel["both"]

    ledger.append(
        stage="stage_7", metric="tiebreak_occurrence_gap",
        selected={"tiebreak_inflation": params.tiebreak_inflation,
                  "scheme": sel_scheme},
        tune_gain=abs(before_tune["tb_gap"]) - abs(after_tune["tb_gap"]),
        fit_rolling_origin_gains=tb_folds, baseline="uncorrected_engine",
        tune_metric_value=abs(after_tune["tb_gap"]),
        baselines={"uncorrected_gap": before_tune["tb_gap"]},
        frozen=False, notes="tiebreak-frequency correction")
    ledger.append(
        stage="stage_7", metric="total_games_pit_deviation",
        selected={"level_sigma": params.level_sigma, "scheme": sel_scheme},
        tune_gain=before_tune["pit_dev"] - after_tune["pit_dev"],
        fit_rolling_origin_gains=pit_folds, baseline="uncorrected_engine",
        tune_metric_value=after_tune["pit_dev"],
        baselines={"uncorrected_pit_dev": before_tune["pit_dev"],
                   "uncorrected_coverage80": before_tune["coverage80"]},
        frozen=False, notes="total-games variance correction")

    # --- gate -------------------------------------------------------------
    tb_ok = (abs(after_tune["tb_gap"]) < MAX_TB_GAP
             and abs(after_tune["tb_gap"]) < abs(before_tune["tb_gap"]))
    pit_ok = (after_tune["pit_dev"] < before_tune["pit_dev"]
              and COVERAGE_BAND[0] <= after_tune["coverage80"] <= COVERAGE_BAND[1])
    tb_alone = sel["tb_rows"][params.tiebreak_inflation]
    no_degrade = (
        abs(after_tune["tb_gap"]) <= abs(tb_alone["tb_gap"]) + NO_DEGRADE_TOL
        and after_tune["pit_dev"] <= sel["sigma_alone"]["pit_dev"] + NO_DEGRADE_TOL
    )
    passed = tb_ok and pit_ok and no_degrade

    lines = [
        "# Stage 7 validation — iid failure-mode corrections\n",
        "Measured on TUNE, conditioned on the format rule in force at match "
        "time, excluding retirements, walkovers and `score_string_suspect` "
        "matches.\n",
        f"\nSelected provenance scheme: **{sel_scheme}** "
        f"(inferred weight {params.inferred_weight}). Tiebreak inflation "
        f"**{params.tiebreak_inflation:.4f}**, level sigma "
        f"**{params.level_sigma:.3f}**.\n",
        "\n## (a) Tiebreak occurrence\n",
        "| | observed | predicted | gap |",
        "|---|---|---|---|",
        f"| before | {before_tune['obs_tb']:.4f} | {before_tune['pred_tb']:.4f} | "
        f"{before_tune['tb_gap']:+.4f} |",
        f"| after | {after_tune['obs_tb']:.4f} | {after_tune['pred_tb']:.4f} | "
        f"**{after_tune['tb_gap']:+.4f}** |",
        "\n### Inflation sweep (selected scheme)\n",
        "| inflation | tiebreak gap |",
        "|---|---|",
    ]
    for infl, row in sel["tb_rows"].items():
        lines.append(f"| {infl:.4f} | {row['tb_gap']:+.5f} |")
    lines += [
        "\n## (b) Total games — PIT and coverage\n",
        "| | PIT deviation from uniform | central-80% coverage |",
        "|---|---|---|",
        f"| before | {before_tune['pit_dev']:.5f} | {before_tune['coverage80']:.4f} |",
        f"| after | **{after_tune['pit_dev']:.5f}** | "
        f"**{after_tune['coverage80']:.4f}** |",
        "\n### Sigma sweep (selected scheme, at the chosen inflation)\n",
        "| level sigma | PIT deviation | coverage80 |",
        "|---|---|---|",
    ]
    for sigma, row in sel["sigma_rows"].items():
        lines.append(f"| {sigma:.3f} | {row['pit_dev']:.5f} | {row['coverage80']:.4f} |")
    lines += [
        "\n## Provenance-weighting schemes compared (ground rule 4)\n",
        "| scheme | inflation | sigma | tiebreak gap | PIT deviation |",
        "|---|---|---|---|---|",
    ]
    for scheme, info in chosen.items():
        mark = " **(selected)**" if scheme == sel_scheme else ""
        lines.append(f"| {scheme}{mark} | {info['inflation']:.4f} | "
                     f"{info['sigma']:.3f} | {info['both']['tb_gap']:+.5f} | "
                     f"{info['both']['pit_dev']:.5f} |")
    lines += [
        "\n## Separation check\n",
        "Neither correction may degrade the other's calibration.\n",
        f"\n- tiebreak gap with the tiebreak correction alone: "
        f"{tb_alone['tb_gap']:+.5f}; with both: {after_tune['tb_gap']:+.5f}",
        f"\n- PIT deviation with the variance correction alone: "
        f"{sel['sigma_alone']['pit_dev']:.5f}; with both: "
        f"{after_tune['pit_dev']:.5f}\n",
        "\n## FIT-internal rolling origin\n",
        "Tiebreak-gap improvement per fold: "
        + ", ".join(f"{g:+.5f}" for g in tb_folds),
        f"\n\nPIT-deviation improvement per fold: "
        + ", ".join(f"{g:+.5f}" for g in pit_folds) + "\n",
        f"\n## Gate\n\nTiebreak calibration acceptable (|gap| < {MAX_TB_GAP} and "
        f"improved): **{tb_ok}**. Games PIT/coverage acceptable (improved and "
        f"coverage in {COVERAGE_BAND}): **{pit_ok}**. Neither correction "
        f"degraded the other: **{no_degrade}**. "
        f"Gate **{'PASSED' if passed else 'FAILED'}**.\n",
    ]
    out = C.STAGE_VALIDATIONS_DIR / "stage_7.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]))

    if passed:
        for metric, sel_d, gain, folds, val in (
            ("tiebreak_occurrence_gap",
             {"tiebreak_inflation": params.tiebreak_inflation, "scheme": sel_scheme},
             abs(before_tune["tb_gap"]) - abs(after_tune["tb_gap"]), tb_folds,
             abs(after_tune["tb_gap"])),
            ("total_games_pit_deviation",
             {"level_sigma": params.level_sigma, "scheme": sel_scheme},
             before_tune["pit_dev"] - after_tune["pit_dev"], pit_folds,
             after_tune["pit_dev"]),
        ):
            ledger.append(stage="stage_7", metric=metric, selected=sel_d,
                          tune_gain=gain, fit_rolling_origin_gains=folds,
                          baseline="uncorrected_engine", tune_metric_value=val,
                          frozen=True, notes="gate passed; frozen")
        fitted["stage_7"] = {
            "tiebreak_inflation": params.tiebreak_inflation,
            "level_sigma": params.level_sigma, "n_mix": params.n_mix,
            "provenance_scheme": sel_scheme,
            "inferred_weight": params.inferred_weight,
            "selected_on": "tune",
            "tune_tb_gap_before": before_tune["tb_gap"],
            "tune_tb_gap_after": after_tune["tb_gap"],
            "tune_pit_dev_before": before_tune["pit_dev"],
            "tune_pit_dev_after": after_tune["pit_dev"],
            "tune_coverage80_after": after_tune["coverage80"],
        }
        C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
