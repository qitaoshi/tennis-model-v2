"""Stage 3 fitting and validation — surface Elo.

Selects K, the Challenger K multiplier, the surface blend weight, the
inactivity regression half-life and the new-player level gap on TUNE
log-loss; records the FIT-internal rolling-origin gain and its spread to the
ledger; runs the cross-level consistency check and fits a level offset only
if that check demands one.

Gate (MODEL_PROMPT.md): decile calibration tracks observed win rates, the
Brier score beats a rankings-based baseline, and the cross-level check is
reported either way.

Run:  python -m scripts.fit_stage3
"""

from __future__ import annotations

import json
import itertools

import numpy as np
import pandas as pd

from model import constants as C
from model import elo as E
from model import ledger

KS = [16.0, 24.0, 32.0, 48.0]
SURFACE_WEIGHTS = [0.3, 0.5, 0.7]
INACTIVITY = [365.0, 1095.0, None]
CHALL_MULTS = [0.75, 1.0]
LEVEL_GAPS = [0.0, 100.0]

#: Calibration is judged against criteria fixed before the run, not eyeballed.
MAX_DECILE_GAP = 0.05
MEAN_DECILE_GAP = 0.025


def _load() -> pd.DataFrame:
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    m = m[m["in_scope"] & m["split"].isin(("fit", "tune"))]
    assert m["date"].max() < C.TEST_START, "Stage 3 must not see TEST data"
    return m


def _fold_gains(res: pd.DataFrame, base: np.ndarray) -> list[float]:
    """FIT-internal rolling origin: log-loss gain over the ranking baseline."""
    fit = res["split"] == "fit"
    edges = np.quantile(res.loc[fit, "t"], np.linspace(0.5, 1.0, C.ROLLING_ORIGIN_FOLDS + 1))
    gains = []
    for lo, hi in zip(edges, edges[1:]):
        m = fit & (res["t"] >= lo) & (res["t"] < hi)
        gains.append(E.log_loss(base[m.to_numpy()])
                     - E.log_loss(res.loc[m, "p_winner"].to_numpy()))
    return gains


def main() -> None:
    matches = _load()

    grid = []
    best = None
    for k, w, inact, cm, gap in itertools.product(
        KS, SURFACE_WEIGHTS, INACTIVITY, CHALL_MULTS, LEVEL_GAPS
    ):
        params = E.EloParams(k=k, surface_weight=w, inactivity_half_life=inact,
                             k_chall_mult=cm, level_gap=gap)
        res = E.run_elo(matches, params)
        tune = res[res["split"] == "tune"]
        ll = E.log_loss(tune["p_winner"].to_numpy())
        grid.append({"k": k, "surface_weight": w,
                     "inactivity_half_life": inact, "k_chall_mult": cm,
                     "level_gap": gap, "tune_logloss": ll,
                     "tune_brier": E.brier(tune["p_winner"].to_numpy())})
        if best is None or ll < best[0]:
            best = (ll, params, res)
        print(f"  k={k:>4} w={w} inact={inact} cm={cm} gap={gap} -> {ll:.5f}")

    grid = pd.DataFrame(grid).sort_values("tune_logloss").reset_index(drop=True)
    _, sel, res = best
    print(f"\nselected: {sel}")

    # --- baseline ---------------------------------------------------------
    fit_mask = (res["split"] == "fit").to_numpy()
    base = E.ranking_baseline(res, fit_mask)
    tune_mask = (res["split"] == "tune").to_numpy()

    model_ll = E.log_loss(res.loc[tune_mask, "p_winner"].to_numpy())
    model_brier = E.brier(res.loc[tune_mask, "p_winner"].to_numpy())
    base_ll = E.log_loss(base[tune_mask])
    base_brier = E.brier(base[tune_mask])

    calib = E.decile_calibration(res.loc[tune_mask, "p_winner"].to_numpy())
    cross = E.cross_level_check(res[tune_mask])

    # --- cross-level offset, fitted only if the check demands it ----------
    offset_note = "not needed"
    if cross["n"] > 0:
        se = float(np.sqrt(0.25 / cross["n"]))
        if abs(cross["gap"]) > 2 * se:
            # convert the probability gap into a rating offset at the margin
            offset = -400.0 / np.log(10) * np.log(
                (1 / max(cross["predicted_chall_winrate"] + cross["gap"], 1e-6) - 1)
                / (1 / max(cross["predicted_chall_winrate"], 1e-6) - 1)
            )
            sel = E.EloParams(k=sel.k, k_chall_mult=sel.k_chall_mult,
                              surface_weight=sel.surface_weight,
                              inactivity_half_life=sel.inactivity_half_life,
                              level_gap=sel.level_gap, level_offset=float(offset))
            res = E.run_elo(matches, sel)
            tune_mask = (res["split"] == "tune").to_numpy()
            base = E.ranking_baseline(res, (res["split"] == "fit").to_numpy())
            model_ll = E.log_loss(res.loc[tune_mask, "p_winner"].to_numpy())
            model_brier = E.brier(res.loc[tune_mask, "p_winner"].to_numpy())
            calib = E.decile_calibration(res.loc[tune_mask, "p_winner"].to_numpy())
            cross_after = E.cross_level_check(res[tune_mask])
            offset_note = (f"fitted level_offset {offset:+.1f} rating points "
                           f"(gap {cross['gap']:+.4f} was {abs(cross['gap']) / se:.1f} "
                           f"SE); gap after {cross_after['gap']:+.4f}")
            cross = cross_after
        else:
            offset_note = (f"gap {cross['gap']:+.4f} is "
                           f"{abs(cross['gap']) / se:.1f} SE — within noise, "
                           "no offset fitted")

    # --- ledger (before the gate, per ground rule 2) ----------------------
    folds = _fold_gains(res, base)
    tune_gain = base_ll - model_ll
    selected = {"k": sel.k, "k_chall_mult": sel.k_chall_mult,
                "surface_weight": sel.surface_weight,
                "inactivity_half_life": sel.inactivity_half_life,
                "level_gap": sel.level_gap, "level_offset": sel.level_offset}
    ledger.append(
        stage="stage_3", metric="match_winner_logloss", selected=selected,
        tune_gain=tune_gain, fit_rolling_origin_gains=folds,
        baseline="ranking_logistic", tune_metric_value=model_ll,
        baselines={"ranking_logloss": base_ll, "ranking_brier": base_brier},
        frozen=False, notes=f"grid of {len(grid)} settings; {offset_note}",
        tune_brier=model_brier,
    )

    # --- gate -------------------------------------------------------------
    max_gap = float(calib["gap"].abs().max())
    mean_gap = float(calib["gap"].abs().mean())
    calib_ok = max_gap < MAX_DECILE_GAP and mean_gap < MEAN_DECILE_GAP
    brier_ok = model_brier < base_brier
    passed = calib_ok and brier_ok

    lines = [
        "# Stage 3 validation — surface Elo\n",
        f"Selected on TUNE ({C.TUNE_START} .. {C.TUNE_END}): K **{sel.k:.0f}**, "
        f"Challenger K x**{sel.k_chall_mult}**, surface blend weight "
        f"**{sel.surface_weight}**, inactivity half-life "
        f"**{sel.inactivity_half_life}**, new-player level gap "
        f"**{sel.level_gap:.0f}**, cross-level offset **{sel.level_offset:+.1f}**.\n",
        "\n## TUNE performance\n",
        "| model | log-loss | Brier |",
        "|---|---|---|",
        f"| **surface Elo** | **{model_ll:.5f}** | **{model_brier:.5f}** |",
        f"| rankings baseline (logistic in log-rank difference) | {base_ll:.5f} "
        f"| {base_brier:.5f} |",
        f"\nn = {int(tune_mask.sum()):,} matches.\n",
        "\n## Decile calibration (TUNE)\n",
        "Each match contributes both orientations, so a bin's observed rate is "
        "a real win rate rather than 1 by construction.\n",
        calib.to_markdown(index=False),
        f"\n\nMax |gap| {max_gap:.4f} (criterion < {MAX_DECILE_GAP}), "
        f"mean |gap| {mean_gap:.4f} (criterion < {MEAN_DECILE_GAP}).\n",
        "\n## Cross-level consistency check\n",
        "Matches where one player's rating is >=80% Challenger-built and the "
        "opponent's is not, both with at least 10 rated matches.\n",
        f"\n- n = {cross.get('n', 0):,}",
        f"\n- predicted Challenger-side win rate {cross.get('predicted_chall_winrate', float('nan')):.4f}",
        f"\n- observed {cross.get('observed_chall_winrate', float('nan')):.4f}",
        f"\n- gap {cross.get('gap', float('nan')):+.4f}",
        f"\n- {offset_note}\n",
        "\n## FIT-internal rolling origin\n",
        "Log-loss gain over the ranking baseline, per fold: "
        + ", ".join(f"{g:+.5f}" for g in folds),
        f"\n\nmean {np.mean(folds):+.5f}, std {np.std(folds, ddof=1):.5f}; "
        f"TUNE gain {tune_gain:+.5f}, envelope "
        f"{C.OVERFIT_SIGNAL_MULTIPLE * np.std(folds, ddof=1):.5f}.\n",
        "\n## Grid (top 10 by TUNE log-loss)\n",
        grid.head(10).to_markdown(index=False),
        f"\n\n## Gate\n\nCalibration tracks observed win rates: **{calib_ok}**. "
        f"Brier beats the rankings baseline: **{brier_ok}**. "
        f"Gate **{'PASSED' if passed else 'FAILED'}**.\n",
    ]
    out = C.STAGE_VALIDATIONS_DIR / "stage_3.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))

    if passed:
        ledger.append(
            stage="stage_3", metric="match_winner_logloss", selected=selected,
            tune_gain=tune_gain, fit_rolling_origin_gains=folds,
            baseline="ranking_logistic", tune_metric_value=model_ll,
            baselines={"ranking_logloss": base_ll, "ranking_brier": base_brier},
            frozen=True, notes="gate passed; frozen for downstream stages",
            tune_brier=model_brier,
        )
        fitted = json.loads(C.FITTED_PARAMS_PATH.read_text()) \
            if C.FITTED_PARAMS_PATH.exists() else {}
        fitted["stage_3"] = {**selected, "selected_on": "tune",
                             "tune_logloss": model_ll, "tune_brier": model_brier,
                             "cross_level": cross}
        C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
