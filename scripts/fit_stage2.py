"""Stage 2 fitting and validation.

Selects the half-life, surface-pooling strength and shrinkage n0 on the TUNE
window, records the FIT-internal rolling-origin value and its fold-to-fold
spread for the same metric, writes both to the tune ledger (ground rule 2)
and the selected values to fitted_params.json (ground rule 4), then evaluates
the Stage 2 gate against the two required baselines.

Gate (MODEL_PROMPT.md): the full rate model predicts next-match serve points
won better (lower MAE, and better match-winner log-loss through the Stage 1
engine) than (a) raw career average and (b) last-10-matches average.

Run:  python -m scripts.fit_stage2
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from model import constants as C
from model import ledger
from model import player_rates as PR
from model.engine import p_match
from model.rules import rules_for

HALF_LIVES = [90.0, 180.0, 365.0, 730.0]
SURFACE_POOLS = [0.0, 100.0, 300.0, 1000.0]
SHRINK_N0S = [0.0, 200.0, 500.0, 1500.0]


def _load() -> pd.DataFrame:
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    m = m[m["in_scope"] & m["split"].isin(("fit", "tune"))]
    assert m["date"].max() < C.TEST_START, "Stage 2 must not see TEST data"
    return m


def _eval_window(rates: pd.DataFrame, lo: float, hi: float) -> dict:
    """Serve-points-won metrics over matches with t in [lo, hi)."""
    w = rates[(rates["t"] >= lo) & (rates["t"] < hi)]
    pred_w = np.array([PR.expected_spw(r, o, lg) for r, o, lg
                       in zip(w["w_rate"], w["l_ret"], w["league"])])
    pred_l = np.array([PR.expected_spw(r, o, lg) for r, o, lg
                       in zip(w["l_rate"], w["w_ret"], w["league"])])
    actual = np.concatenate([w["w_spw"].to_numpy(), w["l_spw"].to_numpy()])
    pred = np.concatenate([pred_w, pred_l])
    pts = np.concatenate([w["w_svpt"].to_numpy(), w["l_svpt"].to_numpy()])
    return PR.spw_metrics(actual, pred, pts)


def _baseline_window(base: pd.DataFrame, lo: float, hi: float) -> dict:
    b = base[(base["t"] >= lo) & (base["t"] < hi)]
    actual = np.concatenate([b["w_spw"].to_numpy(), b["l_spw"].to_numpy()])
    pred = np.concatenate([b["w_pred"].to_numpy(), b["l_pred"].to_numpy()])
    pts = np.concatenate([b["w_svpt"].to_numpy(), b["l_svpt"].to_numpy()])
    return PR.spw_metrics(actual, np.nan_to_num(pred, nan=0.62), pts)


def _fit_fold_gains(rates: pd.DataFrame, base: pd.DataFrame) -> list[float]:
    """FIT-internal rolling origin, as a GAIN over the career-average baseline.

    Raw metric levels are not comparable between FIT and TUNE — different
    periods, different fields — so the ledger records gains on both sides and
    the overfit signal compares like with like (ground rule 2).

    Rates are as-of by construction, so evaluating on a later FIT window is
    already an out-of-sample forecast from everything before it.
    """
    fit = rates[rates["t"] < pd.Timestamp(C.TUNE_START).toordinal()]
    edges = np.quantile(fit["t"], np.linspace(0.5, 1.0, C.ROLLING_ORIGIN_FOLDS + 1))
    return [
        _baseline_window(base, lo, hi)["logloss"] - _eval_window(rates, lo, hi)["logloss"]
        for lo, hi in zip(edges, edges[1:])
    ]


@lru_cache(maxsize=200_000)
def _cached_p_match(pa: float, pb: float, best_of: int, final_set: str,
                    final_tb_at: int | None, final_tb_to: int | None) -> float:
    from model.rules import FormatSpec
    spec = FormatSpec(best_of=best_of, final_set=final_set,
                      final_tb_at=final_tb_at, final_tb_to=final_tb_to,
                      provenance="documented", source="cache key")
    return p_match(pa, pb, spec)


def match_winner_logloss(matches: pd.DataFrame, rates: pd.DataFrame,
                         pred_w: np.ndarray, pred_l: np.ndarray) -> float:
    """Match-winner log-loss through the Stage 1 engine.

    (pa, pb) are rounded to 1e-3 before the engine call: the DP is exact but
    not free, and a 0.001 grid is far finer than the rates' own resolution.
    """
    spec_by_key = {}
    fmt = matches.set_index("match_id")[["tourney_code", "season_file"]]
    probs = []
    for mid, pa, pb in zip(rates["match_id"], pred_w, pred_l):
        code, yr = fmt.loc[mid]
        key = (code, int(yr))
        if key not in spec_by_key:
            spec_by_key[key] = rules_for(code, int(yr))
        s = spec_by_key[key]
        if not s.supported:
            continue
        probs.append(_cached_p_match(round(float(pa), 3), round(float(pb), 3),
                                     s.best_of, s.final_set, s.final_tb_at,
                                     s.final_tb_to))
    return PR.match_logloss(np.array(probs))


def main() -> None:
    matches = _load()
    tune_lo = pd.Timestamp(C.TUNE_START).toordinal()
    tune_hi = pd.Timestamp(C.TEST_START).toordinal()

    # --- grid search ------------------------------------------------------
    results = []
    asof_by_hl = {}
    for hl in HALF_LIVES:
        params = PR.RateParams(half_life_days=hl)
        asof_by_hl[hl] = PR.build_asof(matches, params)
        for pool in SURFACE_POOLS:
            for n0 in SHRINK_N0S:
                p = PR.RateParams(half_life_days=hl, surface_pool=pool, shrink_n0=n0)
                r = PR.serve_rates(asof_by_hl[hl], p)
                tune = _eval_window(r, tune_lo, tune_hi)
                results.append({"half_life_days": hl, "surface_pool": pool,
                                "shrink_n0": n0, **tune})
        print(f"  half-life {hl:5.0f}d done")
    grid = pd.DataFrame(results).sort_values("logloss").reset_index(drop=True)
    best = grid.iloc[0]
    sel = PR.RateParams(half_life_days=best["half_life_days"],
                        surface_pool=best["surface_pool"],
                        shrink_n0=best["shrink_n0"])
    print(f"\nselected: {sel}")

    # --- FIT-internal rolling origin for the selected setting -------------
    rates = PR.serve_rates(asof_by_hl[sel.half_life_days], sel)
    career_full = PR.baseline_rates(matches, last_k=None)
    folds = _fit_fold_gains(rates, career_full)

    # --- baselines --------------------------------------------------------
    tune_mask = (rates["t"] >= tune_lo) & (rates["t"] < tune_hi)
    tw = rates[tune_mask]
    model_pred_w = np.array([PR.expected_spw(r, o, lg) for r, o, lg
                             in zip(tw["w_rate"], tw["l_ret"], tw["league"])])
    model_pred_l = np.array([PR.expected_spw(r, o, lg) for r, o, lg
                             in zip(tw["l_rate"], tw["w_ret"], tw["league"])])
    actual = np.concatenate([tw["w_spw"], tw["l_spw"]])
    pts = np.concatenate([tw["w_svpt"], tw["l_svpt"]])
    model_m = PR.spw_metrics(actual, np.concatenate([model_pred_w, model_pred_l]), pts)

    base_metrics, base_pred = {}, {}
    for name, k in (("career_average", None), ("last_10", 10)):
        b = PR.baseline_rates(matches, last_k=k)
        b = b[(b["t"] >= tune_lo) & (b["t"] < tune_hi)]
        b = b.set_index("match_id").loc[tw["match_id"]]
        pw, pl = b["w_pred"].to_numpy(), b["l_pred"].to_numpy()
        base_pred[name] = (np.nan_to_num(pw, nan=0.62), np.nan_to_num(pl, nan=0.62))
        base_metrics[name] = PR.spw_metrics(actual, np.concatenate([pw, pl]), pts)

    # --- match-winner log-loss through the engine -------------------------
    print("\nengine log-loss (model + baselines) ...")
    ll = {"model": match_winner_logloss(matches, tw, model_pred_w, model_pred_l)}
    for name, (pw, pl) in base_pred.items():
        ll[name] = match_winner_logloss(matches, tw, pw, pl)

    # --- ledger (before the gate is evaluated, per ground rule 2) ---------
    baselines = {f"{n}_mae": m["mae"] for n, m in base_metrics.items()}
    baselines |= {f"{n}_logloss": m["logloss"] for n, m in base_metrics.items()}
    baselines |= {f"{n}_match_logloss": v for n, v in ll.items() if n != "model"}
    tune_gain = base_metrics["career_average"]["logloss"] - model_m["logloss"]
    selected = {"half_life_days": sel.half_life_days,
                "surface_pool": sel.surface_pool,
                "shrink_n0": sel.shrink_n0}
    ledger.append(
        stage="stage_2", metric="spw_per_point_logloss", selected=selected,
        tune_gain=tune_gain, fit_rolling_origin_gains=folds,
        baseline="career_average", tune_metric_value=model_m["logloss"],
        baselines=baselines, frozen=False,
        notes=f"grid over {len(HALF_LIVES)}x{len(SURFACE_POOLS)}x{len(SHRINK_N0S)} "
              f"settings; TUNE MAE {model_m['mae']:.5f}, match-winner log-loss "
              f"{ll['model']:.5f}",
        tune_mae=model_m["mae"], tune_match_logloss=ll["model"],
    )

    # --- gate -------------------------------------------------------------
    mae_ok = all(model_m["mae"] < m["mae"] for m in base_metrics.values())
    ll_ok = all(ll["model"] < v for k, v in ll.items() if k != "model")
    passed = mae_ok and ll_ok

    lines = [
        "# Stage 2 validation — player serve rates\n",
        f"Selected on TUNE ({C.TUNE_START} .. {C.TUNE_END}): "
        f"half-life **{sel.half_life_days:.0f}d**, surface pooling "
        f"**{sel.surface_pool:.0f}** serve points, shrinkage n0 "
        f"**{sel.shrink_n0:.0f}** serve points.\n",
        f"\nOpponent adjustment converged inside {C.RATE_ITER_MAX} sweeps at "
        f"tolerance {C.RATE_ITER_TOL:.0e}; the tour-level average serve-win% "
        "per period is the identifiability anchor.\n",
        "\n## Next-match serve points won (TUNE)\n",
        "| model | MAE | per-point log-loss | match-winner log-loss |",
        "|---|---|---|---|",
        f"| **full rate model** | **{model_m['mae']:.5f}** | "
        f"**{model_m['logloss']:.5f}** | **{ll['model']:.5f}** |",
    ]
    for name, m in base_metrics.items():
        lines.append(f"| {name} | {m['mae']:.5f} | {m['logloss']:.5f} | "
                     f"{ll[name]:.5f} |")
    lines += [
        f"\nn = {model_m['n']:,} player-matches.\n",
        "\n## FIT-internal rolling origin (selected setting)\n",
        "Gain over the career-average baseline, per-point log-loss, over "
        f"{len(folds)} sequential FIT windows: "
        + ", ".join(f"{f:+.5f}" for f in folds),
        f"\n\nmean {np.mean(folds):+.5f}, fold-to-fold std "
        f"{np.std(folds, ddof=1):.5f}. TUNE gain {tune_gain:+.5f} "
        f"(envelope at {C.OVERFIT_SIGNAL_MULTIPLE}x std: "
        f"{C.OVERFIT_SIGNAL_MULTIPLE * np.std(folds, ddof=1):.5f}).\n",
        "\n## Grid (top 10 by TUNE per-point log-loss)\n",
        grid.head(10).to_markdown(index=False),
        f"\n\n## Gate\n\nMAE beats both baselines: **{mae_ok}**. "
        f"Match-winner log-loss beats both baselines: **{ll_ok}**. "
        f"Gate **{'PASSED' if passed else 'FAILED'}**.\n",
    ]
    out = C.STAGE_VALIDATIONS_DIR / "stage_2.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))

    if passed:
        ledger.append(
            stage="stage_2", metric="spw_per_point_logloss", selected=selected,
            tune_gain=tune_gain, fit_rolling_origin_gains=folds,
            baseline="career_average", tune_metric_value=model_m["logloss"],
            baselines=baselines, frozen=True,
            notes="gate passed; frozen for downstream stages",
            tune_mae=model_m["mae"], tune_match_logloss=ll["model"],
        )
        params_path = C.FITTED_PARAMS_PATH
        fitted = json.loads(params_path.read_text()) if params_path.exists() else {}
        fitted["stage_2"] = {
            "half_life_days": sel.half_life_days,
            "surface_pool": sel.surface_pool,
            "shrink_n0": sel.shrink_n0,
            "recency": True, "surface_split": True, "opponent_adjust": True,
            "selected_on": "tune", "tune_logloss": model_m["logloss"],
            "tune_mae": model_m["mae"], "tune_match_logloss": ll["model"],
        }
        params_path.write_text(json.dumps(fitted, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
