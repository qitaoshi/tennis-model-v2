"""Stage 4 fitting and validation — blending serve rates with Elo.

Fits the blend weight w on FIT and selects it on TUNE by match-winner
log-loss, overall and per sample-size bucket (both well sampled / one thin /
both thin), logs the selection with its FIT-internal rolling-origin spread,
and reports the elo-vs-serve disagreement distribution.

Gate (MODEL_PROMPT.md): the fitted w beats w=0 and w=1 on log-loss;
per-bucket w values and the disagreement distribution are reported.

Run:  python -m scripts.fit_stage4
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import combine as CB
from model import constants as C
from model import elo as E
from model import ledger
from model import player_rates as PR
from model.rules import rules_for

W_GRID = np.round(np.arange(0.0, 1.001, 0.05), 3)
THIN_NS = [500.0, 1000.0, 2000.0]


def build_panel() -> pd.DataFrame:
    """One row per match with both opinions, as of that match."""
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    m = m[m["in_scope"] & m["split"].isin(("fit", "tune"))]
    assert m["date"].max() < C.TEST_START, "Stage 4 must not see TEST data"

    s2 = fitted["stage_2"]
    rp = PR.RateParams(half_life_days=s2["half_life_days"],
                       surface_pool=s2["surface_pool"],
                       shrink_n0=s2["shrink_n0"])
    rates = PR.serve_rates(PR.build_asof(m, rp), rp)
    rates["serve_pa"] = [PR.expected_spw(r, o, lg) for r, o, lg
                         in zip(rates["w_rate"], rates["l_ret"], rates["league"])]
    rates["serve_pb"] = [PR.expected_spw(r, o, lg) for r, o, lg
                         in zip(rates["l_rate"], rates["w_ret"], rates["league"])]

    s3 = fitted["stage_3"]
    ep = E.EloParams(k=s3["k"], k_chall_mult=s3["k_chall_mult"],
                     surface_weight=s3["surface_weight"],
                     inactivity_half_life=s3["inactivity_half_life"],
                     level_gap=s3["level_gap"], level_offset=s3["level_offset"])
    elo = E.run_elo(m, ep)[["match_id", "p_winner", "split", "t", "tourney_code",
                            "season_file"]]

    panel = rates.merge(elo, on="match_id", how="inner", suffixes=("", "_elo"))
    panel["elo_p"] = panel["p_winner"]
    panel["n_a"], panel["n_b"] = panel["w_n"], panel["l_n"]
    specs = {(c, int(y)): rules_for(c, int(y))
             for c, y in panel[["tourney_code", "season_file"]].drop_duplicates().values}
    panel["spec_key"] = [CB._spec_key(specs[(c, int(y))]) for c, y
                         in zip(panel["tourney_code"], panel["season_file"])]
    panel = panel[[specs[(c, int(y))].supported for c, y
                   in zip(panel["tourney_code"], panel["season_file"])]]
    panel["level"] = panel["serve_pa"] + panel["serve_pb"]
    panel["serve_gap"] = panel["serve_pa"] - panel["serve_pb"]

    # Elo's opinion expressed as a serve gap, once per match.
    panel["elo_gap"] = [
        CB.elo_gap_for(p, lv, specs[(c, int(y))])
        for p, lv, c, y in zip(panel["elo_p"], panel["level"],
                               panel["tourney_code"], panel["season_file"])
    ]
    panel["serve_p"] = _p_grid(panel["level"].to_numpy(),
                               panel["serve_gap"].to_numpy(),
                               panel["spec_key"].tolist())
    panel["disagreement_pp"] = 100.0 * (panel["elo_p"] - panel["serve_p"])
    return panel


def _p_grid(level: np.ndarray, gap: np.ndarray, keys: list[tuple]) -> np.ndarray:
    out = np.zeros(len(level))
    keys_arr = np.array([hash(k) for k in keys])
    for h in np.unique(keys_arr):
        m = keys_arr == h
        key = keys[int(np.flatnonzero(m)[0])]
        out[m] = CB.p_from_grid(level[m], gap[m], key)
    return out


def logloss_for_w(panel: pd.DataFrame, w: np.ndarray | float) -> float:
    split = (1 - w) * panel["serve_gap"].to_numpy() + w * panel["elo_gap"].to_numpy()
    p = _p_grid(panel["level"].to_numpy(), split, panel["spec_key"].tolist())
    return float(-np.mean(np.log(np.clip(p, 1e-9, 1 - 1e-9))))


def main() -> None:
    panel = build_panel()
    fit = panel[panel["split"] == "fit"]
    tune = panel[panel["split"] == "tune"]
    print(f"panel: {len(panel):,} matches ({len(fit):,} FIT, {len(tune):,} TUNE)")

    # --- overall w: fitted on FIT, selected on TUNE -----------------------
    fit_curve = {w: logloss_for_w(fit, w) for w in W_GRID}
    tune_curve = {w: logloss_for_w(tune, w) for w in W_GRID}
    w_fit = min(fit_curve, key=fit_curve.get)
    w_sel = min(tune_curve, key=tune_curve.get)

    # --- per-bucket w and the thin threshold ------------------------------
    best = None
    for thin_n in THIN_NS:
        b_tune = CB.bucket_of(tune["n_a"].to_numpy(), tune["n_b"].to_numpy(), thin_n)
        b_fit = CB.bucket_of(fit["n_a"].to_numpy(), fit["n_b"].to_numpy(), thin_n)
        ws, per_bucket = {}, {}
        for bucket in (0, 1, 2):
            sub_t = tune[b_tune == bucket]
            sub_f = fit[b_fit == bucket]
            if len(sub_t) < 200:
                ws[bucket] = w_sel
                per_bucket[bucket] = {"n_tune": len(sub_t), "w": w_sel,
                                      "w_fit": float("nan"), "note": "too few matches"}
                continue
            curve = {w: logloss_for_w(sub_t, w) for w in W_GRID}
            fcurve = {w: logloss_for_w(sub_f, w) for w in W_GRID}
            ws[bucket] = min(curve, key=curve.get)
            per_bucket[bucket] = {"n_tune": len(sub_t), "w": ws[bucket],
                                  "w_fit": min(fcurve, key=fcurve.get),
                                  "logloss": curve[ws[bucket]]}
        w_vec = np.array([ws[b] for b in b_tune])
        ll = logloss_for_w(tune, w_vec)
        print(f"  thin_n={thin_n:>6}: per-bucket w={ws} -> TUNE log-loss {ll:.5f}")
        if best is None or ll < best[0]:
            best = (ll, thin_n, ws, per_bucket)

    ll_bucketed, thin_n, ws, per_bucket = best
    params = CB.CombineParams(w=float(w_sel), w_both_well=float(ws[0]),
                              w_one_thin=float(ws[1]), w_both_thin=float(ws[2]),
                              thin_n=float(thin_n))

    ll_w0, ll_w1 = tune_curve[0.0], tune_curve[1.0]
    ll_overall = tune_curve[w_sel]

    # --- FIT-internal rolling origin (gain over the better endpoint) ------
    baseline_w = 0.0 if ll_w0 <= ll_w1 else 1.0
    edges = np.quantile(fit["t"], np.linspace(0.5, 1.0, C.ROLLING_ORIGIN_FOLDS + 1))
    folds = []
    for lo, hi in zip(edges, edges[1:]):
        sub = fit[(fit["t"] >= lo) & (fit["t"] < hi)]
        b = CB.bucket_of(sub["n_a"].to_numpy(), sub["n_b"].to_numpy(), thin_n)
        w_vec = np.array([ws[x] for x in b])
        folds.append(logloss_for_w(sub, baseline_w) - logloss_for_w(sub, w_vec))
    tune_gain = min(ll_w0, ll_w1) - ll_bucketed

    selected = {"w": float(w_sel), "w_both_well": float(ws[0]),
                "w_one_thin": float(ws[1]), "w_both_thin": float(ws[2]),
                "thin_n": float(thin_n), "w_fitted_on_fit": float(w_fit)}
    ledger.append(
        stage="stage_4", metric="match_winner_logloss", selected=selected,
        tune_gain=tune_gain, fit_rolling_origin_gains=folds,
        baseline=f"w={baseline_w:g}", tune_metric_value=ll_bucketed,
        baselines={"w0_logloss": ll_w0, "w1_logloss": ll_w1,
                   "overall_w_logloss": ll_overall},
        frozen=False, notes="blend weight, overall and per sample-size bucket",
    )

    # --- gate -------------------------------------------------------------
    passed = ll_bucketed < ll_w0 and ll_bucketed < ll_w1

    dis = panel["disagreement_pp"]
    lines = [
        "# Stage 4 validation — combining serve rates and Elo\n",
        f"Level (pa + pb) comes from the Stage 2 rate model; the split is "
        f"`(1-w) * serve_gap + w * elo_gap`, with the Elo opinion converted to "
        f"a serve gap by inverting the Stage 1 engine on a "
        f"{len(CB.LEVEL_GRID)}x{len(CB.GAP_GRID)} (level, gap) grid.\n",
        f"\nSelected on TUNE: overall w **{w_sel:.2f}** (w fitted on FIT alone "
        f"would be {w_fit:.2f}), thin threshold **{thin_n:.0f}** effective "
        "serve points.\n",
        "\n## Blend weight by sample-size bucket\n",
        "| bucket | TUNE matches | w (TUNE) | w (FIT) |",
        "|---|---|---|---|",
    ]
    for b, label in ((0, "both well sampled"), (1, "one thin"), (2, "both thin")):
        pb = per_bucket[b]
        lines.append(f"| {label} | {pb['n_tune']:,} | {pb['w']:.2f} | "
                     f"{pb.get('w_fit', float('nan')):.2f} |")
    lines += [
        "\n## TUNE match-winner log-loss\n",
        "| blend | log-loss |",
        "|---|---|",
        f"| **fitted w per bucket** | **{ll_bucketed:.5f}** |",
        f"| fitted w overall ({w_sel:.2f}) | {ll_overall:.5f} |",
        f"| w = 0 (serve rates only) | {ll_w0:.5f} |",
        f"| w = 1 (Elo only) | {ll_w1:.5f} |",
        f"\nn = {len(tune):,} TUNE matches.\n",
        "\n## Elo vs serve disagreement (metadata, not absorbed into the blend)\n",
        "Percentage points of match win probability, all matches in the panel:\n",
        f"\n- mean {dis.mean():+.2f}pp, median {dis.median():+.2f}pp",
        f"\n- absolute: mean {dis.abs().mean():.2f}pp, "
        f"p50 {dis.abs().quantile(0.5):.2f}, p90 {dis.abs().quantile(0.9):.2f}, "
        f"p99 {dis.abs().quantile(0.99):.2f}, max {dis.abs().max():.2f}",
        f"\n- share above 10pp: {(dis.abs() > 10).mean():.1%}; "
        f"above 20pp: {(dis.abs() > 20).mean():.1%}\n",
        "\nThis is carried per match into price.py's metadata alongside each "
        "side's effective n, for downstream stake and confidence decisions.\n",
        "\n## FIT-internal rolling origin\n",
        f"Log-loss gain over w={baseline_w:g}, per fold: "
        + ", ".join(f"{g:+.5f}" for g in folds),
        f"\n\nmean {np.mean(folds):+.5f}, std {np.std(folds, ddof=1):.5f}; "
        f"TUNE gain {tune_gain:+.5f}, envelope "
        f"{C.OVERFIT_SIGNAL_MULTIPLE * np.std(folds, ddof=1):.5f}.\n",
        f"\n## Gate\n\nFitted w beats w=0 ({ll_bucketed:.5f} < {ll_w0:.5f}) and "
        f"w=1 ({ll_bucketed:.5f} < {ll_w1:.5f}). "
        f"Gate **{'PASSED' if passed else 'FAILED'}**.\n",
    ]
    out = C.STAGE_VALIDATIONS_DIR / "stage_4.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))

    if passed:
        ledger.append(
            stage="stage_4", metric="match_winner_logloss", selected=selected,
            tune_gain=tune_gain, fit_rolling_origin_gains=folds,
            baseline=f"w={baseline_w:g}", tune_metric_value=ll_bucketed,
            baselines={"w0_logloss": ll_w0, "w1_logloss": ll_w1},
            frozen=True, notes="gate passed; frozen for downstream stages",
        )
        fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
        fitted["stage_4"] = {**selected, "selected_on": "tune",
                             "tune_logloss": ll_bucketed,
                             "disagreement_pp_mean_abs": float(dis.abs().mean())}
        C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
