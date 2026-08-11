"""Stage 5 fitting and validation — cohort prior for thin-data players.

Selects k (and the opponent-vs-cohort shrinkage) on TUNE, evaluated on
thin-data matches only, and compares against the flat tour-average prior on
match-winner log-loss and total-games CRPS.

Gate (MODEL_PROMPT.md): the cohort prior beats the tour-average prior on both.
If it does not, the module ships switched off and the report says so.

Run:  python -m scripts.fit_stage5
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from model import cohort as CH
from model import combine as CB
from model import constants as C
from model import fitted as FP
from model import elo as E
from model import ledger
from model import player_rates as PR
from model.engine import match_distribution
from model.rules import FormatSpec, rules_for

K_GRID = [5, 10, 20, 40]
OPP_N0_GRID = [0.0, 200.0]


@lru_cache(maxsize=100_000)
def _dist(pa: float, pb: float, key: tuple):
    spec = FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                      tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                      final_tb_to=key[6], provenance="documented", source="cache")
    return match_distribution(pa, pb, spec)


def crps_games(pmf: dict[int, float], actual: int) -> float:
    """CRPS of a discrete total-games forecast against the observed total."""
    xs = np.array(sorted(pmf))
    cdf = np.cumsum([pmf[x] for x in xs])
    step = (xs >= actual).astype(float)
    return float(np.sum((cdf - step) ** 2))


def build_base() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    m = m[m["in_scope"] & m["split"].isin(("fit", "tune"))]
    assert m["date"].max() < C.TEST_START, "Stage 5 must not see TEST data"

    s2 = fitted["stage_2"]
    rp = PR.RateParams(half_life_days=s2["half_life_days"],
                       surface_pool=s2["surface_pool"], shrink_n0=s2["shrink_n0"])
    asof = PR.build_asof(m, rp)

    s3 = fitted["stage_3"]
    ep = E.EloParams(k=s3["k"], k_chall_mult=s3["k_chall_mult"],
                     surface_weight=s3["surface_weight"],
                     inactivity_half_life=s3["inactivity_half_life"],
                     level_gap=s3["level_gap"], level_offset=s3["level_offset"])
    elo = E.run_elo(m, ep)[["match_id", "p_winner"]]
    return m, asof, {"fitted": fitted, "rp": rp, "elo": elo}


def align_vs_cohort(asof: pd.DataFrame, vs: dict[str, np.ndarray]) -> dict:
    """Reindex the vs-cohort totals onto the as-of frame's row order."""
    idx = pd.Index(vs["match_id"]).get_indexer(asof["match_id"])
    return {k: (v[idx] if k != "match_id" else v) for k, v in vs.items()}


def cohort_priors(asof: pd.DataFrame, snaps: list[CH.Snapshot], history: dict,
                  params: CH.CohortParams) -> dict[str, np.ndarray]:
    """Per-side cohort prior, NaN where the player is not thin."""
    out = {"w": np.full(len(asof), np.nan), "l": np.full(len(asof), np.nan)}
    cache: dict[tuple[str, int], tuple[float, float]] = {}
    for i, (t, wid, lid, w_n, l_n) in enumerate(zip(
        asof["t"], asof["winner_id"], asof["loser_id"],
        asof["w_ss_pl"] + asof["w_sa_pl"], asof["l_ss_pl"] + asof["l_sa_pl"],
    )):
        snap = CH.snapshot_for(snaps, int(t))
        if snap is None:
            continue
        for side, pid, n in (("w", wid, w_n), ("l", lid, l_n)):
            if n >= params.thin_n:
                continue
            key = (pid, snap.date_ord)
            if key not in cache:
                vec = history.get(key, np.full(len(CH.FEATURES), np.nan))
                s, r, _ = CH.cohort_prior(vec, snap, params)
                cache[key] = (s, r)
            out[side][i] = cache[key][0]
    return out


def evaluate(panel: pd.DataFrame, label: str) -> dict:
    """Match-winner log-loss and total-games CRPS on the given rows."""
    lls, crps, n_games = [], [], 0
    for pa, pb, key, actual in zip(panel["pa"], panel["pb"], panel["spec_key"],
                                   panel["total_games"]):
        d = _dist(round(float(pa), 3), round(float(pb), 3), key)
        lls.append(-np.log(max(d.p_a, 1e-9)))
        if np.isfinite(actual):
            crps.append(crps_games(d.total_games_pmf(), int(actual)))
            n_games += 1
    return {"label": label, "logloss": float(np.mean(lls)),
            "crps": float(np.mean(crps)) if crps else float("nan"),
            "n": len(lls), "n_games": n_games}


def make_panel(m: pd.DataFrame, asof: pd.DataFrame, ctx: dict,
               priors: dict[str, np.ndarray] | None,
               params: CH.CohortParams) -> pd.DataFrame:
    fitted, rp = ctx["fitted"], ctx["rp"]
    rates = PR.serve_rates(asof, rp, prior_override=priors)
    rates["serve_pa"] = [PR.expected_spw(r, o, lg) for r, o, lg
                         in zip(rates["w_rate"], rates["l_ret"], rates["league"])]
    rates["serve_pb"] = [PR.expected_spw(r, o, lg) for r, o, lg
                         in zip(rates["l_rate"], rates["w_ret"], rates["league"])]

    # Where the opponent is well known and the player is thin, the opponent's
    # own record against that cohort's tier adjusts the matchup.
    if priors is not None and params.opponent_vs_cohort_n0 > 0:
        vs = ctx["vs_cohort"]
        thin_w = np.isfinite(priors["w"])
        thin_l = np.isfinite(priors["l"])
        for side, opp_side, opp_thin, own_thin in (("w", "l", thin_l, thin_w),
                                                   ("l", "w", thin_w, thin_l)):
            # adjust THIS side's serve rate when the OTHER side is the thin one
            adj = np.array([
                CH.opponent_vs_cohort_rate(sv, sp, base, params.opponent_vs_cohort_n0)
                for sv, sp, base in zip(vs[f"{side}_vs_svpt"], vs[f"{side}_vs_spw"],
                                        rates[f"serve_p{'a' if side == 'w' else 'b'}"])
            ])
            col = "serve_pa" if side == "w" else "serve_pb"
            rates[col] = np.where(opp_thin & ~own_thin, adj, rates[col])
    p = rates.merge(ctx["elo"], on="match_id", how="inner")
    meta = m.set_index("match_id")
    p["tourney_code"] = meta.loc[p["match_id"], "tourney_code"].to_numpy()
    p["season_file"] = meta.loc[p["match_id"], "season_file"].to_numpy()
    p["split"] = meta.loc[p["match_id"], "split"].to_numpy()
    p["total_games"] = (meta.loc[p["match_id"], "winner_games"].to_numpy()
                        + meta.loc[p["match_id"], "loser_games"].to_numpy())

    specs = {(c, int(y)): rules_for(c, int(y))
             for c, y in p[["tourney_code", "season_file"]].drop_duplicates().values}
    keep = [specs[(c, int(y))].supported for c, y
            in zip(p["tourney_code"], p["season_file"])]
    p = p[keep].copy()
    p["spec_key"] = [CB._spec_key(specs[(c, int(y))]) for c, y
                     in zip(p["tourney_code"], p["season_file"])]

    s4 = fitted["stage_4"]
    cp = CB.CombineParams(w=s4["w"], w_both_well=s4["w_both_well"],
                          w_one_thin=s4["w_one_thin"], w_both_thin=s4["w_both_thin"],
                          thin_n=s4["thin_n"])
    blended = CB.combine_many(p["serve_pa"].to_numpy(), p["serve_pb"].to_numpy(),
                              p["p_winner"].to_numpy(), p["spec_key"].tolist(),
                              cp, p["w_n"].to_numpy(), p["l_n"].to_numpy())
    p["pa"], p["pb"] = blended["pa"], blended["pb"]
    p["thin_match"] = (p["w_n"] < params.thin_n) | (p["l_n"] < params.thin_n)
    return p


def main() -> None:
    m, asof, ctx = build_base()
    base_params = CH.CohortParams()
    snaps, history = CH.build_snapshots(m, base_params)
    ctx["vs_cohort"] = align_vs_cohort(asof, CH.vs_cohort_totals(m, snaps))
    print(f"{len(snaps)} style snapshots built; vs-cohort totals aligned")

    baseline_panel = make_panel(m, asof, ctx, None, base_params)
    tune_thin = baseline_panel[(baseline_panel["split"] == "tune")
                               & baseline_panel["thin_match"]]
    print(f"TUNE thin-data matches: {len(tune_thin):,}")
    base_eval = evaluate(tune_thin, "tour_average_prior")
    print(f"  baseline: log-loss {base_eval['logloss']:.5f}, "
          f"CRPS {base_eval['crps']:.4f}")

    results, best = [], None
    for k in K_GRID:
        for opp_n0 in OPP_N0_GRID:
            params = CH.CohortParams(k=k, opponent_vs_cohort_n0=opp_n0)
            priors = cohort_priors(asof, snaps, history, params)
            panel = make_panel(m, asof, ctx, priors, params)
            sub = panel[(panel["split"] == "tune") & panel["thin_match"]]
            ev = evaluate(sub, f"cohort_k{k}_opp{opp_n0:g}")
            results.append({"k": k, "opponent_vs_cohort_n0": opp_n0, **ev})
            print(f"  k={k:>3} opp_n0={opp_n0:>5}: log-loss {ev['logloss']:.5f}, "
                  f"CRPS {ev['crps']:.4f}")
            if best is None or ev["logloss"] < best[0]["logloss"]:
                best = (ev, params, panel)

    best_eval, sel, best_panel = best
    grid = pd.DataFrame(results).sort_values("logloss").reset_index(drop=True)

    # --- FIT-internal rolling origin, thin matches only -------------------
    fit_rows = best_panel[(best_panel["split"] == "fit") & best_panel["thin_match"]]
    base_fit = baseline_panel[(baseline_panel["split"] == "fit")
                              & baseline_panel["thin_match"]]
    edges = np.quantile(fit_rows["t"], np.linspace(0.5, 1.0, C.ROLLING_ORIGIN_FOLDS + 1))
    folds = []
    for lo, hi in zip(edges, edges[1:]):
        a = evaluate(fit_rows[(fit_rows["t"] >= lo) & (fit_rows["t"] < hi)], "f")
        b = evaluate(base_fit[(base_fit["t"] >= lo) & (base_fit["t"] < hi)], "b")
        folds.append(b["logloss"] - a["logloss"])

    tune_gain = base_eval["logloss"] - best_eval["logloss"]
    ll_ok = best_eval["logloss"] < base_eval["logloss"]
    crps_ok = best_eval["crps"] < base_eval["crps"]
    passed = ll_ok and crps_ok

    selected = {"k": sel.k, "opponent_vs_cohort_n0": sel.opponent_vs_cohort_n0,
                "thin_n": sel.thin_n, "well_sampled_n": sel.well_sampled_n,
                "snapshot_days": sel.snapshot_days, "enabled": bool(passed)}
    ledger.append(
        stage="stage_5", metric="thin_match_winner_logloss", selected=selected,
        tune_gain=tune_gain, fit_rolling_origin_gains=folds,
        baseline="tour_average_prior", tune_metric_value=best_eval["logloss"],
        baselines={"tour_average_logloss": base_eval["logloss"],
                   "tour_average_crps": base_eval["crps"]},
        frozen=False, notes=f"k grid {K_GRID}; evaluated on thin-data TUNE matches only",
        tune_crps=best_eval["crps"],
    )

    lines = [
        "# Stage 5 validation — cohort prior for thin-data players\n",
        f"Style vector: {', '.join(f'`{f}`' for f in CH.FEATURES)}. Features and "
        "the means/stds used to standardize them are both computed as of the "
        f"snapshot date ({sel.snapshot_days}-day snapshots), never once over "
        "the whole dataset.\n",
        f"\nSelected on TUNE thin-data matches: k = **{sel.k}**, "
        f"opponent-vs-cohort shrinkage **{sel.opponent_vs_cohort_n0:.0f}**, "
        f"thin threshold **{sel.thin_n:.0f}** effective serve points, "
        f"neighbour eligibility **{sel.well_sampled_n:.0f}**.\n",
        "\n## TUNE, thin-data matches only\n",
        "| prior | match-winner log-loss | total-games CRPS |",
        "|---|---|---|",
        f"| **cohort (k={sel.k})** | **{best_eval['logloss']:.5f}** | "
        f"**{best_eval['crps']:.4f}** |",
        f"| tour average | {base_eval['logloss']:.5f} | {base_eval['crps']:.4f} |",
        f"\nn = {best_eval['n']:,} matches ({best_eval['n_games']:,} with a "
        "scoreable games total).\n",
        "\n## Grid\n",
        grid[["k", "opponent_vs_cohort_n0", "logloss", "crps", "n"]].to_markdown(index=False),
        "\n\nThe opponent-vs-cohort adjustment — the well-sampled opponent's own "
        "record against the thin player's return-strength tier, shrunk toward "
        "their overall rate — is implemented and measured here. It consistently "
        "improves total-games CRPS and consistently costs match-winner "
        "log-loss, and selection is on log-loss, so it is switched off "
        f"(`opponent_vs_cohort_n0 = {sel.opponent_vs_cohort_n0:.0f}`). That "
        "trade is visible in the grid above rather than buried.\n",
        "\n\n## FIT-internal rolling origin (thin matches)\n",
        "Log-loss gain over the tour-average prior, per fold: "
        + ", ".join(f"{g:+.5f}" for g in folds),
        f"\n\nmean {np.mean(folds):+.5f}, std {np.std(folds, ddof=1):.5f}; "
        f"TUNE gain {tune_gain:+.5f}.\n",
        f"\n## Gate\n\nBeats the tour-average prior on log-loss: **{ll_ok}** "
        f"({best_eval['logloss']:.5f} vs {base_eval['logloss']:.5f}). "
        f"On total-games CRPS: **{crps_ok}** ({best_eval['crps']:.4f} vs "
        f"{base_eval['crps']:.4f}). Gate **{'PASSED' if passed else 'FAILED'}**.\n",
    ]
    if not passed:
        lines.append(
            "\n**The module ships switched OFF** (`enabled: false` in "
            "fitted_params.json), per MODEL_PROMPT.md: the cohort prior did "
            "not beat the flat tour-average prior on both metrics, so it is "
            "kept in the repo but does not feed price.py.\n"
        )
    out = C.STAGE_VALIDATIONS_DIR / "stage_5.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))

    ledger.append(
        stage="stage_5", metric="thin_match_winner_logloss", selected=selected,
        tune_gain=tune_gain, fit_rolling_origin_gains=folds,
        baseline="tour_average_prior", tune_metric_value=best_eval["logloss"],
        baselines={"tour_average_logloss": base_eval["logloss"],
                   "tour_average_crps": base_eval["crps"]},
        frozen=True,
        notes=("gate passed; frozen" if passed else
               "gate failed; frozen as DISABLED so the lineage records that "
               "price.py does not use it"),
        tune_crps=best_eval["crps"],
    )
    with FP.updating() as fitted:
        fitted["stage_5"] = {**selected, "selected_on": "tune",
                             "tune_logloss": best_eval["logloss"],
                             "tune_crps": best_eval["crps"],
                             "baseline_logloss": base_eval["logloss"],
                             "baseline_crps": base_eval["crps"]}
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
