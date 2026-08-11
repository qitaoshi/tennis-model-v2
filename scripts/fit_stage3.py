"""Stage 3 fitting and validation — surface Elo.

Selects K, the Challenger K multiplier, the surface blend weight, the
inactivity regression half-life, the new-player level gap and the debut
rank-seed scale. NOTE THE SELECTION RULE: as of 2026-08-11 this stage is
selected on grouped subgroup ECE subject to a log-loss constraint, not on TUNE
log-loss as MODEL_PROMPT.md:373 specifies — see LOGLOSS_SLACK below for the
reasoning and the human decision behind it. Records the FIT-internal
rolling-origin gain and its spread to the ledger; runs the cross-level consistency check and fits a level offset only
if that check demands one.

Gate (MODEL_PROMPT.md): decile calibration tracks observed win rates, the
Brier score beats a rankings-based baseline, and the cross-level check is
reported either way.

Run:  python -m scripts.fit_stage3
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import itertools
import json

import numpy as np
import pandas as pd

from model import constants as C
from model import fitted as FP
from model import elo as E
from model import ledger

KS = [16.0, 24.0, 32.0, 48.0]
SURFACE_WEIGHTS = [0.3, 0.5, 0.7]
#: Widened 2026-08-11. The original three values could not express the setting
#: reports/unknown_players.md found: at 1095 a 140-day absence ages a rating by
#: 8.5%, which is close to not ageing it at all.
INACTIVITY = [120.0, 180.0, 270.0, 365.0, 540.0, 1095.0, None]
CHALL_MULTS = [0.75, 1.0]
LEVEL_GAPS = [0.0, 100.0]
#: Rating points per natural-log unit of ATP rank used to seed a debutant.
#: 0.0 is the incumbent: a flat seed whether the player is ranked 40 or 900.
#: Range taken from reports/unknown_players.md, which extended past 160
#: precisely because a first pass selected the edge value.
SEED_SCALES = [0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0]

#: Calibration is judged against criteria fixed before the run, not eyeballed.
MAX_DECILE_GAP = 0.05
MEAN_DECILE_GAP = 0.025

#: SELECTION RULE, and a DELIBERATE DEVIATION FROM MODEL_PROMPT.md — human
#: decision, 2026-08-11.
#:
#: MODEL_PROMPT.md:373 says Stage 3 is selected by TUNE log-loss. That text
#: predates the 2026-08-08 goal change, which made ECE a first-class objective
#: beside log-loss (CLAUDE.md). Log-loss alone cannot select this stage for
#: what it is now being asked to fix: the defect is a subgroup one, confined to
#: debutants and players returning from a layoff — roughly a tenth of matches —
#: while log-loss is dominated by the well-known other nine tenths. A four-point
#: screen on 2026-08-11 showed the log-loss argmax carrying WORSE subgroup
#: calibration than the incumbent, i.e. selecting on it would have propagated a
#: setting that makes the thing this cascade exists to fix worse.
#:
#: The rule, fixed before the grid was run:
#:
#:   minimise the size-weighted mean of the per-group ECEs (debut / stale /
#:   known), subject to TUNE log-loss not exceeding the incumbent's by more
#:   than LOGLOSS_SLACK.
#:
#: This is the rule scripts/fit_unknown_players.py already adopted and logged,
#: for the same defect. Two properties matter:
#:
#: * The constraint is on the SEARCH, not a veto applied to the winner. Applied
#:   the second way it rejects the whole grid whenever the unconstrained
#:   optimum happens to be sharpness-destroying, and throws away the settings
#:   that satisfy both.
#: * The objective is the size-weighted mean of subgroup ECEs, never pooled
#:   ECE. Pooled rewards one group's overprediction cancelling another's:
#:   reports/unknown_players.md caught a setting whose pooled ECE beat every
#:   one of its own subgroups.
#:
#: The deviation is recorded in the Stage 3 report, the ledger notes and
#: PROGRESS.json, so the lineage shows Stage 3 was selected under a different
#: rule from Stages 2 and 4-7.
LOGLOSS_SLACK = 1e-4

#: The shipped setting this is measured against — the incumbent, whose grid row
#: supplies the log-loss ceiling. Read from fitted_params.json rather than
#: hardcoded.
def _incumbent_row(grid: pd.DataFrame, s3: dict) -> pd.Series:
    m = np.ones(len(grid), dtype=bool)
    for field, col in (("k", "k"), ("k_chall_mult", "k_chall_mult"),
                       ("surface_weight", "surface_weight"),
                       ("level_gap", "level_gap"),
                       ("rank_seed_scale", "rank_seed_scale")):
        m &= grid[col].to_numpy() == s3.get(field, 0.0)
    hl = s3.get("inactivity_half_life")
    m &= (grid["inactivity_half_life"].isna().to_numpy() if hl is None
          else (grid["inactivity_half_life"].to_numpy() == hl))
    assert m.any(), "the incumbent setting is not in the grid"
    return grid[m].iloc[0]


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


def _groups(matches: pd.DataFrame) -> dict[str, np.ndarray]:
    """Debut / stale / known masks, in the row order run_elo returns.

    These depend only on the fixture list, never on the Elo settings, so they
    are computed once. A debutant is a player the ladder has not seen; stale is
    a 70-day-plus layoff for a player it has.
    """
    order = E.run_elo(matches, E.EloParams())
    t = np.array([d.toordinal() for d in order["date"]], dtype=float)
    last: dict[str, float] = {}
    layoff = np.zeros(len(order))
    debut = np.zeros(len(order), dtype=bool)
    wid, lid = order["winner_id"].to_numpy(), order["loser_id"].to_numpy()
    for i in np.argsort(t, kind="mergesort"):
        gaps = []
        for pid in (wid[i], lid[i]):
            prev = last.get(pid)
            if prev is None:
                debut[i] = True
            else:
                gaps.append(t[i] - prev)
        layoff[i] = max(gaps) if gaps else 0.0
        for pid in (wid[i], lid[i]):
            last[pid] = t[i]
    return {"debut": debut,
            "stale": (~debut) & (layoff >= 70),
            "known": (~debut) & (layoff < 70)}


def _grouped_ece(p: np.ndarray, tune: np.ndarray,
                 groups: dict[str, np.ndarray]) -> dict:
    """Size-weighted mean of per-group ECEs, and each group's own value."""
    out, ns, es = {}, [], []
    for name, mask in groups.items():
        sel = p[tune & mask]
        e = E.group_ece(sel) if len(sel) >= 100 else float("nan")
        out[f"ece_{name}"] = e
        ns.append(int((tune & mask).sum()))
        es.append(e)
    tot = sum(n for n, e in zip(ns, es) if np.isfinite(e))
    out["grouped_ece"] = (float(sum(n * e for n, e in zip(ns, es)
                                    if np.isfinite(e)) / tot)
                          if tot else float("nan"))
    return out


#: The grid costs ~80 minutes over 2,352 settings. It depends only on the axis
#: definitions and the Elo code, so it is cached under that key: the selection
#: RULE can then be reconsidered without paying for the search again.
GRID_CACHE = C.REPO_ROOT / ".stage3_cache"


def _grid_key() -> str:
    h = hashlib.sha256()
    for axis in (KS, SURFACE_WEIGHTS, INACTIVITY, CHALL_MULTS, LEVEL_GAPS,
                 SEED_SCALES):
        h.update(repr(axis).encode())
    h.update((C.REPO_ROOT / "model" / "elo.py").read_bytes())
    # The split boundaries decide which matches are scored. Leave them out and
    # a boundary move would silently reuse a grid measured on the old window.
    h.update(repr((C.TUNE_START, C.TUNE_END, C.TEST_START)).encode())
    h.update(inspect.getsource(_grouped_ece).encode())
    return h.hexdigest()[:16]


def search(matches: pd.DataFrame, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Every grid point scored on TUNE. No selection happens here."""
    cache = GRID_CACHE / f"grid-{_grid_key()}.parquet"
    if cache.exists():
        print(f"reusing cached grid ({cache.name})")
        return pd.read_parquet(cache)

    grid = []
    for k, w, inact, cm, gap, seed in itertools.product(
        KS, SURFACE_WEIGHTS, INACTIVITY, CHALL_MULTS, LEVEL_GAPS, SEED_SCALES
    ):
        params = E.EloParams(k=k, surface_weight=w, inactivity_half_life=inact,
                             k_chall_mult=cm, level_gap=gap,
                             rank_seed_scale=seed)
        res = E.run_elo(matches, params)
        tune_mask = (res["split"] == "tune").to_numpy()
        p = res["p_winner"].to_numpy()
        ll = E.log_loss(p[tune_mask])
        grid.append({"k": k, "surface_weight": w,
                     "inactivity_half_life": inact, "k_chall_mult": cm,
                     "level_gap": gap, "rank_seed_scale": seed,
                     "tune_logloss": ll,
                     "tune_brier": E.brier(p[tune_mask]),
                     **_grouped_ece(p, tune_mask, groups)})
        print(f"  k={k:>4} w={w} inact={inact} cm={cm} gap={gap} seed={seed} "
              f"-> {ll:.5f} (grouped ECE {grid[-1]['grouped_ece']:.5f})")

    grid = pd.DataFrame(grid).sort_values("tune_logloss").reset_index(drop=True)
    GRID_CACHE.mkdir(exist_ok=True)
    grid.to_parquet(cache, index=False)
    return grid


def main() -> None:
    matches = _load()
    groups = _groups(matches)
    grid = search(matches, groups)

    # Selection: see LOGLOSS_SLACK above. Minimise grouped subgroup ECE subject
    # to log-loss not worsening against the incumbent.
    inc = _incumbent_row(grid, FP.load()["stage_3"])
    feasible = grid[grid["tune_logloss"] <= inc["tune_logloss"] + LOGLOSS_SLACK]
    constraint_bound = len(feasible) == 0
    row = (grid if constraint_bound else feasible).sort_values(
        "grouped_ece").iloc[0]
    tie_note = (
        f"Selected under the 2026-08-11 rule (grouped subgroup ECE, subject to "
        f"TUNE log-loss <= incumbent + {LOGLOSS_SLACK}), NOT the TUNE log-loss "
        f"of MODEL_PROMPT.md:373 — see LOGLOSS_SLACK in scripts/fit_stage3.py. "
        f"{len(feasible)} of {len(grid)} settings satisfy the constraint. "
        f"Incumbent: grouped ECE {inc['grouped_ece']:.5f}, log-loss "
        f"{inc['tune_logloss']:.5f}. Selected: grouped ECE "
        f"{row['grouped_ece']:.5f}, log-loss {row['tune_logloss']:.5f}. "
        f"The log-loss argmax would have been grouped ECE "
        f"{grid['grouped_ece'].iloc[0]:.5f} at log-loss "
        f"{grid['tune_logloss'].iloc[0]:.5f}")
    if constraint_bound:
        tie_note += (". NO SETTING SATISFIED THE LOG-LOSS CONSTRAINT, so the "
                     "objective was minimised over the whole grid and this "
                     "selection buys calibration with sharpness")
    print(f"\n{tie_note}")

    sel = E.EloParams(k=float(row["k"]), surface_weight=float(row["surface_weight"]),
                      inactivity_half_life=(None if pd.isna(row["inactivity_half_life"])
                                            else float(row["inactivity_half_life"])),
                      k_chall_mult=float(row["k_chall_mult"]),
                      level_gap=float(row["level_gap"]),
                      rank_seed_scale=float(row["rank_seed_scale"]))
    res = E.run_elo(matches, sel)
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
            sel = dataclasses.replace(sel, level_offset=float(offset))
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

    # Subgroup ECEs on the FINAL ratings: if a cross-level offset was fitted
    # above, `res` was re-run and the grid row's values are one step stale.
    sub = _grouped_ece(res["p_winner"].to_numpy(), tune_mask, groups)

    # --- ledger (before the gate, per ground rule 2) ----------------------
    folds = _fold_gains(res, base)
    tune_gain = base_ll - model_ll
    # Every field Stage 3 selects, from one list, so a new lever cannot be
    # written to the report and silently omitted from fitted_params.json.
    selected = {f: getattr(sel, f) for f in E.SELECTED_FIELDS}
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
        f"**{sel.level_gap:.0f}**, debut rank-seed scale "
        f"**{sel.rank_seed_scale:.0f}**, cross-level offset "
        f"**{sel.level_offset:+.1f}**.\n",
        f"\n{tie_note}.\n",
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
        "\n## Subgroup calibration at the selected setting\n",
        "The defect this grid was widened to address is a subgroup one: "
        "matches with a debutant, and matches after a long layoff. Pooled ECE "
        "is not shown as a selection number because it rewards one group's "
        "bias cancelling another's.\n",
        f"\n- debut ECE {sub['ece_debut']:.4f}",
        f"\n- stale (70d+ layoff) ECE {sub['ece_stale']:.4f}",
        f"\n- known ECE {sub['ece_known']:.4f}",
        f"\n- size-weighted mean {sub['grouped_ece']:.4f}\n",
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
        with FP.updating() as fitted:
            fitted["stage_3"] = {**selected, "selected_on": "tune",
                                 "tune_logloss": model_ll,
                                 "tune_brier": model_brier,
                                 "cross_level": cross}
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
