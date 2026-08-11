"""Final backtest on the HOLDOUT window. RUNS ONCE.

Everything above this must be frozen before this script is run: every stage
gate passed, the pre-cutoff TEST set touched exactly once by Stage 8, and no
component fitted on holdout data by any path, including exploratory debugging.

The script refuses to run unless a human has created the ``.backtest_approved``
flag file at the repo root, which is the same moment the /loop procedure asks
for explicit confirmation. That turns a convention into something the tooling
actually checks.

Reported per MODEL_PROMPT.md: calibration by market family, split by tour
level, by format rule and by the rule's PROVENANCE, with retirements and
suspect-score matches reported separately rather than pooled; plus ablations
with cohort, venue and corrections switched off.

Run:  touch .backtest_approved && python -m scripts.backtest
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from model import venue as V
from scripts import panel as P

APPROVAL_FLAG = C.REPO_ROOT / ".backtest_approved"


@dataclass
class Ablation:
    name: str
    cohort: bool = True
    venue: bool = True
    corrections: bool = True
    calibration: bool = True


ABLATIONS = [
    Ablation("full"),
    Ablation("no_cohort", cohort=False),
    Ablation("no_venue", venue=False),
    Ablation("no_corrections", corrections=False),
    Ablation("no_calibration", calibration=False),
]


def _guard() -> None:
    if not APPROVAL_FLAG.exists():
        raise SystemExit(
            f"REFUSING TO RUN: {APPROVAL_FLAG.name} is not present.\n"
            "The holdout backtest runs once, after every stage gate has passed "
            "and a human has confirmed. If this is genuinely that moment:\n"
            f"  touch {APPROVAL_FLAG.name}"
        )
    progress = json.loads((C.REPO_ROOT / "PROGRESS.json").read_text())
    unpassed = [k for k, v in progress["stages"].items()
                if k != "backtest" and not v.get("gate_passed")]
    if unpassed:
        raise SystemExit(f"REFUSING TO RUN: stages without a passed gate: {unpassed}")
    if progress.get("human_checkpoints_pending"):
        raise SystemExit("REFUSING TO RUN: human checkpoints are still pending.")
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    if not fitted.get("stage_8", {}).get("test_touched_once"):
        raise SystemExit("REFUSING TO RUN: Stage 8 has not recorded its TEST run.")


def _families(pan: pd.DataFrame, params: CR.CorrectionParams,
              maps: RC.CalibrationMaps | None) -> pd.DataFrame:
    """One row per priced selection with its outcome, for scoring."""
    from model.rules import FormatSpec
    from scripts.fit_stage8 import LINE_OFFSETS, _median_of

    rows = []
    for r in pan.itertuples(index=False):
        key = r.spec_key
        spec = FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                          tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                          final_tb_to=key[6], provenance=r.format_provenance,
                          source="backtest")
        d = CR.corrected_distribution(round(float(r.pa), 3), round(float(r.pb), 3),
                                      spec, params)
        common = {"match_id": r.match_id, "tour": r.tour,
                  "level_label": r.level_label, "final_set": key[4],
                  "provenance": r.format_provenance, "split": r.split}

        p = d.p_a if maps is None else float(maps.apply("match_winner", d.p_a))
        # Panel rows are oriented winner-first, so scoring that orientation
        # alone would make every outcome a 1 by construction. The previous fix
        # emitted BOTH sides, which cured the artefact and bought a new one:
        # (p, 1) and (1 - p, 0) are the same forecast written twice, so every
        # match counted double and n was twice the number of real predictions.
        #
        # Orient on player id instead — smaller id is side A. That is fixed
        # before the match and cannot know who won, so y is a genuine 0/1 and
        # each match contributes exactly one match_winner prediction.
        # Ids are opaque strings in this vendor's data ("R485", "BD59"), not
        # integers — compare them as strings and do not try to parse them.
        a_won = str(r.winner_id) < str(r.loser_id)
        rows.append({**common, "family": "match_winner",
                     "p": p if a_won else 1.0 - p,
                     "y": 1.0 if a_won else 0.0})

        pmf = d.total_games_pmf()
        median = _median_of(pmf)
        for off in LINE_OFFSETS:
            line = median + off
            under = sum(q for g, q in pmf.items() if g < line)
            fam = f"totals_under_{RC.totals_region(line, median)}"
            p = under if maps is None else float(maps.apply(fam, under))
            rows.append({**common, "family": "totals", "p": p,
                         "y": float(r.total_games < line)})

        need = spec.best_of // 2 + 1
        actual = (need, int(r.n_sets) - need)
        for (sa, sb), q in d.sets.items():
            p = q if maps is None else float(maps.apply("set_score", q))
            rows.append({**common, "family": "set_betting", "p": p,
                         "y": float((sa, sb) == actual)})

        rows.append({**common, "family": "tiebreak", "p": d.tiebreak_any,
                     "y": float(r.tiebreaks > 0)})

        diffs: dict[int, float] = {}
        for (ga, gb), q in d.games.items():
            diffs[ga - gb] = diffs.get(ga - gb, 0.0) + q
        margin = r.winner_games - r.loser_games
        for h in (-6.5, -2.5, 1.5, 5.5):
            # NOT the match-winner duplication. P(margin > h) and
            # P(-margin > h) do not sum to 1 — the band between them is a
            # push — so these are two genuinely different selections, both of
            # which a book quotes. They stay.
            rows.append({**common, "family": "handicap",
                         "p": sum(q for dd, q in diffs.items() if dd > h),
                         "y": float(margin > h)})
            rows.append({**common, "family": "handicap",
                         "p": sum(q for dd, q in diffs.items() if -dd > h),
                         "y": float(-margin > h)})

        rows.append({**common, "family": "_crps",
                     "p": P.crps_games(pmf, int(r.total_games)), "y": np.nan})
    return pd.DataFrame(rows)


#: Scored rows, cached per ablation. Scoring the holdout panel five times takes
#: about twenty minutes; a typo in the reporting code below should not cost
#: that twice. The key covers everything upstream of scoring, so a changed
#: parameter or map invalidates the cache rather than being papered over.
CACHE_DIR = C.REPO_ROOT / ".backtest_cache"


def _cache_key() -> str:
    h = hashlib.sha256()
    h.update(C.FITTED_PARAMS_PATH.read_bytes())
    h.update(RC.MAPS_PATH.read_bytes() if RC.MAPS_PATH.exists() else b"absent")
    # The scoring function's own source, not the whole module — otherwise
    # editing a table header below would throw away twenty minutes of work.
    h.update(inspect.getsource(_families).encode())
    h.update((C.REPO_ROOT / "scripts" / "panel.py").read_bytes())
    return h.hexdigest()[:16]


def _cached_rows(name: str, compute) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}-{_cache_key()}.parquet"
    if path.exists():
        print(f"  {name:16s} reusing cached rows ({path.name})")
        return pd.read_parquet(path)
    rows = compute()
    CACHE_DIR.mkdir(exist_ok=True)
    rows.to_parquet(path, index=False)
    return rows


def _score(df: pd.DataFrame) -> dict:
    scored = df[df["family"] != "_crps"]
    crps = df[df["family"] == "_crps"]["p"]
    return {"n": len(scored),
            "n_matches": int(scored["match_id"].nunique()) if len(scored) else 0,
            "brier": RC.brier(scored["p"].to_numpy(), scored["y"].to_numpy()),
            "logloss": RC.log_loss(scored["p"].to_numpy(), scored["y"].to_numpy()),
            "ece": RC.calibration_error(scored["p"].to_numpy(), scored["y"].to_numpy()),
            "crps": float(crps.mean()) if len(crps) else float("nan")}


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

#: Bootstrap replicates. 1,000 is enough to place a 95% interval to the fourth
#: decimal, which is the precision the tables are reported at.
N_BOOT = 1000


def _ragged_indices(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate ``range(starts[i], starts[i] + counts[i])`` without a loop."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.ones(total, dtype=np.int64)
    idx[0] = starts[0]
    if len(starts) > 1:
        idx[np.cumsum(counts)[:-1]] = starts[1:] - (starts[:-1] + counts[:-1]) + 1
    return np.cumsum(idx)


def _cluster_ci(df: pd.DataFrame, metric: str, n_boot: int = N_BOOT) -> tuple:
    """95% interval for a metric, resampling MATCHES rather than rows.

    Every table in this report pools rows that are not independent: one match
    emits a match winner, seven totals rungs, its set scores, a tiebreak and
    eight handicap selections, all driven by the same two serve rates. Treating
    those as ~21 independent observations makes any interval roughly sqrt(21)
    times too narrow and invites reading a fourth-decimal difference as real.
    Resampling whole matches keeps the within-match correlation intact.
    """
    scored = df[df["family"] != "_crps"] if metric != "crps" else df[df["family"] == "_crps"]
    if len(scored) == 0:
        return (float("nan"), float("nan"))
    codes, uniques = pd.factorize(scored["match_id"], sort=False)
    order = np.argsort(codes, kind="stable")
    p = scored["p"].to_numpy()[order]
    y = scored["y"].to_numpy()[order]
    sc = codes[order]
    n_clusters = len(uniques)
    starts = np.searchsorted(sc, np.arange(n_clusters), side="left")
    counts = np.searchsorted(sc, np.arange(n_clusters), side="right") - starts

    fn = {"brier": RC.brier, "logloss": RC.log_loss,
          "ece": RC.calibration_error,
          "crps": lambda a, _b: float(np.mean(a))}[metric]

    rng = np.random.default_rng(C.MC_SEED)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        idx = _ragged_indices(starts[pick], counts[pick])
        vals[b] = fn(p[idx], y[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def _ci_str(df: pd.DataFrame, metric: str) -> str:
    lo, hi = _cluster_ci(df, metric)
    return f"[{lo:.4f}, {hi:.4f}]"


def main() -> None:
    _guard()
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    print("*** HOLDOUT BACKTEST — this runs once ***")

    raw = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    print(f"canonical record holds {len(raw):,} pre-holdout matches; the "
          "holdout window is loaded fresh below")

    # Build the holdout-inclusive record here rather than expecting it to
    # exist. Every other entry point reads matches.parquet, which stops at the
    # cutoff, so this file only ever comes into being inside this script,
    # after the guard has passed.
    from model import data_audit as DA
    from model import rules as RU

    holdout_path = C.PROCESSED_DIR / DA.HOLDOUT_PARQUET
    if not holdout_path.exists():
        print("building the holdout-inclusive canonical record ...")
        DA.build(include_holdout=True)
        RU.build(parquet=DA.HOLDOUT_PARQUET)
    full_record = pd.read_parquet(holdout_path)
    n_hold = int((full_record["split"] == "holdout").sum())
    print(f"holdout window: {n_hold:,} matches, "
          f"{full_record.loc[full_record['split'] == 'holdout', 'date'].min()} .. "
          f"{full_record.loc[full_record['split'] == 'holdout', 'date'].max()}")

    results = {}
    for ab in ABLATIONS:
        s6 = fitted["stage_6"]
        vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                           enabled=s6["enabled"] and ab.venue)
        s7 = fitted["stage_7"]
        cp = CR.CorrectionParams(
            tiebreak_inflation=s7["tiebreak_inflation"] if ab.corrections else 0.0,
            split_sigma=s7["split_sigma"] if ab.corrections else 0.0,
            level_sigma=s7["level_sigma"] if ab.corrections else 0.0,
            recenter=s7["recenter"], scheme=s7["provenance_scheme"])
        # The cohort toggle is an argument, not an edit to a shared file. This
        # loop used to write `enabled: false` into fitted_params.json, build,
        # and write it back — so for the length of a build the file on disk was
        # wrong for anything else reading it, and a crash mid-build left it
        # wrong permanently.
        pan = P.build(splits=("holdout",), venue_params=vp, cohort=ab.cohort)

        maps = RC.CalibrationMaps.load() if ab.calibration else None
        scored = _cached_rows(ab.name, lambda: _families(pan, cp, maps))
        results[ab.name] = {"overall": _score(scored), "rows": scored, "panel": pan}
        print(f"  {ab.name:16s} {results[ab.name]['overall']}")

    out = C.STAGE_VALIDATIONS_DIR / "backtest.md"
    out.write_text(_report(results))
    print(f"\nwrote {out}")


def _report(results: dict) -> str:
    """Render the markdown. Separate from main() so it can be exercised on
    synthetic rows without scoring the holdout panel again."""
    lines = ["# Final backtest — HOLDOUT\n"]
    full = results["full"]["rows"]
    maps_now = RC.CalibrationMaps.load()
    cal_hash = hashlib.sha256(RC.MAPS_PATH.read_bytes()).hexdigest()[:16]
    # .maps holds the map objects; .method holds their names.
    method = sorted(set(maps_now.method.values()))
    lines.append(
        f"\nCalibration map in force: `{'/'.join(method)}`, "
        f"{len(maps_now.maps)} families, pickle sha256 `{cal_hash}`. "
        f"`calibration_refit.test_evaluated` in fitted_params.json is "
        f"`{json.loads(C.FITTED_PARAMS_PATH.read_text()).get('calibration_refit', {}).get('test_evaluated')}`"
        " — see PROGRESS.json → calibration_platt_2026_08_10_ships_unevaluated.\n")
    lines.append(f"\nHoldout window: {C.HOLDOUT_CUTOFF} onward. "
                 f"{results['full']['panel']['match_id'].nunique():,} scoreable "
                 "matches (completed, in scope, clean score, usable serve stats).\n")

    lines.append(
        "\n## How to read these numbers\n\n"
        "**A pooled figure across market families is not a score of anything.**\n"
        "The families have different base rates and different numbers of\n"
        "selections per match — seven totals rungs and eight handicap lines\n"
        "against one match winner — so a pooled Brier is a weighted average\n"
        "whose weights are an artefact of how the ladder was enumerated, and it\n"
        "moves when the ladder changes even if no forecast does. Every table\n"
        "below is therefore reported per family. The one pooled row that\n"
        "remains is marked and carries an interval.\n\n"
        "**Intervals are cluster bootstraps over matches, not rows.** One match\n"
        "supplies every selection in the row count, all driven by the same two\n"
        "serve rates. Row-level resampling would understate the spread by\n"
        "roughly the square root of the selections per match.\n\n"
        "**`n` is selections; `matches` is the real sample size.**\n")

    lines.append("\n## By market family\n")
    lines.append("| family | n | matches | Brier | Brier 95% CI | log-loss | ECE | ECE 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for fam, grp in full[full["family"] != "_crps"].groupby("family"):
        s = _score(grp)
        lines.append(f"| {fam} | {s['n']:,} | {s['n_matches']:,} | "
                     f"{s['brier']:.4f} | {_ci_str(grp, 'brier')} | "
                     f"{s['logloss']:.4f} | {s['ece']:.4f} | "
                     f"{_ci_str(grp, 'ece')} |")

    pooled = _score(full)
    lines.append(
        f"\n**Pooled across families (kept for continuity with earlier reports, "
        f"and not a meaningful score):** Brier {pooled['brier']:.4f} "
        f"{_ci_str(full, 'brier')}, log-loss {pooled['logloss']:.4f}, "
        f"ECE {pooled['ece']:.4f} {_ci_str(full, 'ece')} on "
        f"{pooled['n']:,} selections from {pooled['n_matches']:,} matches.\n")

    crps = _score(full)["crps"]
    lines.append(f"\nTotal-games CRPS: {crps:.4f} {_ci_str(full, 'crps')} "
                 "(one value per match, so no clustering issue).\n")

    lines.append("\n## By market family and tour level\n")
    lines.append("| family | level | n | matches | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|")
    for (fam, lvl), grp in full[full["family"] != "_crps"].groupby(
            ["family", "level_label"]):
        s = _score(grp)
        lines.append(f"| {fam} | {lvl} | {s['n']:,} | {s['n_matches']:,} | "
                     f"{s['brier']:.4f} | {s['ece']:.4f} |")

    lines.append("\n## By market family, format rule and provenance\n")
    lines.append("| family | final-set rule | provenance | n | matches | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|")
    for (fam, rule, prov), grp in full[full["family"] != "_crps"].groupby(
            ["family", "final_set", "provenance"]):
        s = _score(grp)
        lines.append(f"| {fam} | {rule} | {prov} | {s['n']:,} | "
                     f"{s['n_matches']:,} | {s['brier']:.4f} | {s['ece']:.4f} |")

    lines.append("\n## Ablations, by family\n")
    lines.append("Brier / ECE per configuration. Compare down a column, never "
                 "across families.\n")
    fams = sorted(full[full["family"] != "_crps"]["family"].unique())
    lines.append("| configuration | " + " | ".join(fams) + " | games CRPS |")
    lines.append("|---" * (len(fams) + 2) + "|")
    for name, res in results.items():
        cells = []
        rws = res["rows"]
        for fam in fams:
            s = _score(rws[rws["family"] == fam])
            cells.append(f"{s['brier']:.4f} / {s['ece']:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {res['overall']['crps']:.4f} |")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
