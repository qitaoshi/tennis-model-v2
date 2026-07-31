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
        # Panel rows are oriented winner-first, so scoring only that
        # orientation would make every outcome a 1 by construction and turn
        # calibration error into (1 - mean prediction). Emit both sides.
        rows.append({**common, "family": "match_winner", "p": p, "y": 1.0})
        rows.append({**common, "family": "match_winner", "p": 1.0 - p, "y": 0.0})

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
            # Same orientation trap as the match winner: the row knows who won
            # and the model does not, so score both perspectives.
            rows.append({**common, "family": "handicap",
                         "p": sum(q for dd, q in diffs.items() if dd > h),
                         "y": float(margin > h)})
            rows.append({**common, "family": "handicap",
                         "p": sum(q for dd, q in diffs.items() if -dd > h),
                         "y": float(-margin > h)})

        rows.append({**common, "family": "_crps",
                     "p": P.crps_games(pmf, int(r.total_games)), "y": np.nan})
    return pd.DataFrame(rows)


def _score(df: pd.DataFrame) -> dict:
    scored = df[df["family"] != "_crps"]
    crps = df[df["family"] == "_crps"]["p"]
    return {"n": len(scored),
            "brier": RC.brier(scored["p"].to_numpy(), scored["y"].to_numpy()),
            "logloss": RC.log_loss(scored["p"].to_numpy(), scored["y"].to_numpy()),
            "ece": RC.calibration_error(scored["p"].to_numpy(), scored["y"].to_numpy()),
            "crps": float(crps.mean()) if len(crps) else float("nan")}


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

    lines = ["# Final backtest — HOLDOUT\n"]
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
        saved = fitted["stage_5"]["enabled"]
        fitted["stage_5"]["enabled"] = saved and ab.cohort
        C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
        try:
            pan = P.build(splits=("holdout",), venue_params=vp)
        finally:
            fitted["stage_5"]["enabled"] = saved
            C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))

        maps = RC.CalibrationMaps.load() if ab.calibration else None
        scored = _families(pan, cp, maps)
        results[ab.name] = {"overall": _score(scored), "rows": scored, "panel": pan}
        print(f"  {ab.name:16s} {results[ab.name]['overall']}")

    full = results["full"]["rows"]
    lines.append(f"\nHoldout window: {C.HOLDOUT_CUTOFF} onward. "
                 f"{results['full']['panel']['match_id'].nunique():,} scoreable "
                 "matches (completed, in scope, clean score, usable serve stats).\n")

    lines.append("\n## By market family\n")
    lines.append("| family | n | Brier | log-loss | ECE |")
    lines.append("|---|---|---|---|---|")
    for fam, grp in full[full["family"] != "_crps"].groupby("family"):
        s = _score(grp)
        lines.append(f"| {fam} | {s['n']:,} | {s['brier']:.4f} | "
                     f"{s['logloss']:.4f} | {s['ece']:.4f} |")

    lines.append("\n## By tour level\n")
    lines.append("| level | n | Brier | log-loss | ECE |")
    lines.append("|---|---|---|---|---|")
    for lvl, grp in full[full["family"] != "_crps"].groupby("level_label"):
        s = _score(grp)
        lines.append(f"| {lvl} | {s['n']:,} | {s['brier']:.4f} | "
                     f"{s['logloss']:.4f} | {s['ece']:.4f} |")

    lines.append("\n## By format rule and provenance\n")
    lines.append("| final-set rule | provenance | n | Brier | ECE |")
    lines.append("|---|---|---|---|---|")
    for (rule, prov), grp in full[full["family"] != "_crps"].groupby(
            ["final_set", "provenance"]):
        s = _score(grp)
        lines.append(f"| {rule} | {prov} | {s['n']:,} | {s['brier']:.4f} | "
                     f"{s['ece']:.4f} |")

    lines.append("\n## Ablations\n")
    lines.append("| configuration | Brier | log-loss | ECE | games CRPS |")
    lines.append("|---|---|---|---|---|")
    for name, res in results.items():
        s = res["overall"]
        lines.append(f"| {name} | {s['brier']:.4f} | {s['logloss']:.4f} | "
                     f"{s['ece']:.4f} | {s['crps']:.4f} |")

    out = C.STAGE_VALIDATIONS_DIR / "backtest.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
