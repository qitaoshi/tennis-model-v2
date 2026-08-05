"""Is the match_winner edge concentrated in a segment, or absent everywhere?

Measurement only: no parameter fitted or selected. Same HOLDOUT odds join as
clv_backtest.py, raw (uncalibrated) model probability only — the calibration
ablation (reports/calibration_ablation_clv.md) showed the isotonic map isn't
the cause of the CLV+/ROI- gap, so this asks the next question: does the raw
model have real edge in some slice (level, surface, favorite vs underdog)
that a pooled average would wash out?

Run with ``python3 -m scripts.segment_edge_holdout``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import constants as C
from scripts.calibration_ablation_clv import _model_predictions
from scripts.clv_backtest import BOOTSTRAP_N, _bootstrap_mean, _read_odds

REPORT_PATH = C.REPORTS_DIR / "segment_edge_holdout.md"


def _segment_stats(group: pd.DataFrame) -> dict:
    edge = group["model_p_raw"] - group["market_p"]
    clv = group["model_p_raw"] / group["market_p"] - 1.0
    mean, lo, hi = _bootstrap_mean(clv.to_numpy())
    positive = group[edge > 0]
    if lo > 0 and len(positive) >= 20:
        profit = np.where(positive["won"].to_numpy() > 0,
                          positive["selected_odds"].to_numpy() - 1.0, -1.0)
        roi, roi_lo, roi_hi = _bootstrap_mean(profit)
        roi_str = f"{roi:+.2%} (CI {roi_lo:+.2%} to {roi_hi:+.2%})"
    else:
        roi_str = "n/a"
    return {"n": len(group), "mean_edge": float(edge.mean()),
            "clv": mean, "clv_lo": lo, "clv_hi": hi,
            "n_positive_edge": len(positive), "roi": roi_str}


def main() -> None:
    odds = _read_odds()
    model = _model_predictions()
    holdout = pd.read_parquet(C.PROCESSED_DIR / "matches_with_holdout.parquet")
    meta = holdout.set_index("match_id")
    model["level_label"] = meta.loc[model["match_id"], "level_label"].to_numpy()
    model["surface"] = meta.loc[model["match_id"], "surface"].to_numpy()

    join_keys = ["year", "surface_key"]
    winner_join = join_keys + ["winner_key", "loser_key"]
    joined = model.merge(odds, left_on=winner_join, right_on=winner_join,
                         how="left", suffixes=("", "_odds"))
    joined["market_p"] = np.where(
        joined["model_edge_side"].eq("winner"),
        joined["p_w_market"], joined["p_l_market"])
    joined["selected_odds"] = np.where(
        joined["model_edge_side"].eq("winner"),
        joined["odds_w"], joined["odds_l"])
    joined["won"] = (joined["model_edge_side"].eq("winner")).astype(float)
    joined = joined.dropna(subset=["market_p"]).copy()
    if not len(joined):
        raise SystemExit("No holdout matches joined to odds; inspect name/tournament keys.")

    joined["market_favorite"] = np.where(
        joined["model_edge_side"].eq("winner"),
        joined["p_w_market"] >= joined["p_l_market"],
        joined["p_l_market"] >= joined["p_w_market"])
    joined["agreement"] = np.where(
        joined["market_favorite"], "model picks market favorite", "model picks market underdog")

    def table(col: str) -> list[str]:
        rows = ["| segment | n | mean edge | mean CLV | CLV 95% CI | ROI (n edge>0) |",
                "|---|---:|---:|---:|---|---|"]
        for val, grp in sorted(joined.groupby(col), key=lambda kv: -len(kv[1])):
            s = _segment_stats(grp)
            rows.append(f"| {val} | {s['n']:,} | {s['mean_edge']:+.2%} | "
                        f"{s['clv']:+.2%} | {s['clv_lo']:+.2%} to {s['clv_hi']:+.2%} | "
                        f"{s['roi']} (n={s['n_positive_edge']:,}) |")
        return rows

    overall = _segment_stats(joined)
    report = f"""# Segment edge check — match_winner, raw probability, HOLDOUT

Measurement only: no parameter fitted or selected. Uses the raw
(uncalibrated) model probability throughout, since the calibration ablation
(`reports/calibration_ablation_clv.md`) already showed the isotonic map
doesn't explain the CLV+/ROI- gap. {len(joined):,} matched rows.

Overall: mean edge {overall['mean_edge']:+.2%}, mean CLV {overall['clv']:+.2%}
(CI {overall['clv_lo']:+.2%} to {overall['clv_hi']:+.2%}), ROI {overall['roi']}.

## By tour level

{chr(10).join(table('level_label'))}

## By surface

{chr(10).join(table('surface'))}

## By market agreement

{chr(10).join(table('agreement'))}

## Reading this

Any segment whose ROI interval clears zero has real, isolable edge even if
the pooled average doesn't. A segment with a positive mean edge but n_edge>0
too small for a ROI estimate ("n/a") is a candidate to revisit once more
holdout matches accrue, not a demonstrated edge yet.
"""
    REPORT_PATH.write_text(report)
    print(report)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
