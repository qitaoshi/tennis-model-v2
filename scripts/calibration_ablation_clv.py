"""Is the shipped match_winner isotonic map helping or hurting CLV/ROI?

Measurement only: no parameter is fitted or selected. Same HOLDOUT odds join
as scripts/clv_backtest.py, but scores the model's match-winner probability
both WITH and WITHOUT the Stage 8 calibration map applied, side by side.

Why this exists: reports/stage_validations/stage_8.md shows the match_winner
map made Brier worse on its own pre-cutoff TEST window (0.23981 -> 0.24013,
improved=False) and passed only on ECE; it shipped anyway riding on the
totals_under_high accept decision. HOLDOUT ECE for match_winner (0.0693) is
4x worse than the post-calibration TEST figure (0.0169), consistent with a
map that doesn't generalize. clv_backtest.py's CLV+/ROI- result was measured
entirely through this map. This script isolates whether the map is the cause.

Run with ``python3 -m scripts.calibration_ablation_clv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from scripts import panel as P
from scripts.clv_backtest import (
    BOOTSTRAP_N,
    _bootstrap_mean,
    _name_key,
    _read_odds,
    _text_key,
)

REPORT_PATH = C.REPORTS_DIR / "calibration_ablation_clv.md"


def _model_predictions() -> pd.DataFrame:
    panel = P.build(splits=("holdout",))
    holdout = pd.read_parquet(C.PROCESSED_DIR / "matches_with_holdout.parquet")
    meta = holdout.set_index("match_id")
    panel = panel[meta.loc[panel["match_id"], "tour"].eq("atp").to_numpy()].copy()
    panel["year"] = meta.loc[panel["match_id"], "season_file"].to_numpy()
    panel["tournament_key"] = meta.loc[panel["match_id"], "tourney_name"].map(_text_key).to_numpy()
    panel["surface_key"] = meta.loc[panel["match_id"], "surface"].map(_text_key).to_numpy()
    panel["winner_key"] = meta.loc[panel["match_id"], "winner_name"].map(_name_key).to_numpy()
    panel["loser_key"] = meta.loc[panel["match_id"], "loser_name"].map(_name_key).to_numpy()

    fitted = P.load_fitted()
    s7 = fitted["stage_7"]
    params = CR.CorrectionParams(
        tiebreak_inflation=s7["tiebreak_inflation"],
        split_sigma=s7["split_sigma"],
        level_sigma=s7["level_sigma"],
        recenter=s7["recenter"],
        scheme=s7["provenance_scheme"],
    )
    maps = RC.CalibrationMaps.load()
    rows = []
    from model.rules import FormatSpec

    for row in panel.itertuples(index=False):
        key = row.spec_key
        spec = FormatSpec(
            best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
            tb_to=key[3], final_set=key[4], final_tb_at=key[5],
            final_tb_to=key[6], provenance=row.format_provenance, source="clv_ablation",
        )
        dist = CR.corrected_distribution(
            round(float(row.pa), 3), round(float(row.pb), 3), spec, params)
        # Side selection fixed by the RAW model, so both variants price the
        # same bet; only the probability estimate (and therefore edge, CLV
        # and stake sizing) differs between them.
        model_side_winner = dist.p_a >= 0.5
        p_raw = dist.p_a if model_side_winner else 1 - dist.p_a
        p_cal = float(maps.apply("match_winner", dist.p_a))
        p_cal = p_cal if model_side_winner else 1 - p_cal
        rows.append({
            "match_id": row.match_id,
            "year": row.year,
            "tournament_key": row.tournament_key,
            "surface_key": row.surface_key,
            "winner_key": row.winner_key,
            "loser_key": row.loser_key,
            "model_edge_side": "winner" if model_side_winner else "loser",
            "model_p_raw": p_raw,
            "model_p_calibrated": p_cal,
        })
    return pd.DataFrame(rows)


def _score_variant(joined: pd.DataFrame, p_col: str) -> dict:
    edge = joined[p_col] - joined["market_p"]
    clv = joined[p_col] / joined["market_p"] - 1.0
    mean, lo, hi = _bootstrap_mean(clv.to_numpy())
    result = {
        "n": len(joined), "clv_mean": mean, "clv_lo": lo, "clv_hi": hi,
        "positive_clv_share": float((clv > 0).mean()),
        "mean_edge": float(edge.mean()),
    }
    positive = joined[edge > 0]
    if lo > 0 and len(positive):
        profit = np.where(positive["won"].to_numpy() > 0,
                          positive["selected_odds"].to_numpy() - 1.0, -1.0)
        roi, roi_lo, roi_hi = _bootstrap_mean(profit)
        result.update({"n_positive_edge": len(positive), "roi": roi,
                       "roi_lo": roi_lo, "roi_hi": roi_hi})
    else:
        result.update({"n_positive_edge": len(positive), "roi": float("nan"),
                       "roi_lo": float("nan"), "roi_hi": float("nan")})
    return result


def main() -> None:
    odds = _read_odds()
    model = _model_predictions()
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

    raw = _score_variant(joined, "model_p_raw")
    cal = _score_variant(joined, "model_p_calibrated")

    def fmt_roi(r: dict) -> str:
        if np.isnan(r["roi"]):
            return "CLV CI includes zero; ROI sim skipped"
        return f"{r['roi']:+.2%} (CI {r['roi_lo']:+.2%} to {r['roi_hi']:+.2%}), n={r['n_positive_edge']:,}"

    report = f"""# Calibration ablation — match_winner CLV/ROI, with vs without the Stage 8 map

Measurement only: no parameter fitted or selected. Same HOLDOUT odds join as
`clv_backtest.py` ({len(joined):,} matched rows), scored twice: once through
the shipped isotonic map (`model_p_calibrated`), once with the model's raw
Elo/serve-blend probability (`model_p_raw`). Bet side is fixed by the raw
model in both variants, so this isolates the calibration map's effect on the
probability estimate from its effect on bet selection.

## Why this was run

`reports/stage_validations/stage_8.md` shows the match_winner isotonic map
made Brier *worse* on its own pre-cutoff TEST window (0.23981 -> 0.24013,
`improved: False`); it passed only on ECE, and shipped riding on the human
accept decision for `totals_under_high`. HOLDOUT ECE for match_winner
(0.0693, `reports/stage_validations/backtest.md`) is 4x worse than the
0.0169 the map showed right after fitting, consistent with a map that
doesn't generalize past the window it was tuned on.

## Result

| variant | n | mean CLV | CLV 95% CI | positive CLV | mean edge | flat-stake ROI |
|---|---:|---:|---|---:|---:|---|
| calibrated (shipped) | {cal['n']:,} | {cal['clv_mean']:+.2%} | {cal['clv_lo']:+.2%} to {cal['clv_hi']:+.2%} | {cal['positive_clv_share']:.1%} | {cal['mean_edge']:+.2%} | {fmt_roi(cal)} |
| raw (uncalibrated) | {raw['n']:,} | {raw['clv_mean']:+.2%} | {raw['clv_lo']:+.2%} to {raw['clv_hi']:+.2%} | {raw['positive_clv_share']:.1%} | {raw['mean_edge']:+.2%} | {fmt_roi(raw)} |

## Reading this

If raw ROI's interval clears zero (or is materially less negative than
calibrated), the Stage 8 match_winner map is actively hurting, not helping,
and the fix is to drop it from `fitted_params.json["stage_8"]["families"]`
rather than fit a new one. If both are similarly negative, the map is not
the explanation and the CLV+/ROI- gap has some other cause (staking,
selection threshold, or a real absence of edge net of vig).
"""
    REPORT_PATH.write_text(report)
    print(report)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
