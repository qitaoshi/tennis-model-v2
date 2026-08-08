"""Measure closing-line value against the frozen model on ATP HOLDOUT matches.

This is evaluation only: no parameters are fitted or selected here.
Run with ``python3 -m scripts.clv_backtest``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from scripts import panel as P


RAW_ODDS_DIR = C.PROCESSED_DIR / "raw"
REPORT_PATH = C.REPORTS_DIR / "clv_backtest.md"
BOOTSTRAP_N = 10_000


def _name_key(name: object, odds_style: bool = False) -> str:
    """Key ``First Last`` and ``Last F.`` forms consistently."""
    if pd.isna(name):
        return ""
    tokens = re.findall(r"[a-z]+", str(name).lower().replace("'", ""))
    if not tokens:
        return ""
    if odds_style:
        # Trailing single letters are given-name initials, and there may be
        # more than one: "Etcheverry T. M.", "Struff J-L.", "Wolf J.J.".
        surname_tokens, initials = list(tokens), []
        while len(surname_tokens) > 1 and len(surname_tokens[-1]) == 1:
            initials.insert(0, surname_tokens.pop())
        initial = initials[0] if initials else surname_tokens[-1][0]
        surname = "".join(surname_tokens)
    else:
        initial, surname = tokens[0][0], "".join(tokens[1:])
    return initial + surname


def _decimal_probability(value: object) -> float:
    value = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(value) or value <= 1:
        return np.nan
    return 1.0 / float(value)


def _read_odds() -> pd.DataFrame:
    """Read only holdout-era sheets; old 2010 binary files are irrelevant."""
    frames = []
    for path in sorted(RAW_ODDS_DIR.glob("202[4-6].xlsx")):
        frame = pd.read_excel(path, engine="openpyxl")
        frame["source_file"] = path.name
        frames.append(frame)
    odds = pd.concat(frames, ignore_index=True)
    odds["date"] = pd.to_datetime(odds["Date"], errors="coerce").dt.date
    odds = odds[odds["date"] >= C.HOLDOUT_CUTOFF].copy()

    def side_probability(
        row: pd.Series, primary: str, fallback: str
    ) -> tuple[float, str, float]:
        for col, source in ((primary, "Pinnacle"), (fallback, "Bet365")):
            odds = pd.to_numeric(row[col], errors="coerce")
            p = _decimal_probability(odds)
            if np.isfinite(p):
                return p, source, float(odds)
        return np.nan, "missing", np.nan

    odds[["p_w_raw", "source_w", "odds_w"]] = odds.apply(
        lambda r: pd.Series(side_probability(r, "PSW", "B365W")), axis=1)
    odds[["p_l_raw", "source_l", "odds_l"]] = odds.apply(
        lambda r: pd.Series(side_probability(r, "PSL", "B365L")), axis=1)
    odds = odds.dropna(subset=["p_w_raw", "p_l_raw"]).copy()
    total = odds["p_w_raw"] + odds["p_l_raw"]
    odds["p_w_market"] = odds["p_w_raw"] / total
    odds["p_l_market"] = odds["p_l_raw"] / total
    odds["year"] = pd.to_datetime(odds["date"]).dt.year
    odds["tournament_key"] = odds["Tournament"].map(_text_key)
    odds["surface_key"] = odds["Surface"].map(_text_key)
    odds["winner_key"] = odds["Winner"].map(lambda x: _name_key(x, True))
    odds["loser_key"] = odds["Loser"].map(lambda x: _name_key(x, True))
    return odds[
        ["year", "tournament_key", "surface_key", "winner_key", "loser_key",
         "p_w_market", "p_l_market", "source_w", "source_l",
         "odds_w", "odds_l"]
    ]


def _text_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower()) if not pd.isna(value) else ""


def _bootstrap_mean(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(C.MC_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_N, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


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
            final_tb_to=key[6], provenance=row.format_provenance, source="clv",
        )
        dist = CR.corrected_distribution(
            round(float(row.pa), 3), round(float(row.pb), 3), spec, params)
        p_winner = float(maps.apply("match_winner", dist.p_a))
        model_side_winner = p_winner >= 0.5
        rows.append({
            "match_id": row.match_id,
            "year": row.year,
            "tournament_key": row.tournament_key,
            "surface_key": row.surface_key,
            "winner_key": row.winner_key,
            "loser_key": row.loser_key,
            "model_p": p_winner if model_side_winner else 1 - p_winner,
            "model_edge_side": "winner" if model_side_winner else "loser",
        })
    return pd.DataFrame(rows)


def main() -> None:
    odds = _read_odds()
    model = _model_predictions()
    # TML records tournament start date, while bookmaker rows use each match's
    # played date and different tournament labels. Year/surface/player-pair is
    # the common key; the resulting coverage is reported explicitly.
    join_keys = ["year", "surface_key"]
    winner_join = join_keys + ["winner_key", "loser_key"]
    joined = model.merge(
        odds, left_on=winner_join, right_on=winner_join, how="left",
        suffixes=("", "_odds"),
    )
    # Model rows are winner-first; select the corresponding bookmaker side.
    joined["market_p"] = np.where(
        joined["model_edge_side"].eq("winner"),
        joined["p_w_market"], joined["p_l_market"])
    joined["selected_odds"] = np.where(
        joined["model_edge_side"].eq("winner"),
        joined["odds_w"], joined["odds_l"])
    joined["outcome"] = np.where(
        joined["model_edge_side"].eq("winner"), 1.0, 0.0)
    joined = joined.dropna(subset=["market_p"]).copy()
    joined["edge"] = joined["model_p"] - joined["market_p"]
    joined["clv"] = joined["model_p"] / joined["market_p"] - 1.0
    joined["won"] = (joined["outcome"] == 1.0).astype(float)

    n_model = len(model)
    n_odds = len(odds)
    holdout = pd.read_parquet(C.PROCESSED_DIR / "matches_with_holdout.parquet")
    n_atp_holdout = int(((holdout["split"] == "holdout")
                         & holdout["tour"].eq("atp")).sum())
    n_matched = len(joined)
    if not n_matched:
        raise SystemExit("No holdout matches joined to odds; inspect name/tournament keys.")
    mean, lo, hi = _bootstrap_mean(joined["clv"].to_numpy())

    joined["bucket"] = pd.qcut(
        joined["edge"], q=5, labels=["Q1 (smallest)", "Q2", "Q3", "Q4", "Q5 (largest)"],
        duplicates="drop",
    )
    bucket_lines = []
    for bucket, group in joined.groupby("bucket", observed=True):
        bucket_lines.append(
            f"| {bucket} | {len(group):,} | {group['edge'].mean():+.4f} | "
            f"{group['clv'].mean():+.4f} | {(group['clv'] > 0).mean():.1%} |"
        )

    roi_section = (
        "The CLV confidence interval excludes zero, so a flat-stake simulation "
        "was run on model-positive-edge selections.\n"
    )
    positive = joined[joined["edge"] > 0]
    if lo > 0 and len(positive):
        profit = np.where(positive["won"].to_numpy() > 0,
                          positive["selected_odds"].to_numpy() - 1.0, -1.0)
        roi, roi_lo, roi_hi = _bootstrap_mean(profit)
        roi_section += (
            f"\nThose {len(positive):,} selections returned {roi:+.2%} ROI "
            f"(bootstrap 95% CI {roi_lo:+.2%} to {roi_hi:+.2%}).\n"
        )
    else:
        roi_section = (
            "The CLV confidence interval includes zero, so the requested flat-stake "
            "ROI simulation was skipped; it would only simulate noise.\n"
        )

    verdict = "yes" if lo > 0 else "no"
    source_counts = joined[["source_w", "source_l"]].stack().value_counts()
    report = f"""# Closing-line value backtest — ATP match-winner

The frozen model's higher-probability side had mean CLV of **{mean:+.2%}**
(bootstrap 95% CI **{lo:+.2%} to {hi:+.2%}**). This means the model's selected
side was priced more generously than the de-vigged closing market by that
amount on average; the interval is the uncertainty around the estimate.

## Coverage and method

- The holdout model produced {n_model:,} supported ATP rows; the
  odds source contributed {n_odds:,} ATP rows dated {C.HOLDOUT_CUTOFF} onward.
- The holdout contains {n_atp_holdout:,} ATP matches; {n_matched:,} joined
  ({n_matched / n_atp_holdout:.1%} of all ATP holdout matches, {n_matched / n_model:.1%}
  of supported model rows). Unmatched rows were counted, not silently treated as zero CLV.
- Odds were decimal (for example, 1.72), using Pinnacle PSW/PSL where
  available and Bet365 B365W/B365L per-side as fallback. Implied probabilities
  were normalized across both sides to remove vig.
- CLV is `model probability / market probability - 1`, evaluated on the side
  the model preferred. No model parameter was fitted or selected.
- ROI uses the selected side's original decimal bookmaker odds: a win pays
  `decimal odds - 1` per unit stake; a loss costs one unit.
- Odds source sides: Pinnacle {int(source_counts.get("Pinnacle", 0)):,},
  Bet365 fallback {int(source_counts.get("Bet365", 0)):,}.

## CLV distribution

- Positive CLV: {(joined["clv"] > 0).mean():.1%} of matches
- Median CLV: {joined["clv"].median():+.2%}
- Mean model edge (`model_p - market_p`): {joined["edge"].mean():+.2%}

| model edge bucket | matches | mean edge | mean CLV | positive CLV |
|---|---:|---:|---:|---:|
{chr(10).join(bucket_lines)}

## Optional flat-stake simulation

{roi_section}

## Verdict

**Demonstrated pricing edge on ATP match-winner: {verdict.upper()}.** The
CLV interval {"is entirely above zero" if verdict == "yes" else "straddles or reaches zero"}.
The realized flat-stake ROI is reported separately and is not treated as
positive evidence unless its own interval excludes zero. This is measurement
only; the result does not authorize retuning the model.
"""
    REPORT_PATH.write_text(report)
    print(f"matched {n_matched:,}/{n_model:,}; mean CLV {mean:+.2%} "
          f"(95% CI {lo:+.2%}, {hi:+.2%})")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
