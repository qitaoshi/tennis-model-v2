"""Is the backtest's totals ECE flattered by the lines it chooses to score?

This check exists because the same question had a bad answer for handicaps.
`backtest.py` scores handicaps at fixed margins (-6.5, -2.5, 1.5, 5.5); for a
winner-oriented row the first two are nearly certain, so half the scored
selections are freebies and the pooled ECE comes out at 0.0098 against 0.0624
on the lines bookmakers actually quote.

Totals are scored differently — at offsets from each match's OWN median
(`LINE_OFFSETS`), so the ladder is at least centred on the match. But an
offset of -6.5 or +5.5 games is still far into a tail, and if those rungs are
near-certain they dilute the pooled number the same way.

Totals ECE 0.0078 is the headline "this model is good" figure in the holdout
report, so it is worth knowing whether it survives being measured only where
a real line exists.

Measured on TUNE. Nothing is fitted. TEST and HOLDOUT are not touched.

Run:  python -m scripts.check_totals_ladder
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from model import venue as V
from model.rules import FormatSpec
from scripts import panel as P
from scripts.fit_stage8 import LINE_OFFSETS, _median_of

REPORT = C.REPORTS_DIR / "totals_ladder_check.md"


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s6, s7 = fitted["stage_6"], fitted["stage_7"]
    vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                       enabled=s6["enabled"])
    cp = CR.CorrectionParams(
        tiebreak_inflation=s7["tiebreak_inflation"], split_sigma=s7["split_sigma"],
        level_sigma=s7["level_sigma"], recenter=s7["recenter"],
        scheme=s7["provenance_scheme"])

    pan = P.build(splits=("tune",), venue_params=vp)
    pan = pan.sample(min(len(pan), 6_000), random_state=C.MC_SEED)
    print(f"TUNE panel: {len(pan):,} matches")

    rows = []
    medians = []
    for r in pan.itertuples(index=False):
        k = r.spec_key
        spec = FormatSpec(best_of=k[0], games_to_win_set=k[1], tb_at=k[2],
                          tb_to=k[3], final_set=k[4], final_tb_at=k[5],
                          final_tb_to=k[6], provenance="documented", source="chk")
        d = CR.corrected_distribution(round(float(r.pa), 3),
                                      round(float(r.pb), 3), spec, cp)
        pmf = d.total_games_pmf()
        med = _median_of(pmf)
        medians.append(med)
        for off in LINE_OFFSETS:
            line = med + off
            under = sum(p for g, p in pmf.items() if g < line)
            rows.append({"offset": off, "p": under,
                         "y": float(r.total_games < line)})
    df = pd.DataFrame(rows)

    per = []
    for off, g in df.groupby("offset"):
        per.append({"offset": off, "n": len(g), "mean_p": g["p"].mean(),
                    "base_rate": g["y"].mean(),
                    "ece": RC.calibration_error(g["p"].to_numpy(),
                                                g["y"].to_numpy()),
                    "brier": RC.brier(g["p"].to_numpy(), g["y"].to_numpy())})
    per = pd.DataFrame(per)
    print("\nper-offset (the backtest pools all of these into one number):")
    print(per.to_string(index=False))

    pooled = RC.calibration_error(df["p"].to_numpy(), df["y"].to_numpy())
    print(f"\npooled ECE over the whole ladder: {pooled:.5f}")

    # What offsets do bookmakers actually quote, relative to the model median?
    odds = pd.read_parquet(C.PROCESSED_DIR / "odds_portal_flat_2024.parquet")
    odds = odds[odds["market"] == "total_games"].copy()
    odds["date"] = pd.to_datetime(odds["date"]).dt.date
    odds = odds[(odds["date"] >= C.TUNE_START) & (odds["date"] <= C.TUNE_END)]
    med_all = float(np.median(medians))
    q = odds["line"].astype(float)
    rel = q - med_all
    print(f"\nmarket total-games lines: median {q.median():.1f}, "
          f"p5 {q.quantile(0.05):.1f}, p95 {q.quantile(0.95):.1f}")
    print(f"model median total (typical): {med_all:.1f}")
    print(f"market line minus typical model median: p5 {rel.quantile(0.05):+.1f}, "
          f"p50 {rel.quantile(0.5):+.1f}, p95 {rel.quantile(0.95):+.1f}")

    # Restrict to the rungs a real line plausibly lands on.
    lo, hi = float(rel.quantile(0.05)), float(rel.quantile(0.95))
    keep = df[df["offset"].between(lo, hi)]
    realistic = (RC.calibration_error(keep["p"].to_numpy(), keep["y"].to_numpy())
                 if len(keep) else float("nan"))
    inner = df[df["offset"].abs() <= 2.5]
    inner_ece = RC.calibration_error(inner["p"].to_numpy(), inner["y"].to_numpy())
    print(f"\nECE on offsets inside the market's own p5-p95 "
          f"({lo:+.1f} to {hi:+.1f}): {realistic:.5f}  (n={len(keep):,})")
    print(f"ECE on the central rungs only (|offset| <= 2.5): {inner_ece:.5f} "
          f"(n={len(inner):,})")
    inflated = bool(np.isfinite(realistic) and realistic > pooled * 1.5)
    print(f"\nPooled number materially flattered by the tails: {inflated}")

    lines = [
        "# Is the backtest's totals ECE flattered by its line choices?\n",
        "The same question had a bad answer for handicaps: `backtest.py` "
        "scores those at fixed margins where half the selections are nearly "
        "certain, giving ECE 0.0098 against 0.0624 on real market lines. "
        "Totals ECE 0.0078 is the headline 'this model is good' figure in the "
        "holdout report, so it is worth the same scrutiny.\n",
        "\nMeasured on TUNE. Nothing fitted; TEST and HOLDOUT untouched.\n",
        "\n## Per rung of the ladder\n",
        "`backtest.py` pools all seven of these into a single totals number. "
        "A rung whose base rate is near 0 or 1 is close to a free bet and "
        "contributes almost nothing to calibration error.\n",
        per.to_markdown(index=False),
        f"\n\nPooled ECE over the whole ladder: **{pooled:.5f}**.\n",
        "\n## What lines actually exist\n",
        f"\nMarket total-games lines on TUNE: median {q.median():.1f}, "
        f"p5 {q.quantile(0.05):.1f}, p95 {q.quantile(0.95):.1f}. Typical model "
        f"median total {med_all:.1f}, so real lines sit roughly "
        f"{lo:+.1f} to {hi:+.1f} games from it.\n",
        f"\n| scope | ECE | n |\n|---|---:|---:|\n"
        f"| whole ladder (what the backtest reports) | {pooled:.5f} "
        f"| {len(df):,} |\n"
        f"| offsets inside the market's p5-p95 | {realistic:.5f} "
        f"| {len(keep):,} |\n"
        f"| central rungs only, abs(offset) <= 2.5 | {inner_ece:.5f} "
        f"| {len(inner):,} |\n",
        f"\n**Materially flattered by the tails: {inflated}.**\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
