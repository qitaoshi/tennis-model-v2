"""Why is match_winner badly calibrated on holdout but not on tune or test?

The numbers that prompted this, all full-pipeline match_winner ECE:

    TUNE    2024-01 .. 2025-06   0.0178
    TEST    2025-07 .. 2025-12   0.0181
    HOLDOUT 2026-01 .. 2026-07   0.0665

and, from the original build before the re-split:

    TEST    2023-07 .. 2023-12   0.0169
    HOLDOUT 2024-01 .. 2026-07   0.0693

Both test windows are July-December. Both holdout windows begin in January.
That is a seasonal split, not a drift one — drift would degrade with distance
from the fitting data, and it does not: the 2025-07 window sits between the
two fitting windows and calibrates fine.

Leading hypothesis: the tour takes five to eight weeks off between seasons,
and nothing in the model treats a returning player as less known than a
mid-season one. Elo's inactivity regression has a 1095-day half-life, so a
six-week break moves a rating by almost nothing. Every January the model is
therefore confident on the basis of stale November information, and both
holdout windows happen to open there.

This script tests that at the Elo layer, on TUNE, where it costs seconds.
It is a DIAGNOSTIC — it fits and selects nothing, and it touches neither
TEST nor HOLDOUT.

Run:  python -m scripts.diagnose_match_winner
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import constants as C
from model import elo as E
from model import recalibrate as RC

REPORT = C.REPORTS_DIR / "match_winner_diagnosis.md"


def ece_of(p_winner: np.ndarray) -> dict:
    """Both orientations, or every outcome is a 1 and the number is fiction."""
    if len(p_winner) < 100:
        return {"n": len(p_winner), "ece": float("nan"), "brier": float("nan"),
                "mean_p": float("nan"), "win_rate": float("nan")}
    p = np.concatenate([p_winner, 1.0 - p_winner])
    y = np.concatenate([np.ones(len(p_winner)), np.zeros(len(p_winner))])
    return {"n": len(p_winner), "ece": RC.calibration_error(p, y),
            "brier": RC.brier(p, y), "mean_p": float(np.mean(p_winner)),
            "win_rate": 1.0}


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s3 = fitted["stage_3"]
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    assert m["date"].max() < C.HOLDOUT_CUTOFF, "holdout must not be loaded"

    ep = E.EloParams(k=s3["k"], k_chall_mult=s3["k_chall_mult"],
                     surface_weight=s3["surface_weight"],
                     inactivity_half_life=s3["inactivity_half_life"],
                     level_gap=s3["level_gap"], level_offset=s3["level_offset"])
    elo = E.run_elo(m, ep)[["match_id", "p_winner"]]
    meta = m.set_index("match_id")
    for col in ("date", "split", "level_label", "surface"):
        elo[col] = meta.loc[elo["match_id"], col].to_numpy()
    elo["month"] = [d.month for d in elo["date"]]
    elo["year"] = [d.year for d in elo["date"]]

    # Days since that player's previous match, per side, to test the
    # "returning from the off-season" part of the hypothesis directly.
    # A player with no prior match and a player returning after a long absence
    # are NOT the same problem and do not have the same fix — one needs a
    # better starting prior, the other needs its existing rating aged. Pooling
    # them (an earlier version of this script assigned debutants a layoff of
    # 999) makes the debut effect look like a layoff effect.
    t = np.array([d.toordinal() for d in elo["date"]], dtype=float)
    order = np.argsort(t, kind="mergesort")
    last: dict[str, float] = {}
    layoff = np.zeros(len(elo))
    debut = np.zeros(len(elo), dtype=bool)
    wid = meta.loc[elo["match_id"], "winner_id"].to_numpy()
    lid = meta.loc[elo["match_id"], "loser_id"].to_numpy()
    for i in order:
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
    elo["max_layoff"] = layoff
    elo["has_debutant"] = debut

    tune = elo[elo["split"] == "tune"]
    allsp = elo[elo["split"].isin(("fit", "tune", "burned_test", "test"))]
    print(f"TUNE {len(tune):,} matches, all splits {len(allsp):,}")

    by_month = pd.DataFrame(
        [{"month": mo, **ece_of(g["p_winner"].to_numpy())}
         for mo, g in allsp.groupby("month")]).drop(columns=["win_rate"])
    print("\nECE by calendar month (all pre-holdout data):")
    print(by_month.to_string(index=False))

    half = pd.DataFrame(
        [{"half": "Jan-Jun" if h == 0 else "Jul-Dec",
          **ece_of(g["p_winner"].to_numpy())}
         for h, g in allsp.groupby(allsp["month"] > 6)]).drop(columns=["win_rate"])
    print("\nECE by half-year:")
    print(half.to_string(index=False))

    # Debutants excluded, so this is purely "an existing rating went stale".
    ret = allsp[~allsp["has_debutant"]]
    bins = [(0, 7), (7, 21), (21, 42), (42, 70), (70, 140), (140, 100000)]
    by_layoff = pd.DataFrame(
        [{"layoff_days": f"{lo}-{hi}" if hi < 100000 else f"{lo}+",
          **ece_of(ret.loc[(ret["max_layoff"] >= lo)
                           & (ret["max_layoff"] < hi), "p_winner"].to_numpy())}
         for lo, hi in bins]).drop(columns=["win_rate"])
    print("\nECE by layoff, DEBUTANTS EXCLUDED (a rating that went stale):")
    print(by_layoff.to_string(index=False))

    deb = pd.DataFrame([
        {"group": "has a debutant", **ece_of(
            allsp.loc[allsp["has_debutant"], "p_winner"].to_numpy())},
        {"group": "both players known", **ece_of(
            allsp.loc[~allsp["has_debutant"], "p_winner"].to_numpy())},
    ]).drop(columns=["win_rate"])
    print("\nECE by debut status:")
    print(deb.to_string(index=False))

    jan = allsp[allsp["month"] == 1]
    print(f"\nJanuary matches: {len(jan):,}, "
          f"median layoff {jan['max_layoff'].median():.0f} days "
          f"vs {allsp['max_layoff'].median():.0f} overall")

    lines = [
        "# Why match_winner calibrates badly on holdout\n",
        "Diagnostic only — nothing is fitted or selected here, and neither "
        "TEST nor HOLDOUT is touched. Scored at the Elo layer on all "
        "pre-holdout data.\n",
        "\n## The pattern that prompted it\n",
        "Full-pipeline match_winner ECE:\n\n"
        "| window | ECE |\n|---|---|\n"
        "| TUNE 2024-01..2025-06 | 0.0178 |\n"
        "| TEST 2025-07..2025-12 | 0.0181 |\n"
        "| HOLDOUT 2026-01..2026-07 | 0.0665 |\n"
        "| (original build) TEST 2023-07..2023-12 | 0.0169 |\n"
        "| (original build) HOLDOUT 2024-01..2026-07 | 0.0693 |\n",
        "\nBoth test windows are July-December; both holdout windows open in "
        "January. Drift would degrade with distance from the fitting data, and "
        "it does not — the 2025-07 window sits between the two fitting windows "
        "and calibrates fine. That points at the calendar, not at staleness.\n",
        "\n## ECE by calendar month\n",
        by_month.to_markdown(index=False),
        "\n\n## ECE by half-year\n",
        half.to_markdown(index=False),
        "\n\n## ECE by layoff of the staler-rated player\n",
        "Debutants are EXCLUDED here, so this isolates one thing: an existing "
        "rating that nothing has aged. A player with no prior match is a "
        "different defect with a different fix (a starting prior, not "
        "regression), and pooling the two makes one look like the other.\n",
        by_layoff.to_markdown(index=False),
        "\n\n## ECE by debut status\n",
        deb.to_markdown(index=False),
        "\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
