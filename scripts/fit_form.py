"""Short-horizon form: does a fast-adapting rating fix match_winner?

Two holdout windows now agree that match_winner is the model's one badly
calibrated output (ECE ~0.07 against <0.025 for every distribution family),
and two different calibration maps have failed to move it. A monotone map
cannot fix it, which means the defect is upstream, in how player strength is
judged rather than in how the output is dressed.

The hypothesis this script tests: the rating is too slow. Stage 2's serve
rates decay with a 730-day half-life and Stage 3's Elo moves at K=48, so a
player who is currently hot or currently injured takes months to be
recognised. The fatigue experiment pointed the same way without being asked
to — its coefficients said recently-active players outperform their rating,
which is what a stale rating looks like.

The test is a second Elo ladder with a much larger K, blended with the frozen
one in logit space:

    logit(p) = (1 - w) * logit(p_slow) + w * logit(p_fast)

w = 0 is exactly today's model, so the baseline is inside the grid rather
than alongside it.

THIS IS A SCREEN, NOT THE FIT. It scores the blend at the Elo layer, where a
grid point costs seconds, instead of pushing all of it through the engine,
where it would cost hours. If the blend cannot beat w=0 here it cannot help
downstream either, because everything downstream is a monotone function of
this probability. Only a winner earns a full pipeline run.

Fitted on FIT, selected on TUNE. TEST and HOLDOUT are both already spent and
are NOT touched.

Run:  python -m scripts.fit_form
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import constants as C
from model import elo as E
from model import recalibrate as RC

#: Fast-ladder K values to try. Stage 3 selected K=48 for the slow ladder.
K_FAST = (96.0, 144.0, 192.0, 288.0)

#: Weight on the fast ladder. 0.0 is the current model.
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)

REPORT = C.REPORTS_DIR / "form_screen.md"


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def _blend(p_slow: np.ndarray, p_fast: np.ndarray, w: float) -> np.ndarray:
    if w == 0.0:
        return p_slow
    z = (1 - w) * _logit(p_slow) + w * _logit(p_fast)
    return 1.0 / (1.0 + np.exp(-z))


def _score(p_winner: np.ndarray) -> dict:
    """Metrics for winner-first probabilities.

    ``p_winner`` is the probability assigned to the player who actually won,
    so log-loss is just -mean(log p). ECE needs both orientations or every
    outcome is a 1 by construction and the number is an artefact — the same
    trap that produced a bogus ECE in the first backtest.
    """
    p = np.concatenate([p_winner, 1.0 - p_winner])
    y = np.concatenate([np.ones(len(p_winner)), np.zeros(len(p_winner))])
    return {"logloss": float(-np.mean(np.log(np.clip(p_winner, 1e-9, 1)))),
            "brier": RC.brier(p, y), "ece": RC.calibration_error(p, y)}


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s3 = fitted["stage_3"]

    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    assert m["date"].max() < C.HOLDOUT_CUTOFF, "holdout must not be loaded here"
    # Ratings replay over all history; scoring is restricted to TUNE below.
    print(f"{len(m):,} matches loaded, through {m['date'].max()}")

    def params(k: float) -> E.EloParams:
        return E.EloParams(k=k, k_chall_mult=s3["k_chall_mult"],
                           surface_weight=s3["surface_weight"],
                           inactivity_half_life=s3["inactivity_half_life"],
                           level_gap=s3["level_gap"],
                           level_offset=s3["level_offset"])

    slow = E.run_elo(m, params(s3["k"]))[["match_id", "p_winner"]]
    meta = m.set_index("match_id")
    slow["split"] = meta.loc[slow["match_id"], "split"].to_numpy()
    tune_mask = (slow["split"] == "tune").to_numpy()
    print(f"slow ladder K={s3['k']}: {int(tune_mask.sum()):,} TUNE matches")

    rows = []
    for kf in K_FAST:
        fast = E.run_elo(m, params(kf))[["match_id", "p_winner"]]
        assert (fast["match_id"].to_numpy() == slow["match_id"].to_numpy()).all(), \
            "ladders must align row for row"
        ps, pf = slow["p_winner"].to_numpy(), fast["p_winner"].to_numpy()
        for w in WEIGHTS:
            sc = _score(_blend(ps, pf, w)[tune_mask])
            rows.append({"k_fast": kf, "weight": w, **sc})
        print(f"  K_fast={kf:5.0f} done")

    grid = pd.DataFrame(rows)
    # w=0 is the same model regardless of k_fast, so collapse those duplicates.
    base = grid[grid["weight"] == 0.0].iloc[0]
    grid = pd.concat([pd.DataFrame([{**base.to_dict(), "k_fast": 0.0}]),
                      grid[grid["weight"] > 0.0]], ignore_index=True)

    grid = grid.sort_values("logloss").reset_index(drop=True)
    best = grid.iloc[0]
    gain = float(base["logloss"] - best["logloss"])
    helped = bool(best["weight"] > 0.0 and gain > 0)

    print(f"\nbaseline (w=0): log-loss {base['logloss']:.5f}, "
          f"Brier {base['brier']:.5f}, ECE {base['ece']:.5f}")
    print(f"best: K_fast={best['k_fast']:.0f} w={best['weight']:.2f} "
          f"log-loss {best['logloss']:.5f} ECE {best['ece']:.5f}")
    print(f"log-loss gain over the current model: {gain:+.5f}")
    print(f"\nSCREEN {'PASSED' if helped else 'FAILED'} — "
          f"{'take it through the full pipeline' if helped else 'stop here'}")
    print("\ntop of grid:")
    print(grid.head(10).to_string(index=False))

    lines = [
        "# Short-horizon form — Elo-layer screen\n",
        "Does blending a fast-adapting Elo ladder into the frozen one improve "
        "match-winner prediction? Scored at the Elo layer, where a grid point "
        "costs seconds. Everything downstream is a monotone function of this "
        "probability, so a blend that cannot win here cannot win there.\n",
        "\nFitted on all history up to the cutoff, **scored on TUNE only** "
        "(2024-01-01 .. 2025-06-30). TEST and HOLDOUT are both already spent "
        "and were not touched.\n",
        "\n## Why\n",
        "match_winner is the model's one badly calibrated family — ECE ~0.07 "
        "on two independent holdouts, against under 0.025 for every "
        "distribution family — and two different calibration maps failed to "
        "move it. A monotone map cannot fix it, so the defect is upstream. "
        "The rating being too slow is the leading candidate, and the fatigue "
        "experiment pointed the same way unprompted.\n",
        f"\n## Baseline\n\nThe current model is `weight = 0`: log-loss "
        f"{base['logloss']:.5f}, Brier {base['brier']:.5f}, "
        f"ECE {base['ece']:.5f}.\n",
        "\n## Grid, ranked by TUNE match-winner log-loss\n",
        grid.head(15).to_markdown(index=False),
        f"\n\n**Best: K_fast {best['k_fast']:.0f}, weight {best['weight']:.2f}, "
        f"log-loss gain {gain:+.5f}.** Screen "
        f"**{'PASSED' if helped else 'FAILED'}**.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
