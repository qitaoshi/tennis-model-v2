"""Does the underdog CLV vanish under a better de-vig than proportional?

Measurement only: no parameter fitted or selected. segment_edge_holdout.py
found the pooled +2.89% match_winner CLV is carried almost entirely by 917
market-underdog picks showing +38% mean CLV yet -8.4% real ROI — the
signature of favorite-longshot bias in PROPORTIONAL de-vig (normalizing
1/odds across both sides), which over-credits the underdog side because
bookmaker margin is fatter there.

This recomputes the de-vigged market probability three ways on the same
HOLDOUT join and reports favorite/underdog CLV under each:
  - proportional : pw = (1/ow) / (1/ow + 1/ol)      (what clv_backtest uses)
  - power        : (1/ow)^n + (1/ol)^n = 1, pw=(1/ow)^n
  - shin         : Shin (1993) insider-trader model, solved per match
If the underdog CLV collapses toward zero under power/Shin, the "edge" was a
de-vig artifact, not model skill.

Run with ``python3 -m scripts.devig_method_clv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from model import constants as C
from scripts.calibration_ablation_clv import _model_predictions
from scripts.clv_backtest import _bootstrap_mean, _read_odds

REPORT_PATH = C.REPORTS_DIR / "devig_method_clv.md"


def _power_pw(inv_w: float, inv_l: float) -> float:
    """Two-way power de-vig: exponent n with (inv_w^n + inv_l^n) = 1."""
    f = lambda n: inv_w ** n + inv_l ** n - 1.0
    # margin>0 => inv_w+inv_l>1 => f(1)>0; f(large)->0-; root n>1 shrinks longshot.
    # overround<=1 (noisy/crossed odds) leaves no bracket -> proportional fallback.
    if f(1.0) <= 0.0:
        return inv_w / (inv_w + inv_l)
    n = brentq(f, 1.0, 50.0)
    return inv_w ** n


def _shin_pw(inv_w: float, inv_l: float) -> float:
    """Two-way Shin de-vig: solve insider fraction z, return favorite side pw."""
    booksum = inv_w + inv_l  # overround > 1
    def z_resid(z: float) -> float:
        s = 0.0
        for pi in (inv_w, inv_l):
            s += (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / booksum) - z) / (2.0 * (1.0 - z))
        return s - 1.0
    try:
        z = brentq(z_resid, 1e-9, 0.5)
    except ValueError:
        return inv_w / booksum  # degenerate: fall back to proportional
    return (np.sqrt(z * z + 4.0 * (1.0 - z) * inv_w * inv_w / booksum) - z) / (2.0 * (1.0 - z))


def main() -> None:
    odds = _read_odds()
    model = _model_predictions()

    join_keys = ["year", "surface_key"]
    winner_join = join_keys + ["winner_key", "loser_key"]
    joined = model.merge(odds, left_on=winner_join, right_on=winner_join,
                         how="left", suffixes=("", "_odds"))
    joined = joined.dropna(subset=["odds_w", "odds_l"]).copy()
    if not len(joined):
        raise SystemExit("No holdout matches joined to odds; inspect name/tournament keys.")

    inv_w = 1.0 / joined["odds_w"].to_numpy()
    inv_l = 1.0 / joined["odds_l"].to_numpy()

    methods = {"proportional": inv_w / (inv_w + inv_l)}
    methods["power"] = np.array([_power_pw(a, b) for a, b in zip(inv_w, inv_l)])
    methods["shin"] = np.array([_shin_pw(a, b) for a, b in zip(inv_w, inv_l)])

    is_winner = joined["model_edge_side"].eq("winner").to_numpy()
    won = is_winner.astype(float)
    sel_odds = np.where(is_winner, joined["odds_w"].to_numpy(), joined["odds_l"].to_numpy())
    p_raw = joined["model_p_raw"].to_numpy()

    def block(name: str, pw: np.ndarray) -> tuple[str, dict]:
        # market prob for the side the model bet
        market_p = np.where(is_winner, pw, 1.0 - pw)
        edge = p_raw - market_p
        clv = p_raw / market_p - 1.0
        market_fav = np.where(is_winner, pw >= 0.5, (1.0 - pw) >= 0.5)

        def sub(mask: np.ndarray, label: str) -> str:
            g_clv, g_edge = clv[mask], edge[mask]
            mean, lo, hi = _bootstrap_mean(g_clv)
            pos = mask & (edge > 0)
            if lo > 0 and pos.sum() >= 20:
                profit = np.where(won[pos] > 0, sel_odds[pos] - 1.0, -1.0)
                roi, rlo, rhi = _bootstrap_mean(profit)
                roi_s = f"{roi:+.2%} (CI {rlo:+.2%} to {rhi:+.2%})"
            else:
                roi_s = "n/a"
            return (f"| {name} | {label} | {mask.sum():,} | {g_edge.mean():+.2%} | "
                    f"{mean:+.2%} | {lo:+.2%} to {hi:+.2%} | {roi_s} (n={int(pos.sum()):,}) |")

        lines = [sub(np.ones(len(pw), bool), "all"),
                 sub(market_fav, "model picks favorite"),
                 sub(~market_fav, "model picks underdog")]
        overround = float((inv_w + inv_l).mean())
        return "\n".join(lines), {"overround": overround}

    rows, meta = [], {}
    for name, pw in methods.items():
        blk, m = block(name, pw)
        rows.append(blk)
        meta = m

    report = f"""# De-vig method vs match_winner CLV — HOLDOUT

Measurement only: no parameter fitted or selected. Raw (uncalibrated) model
probability throughout. {len(joined):,} matched rows. Mean two-way overround
{meta['overround']:.4f} ({(meta['overround'] - 1) * 100:.2f}% margin).

`segment_edge_holdout.py` traced the pooled +2.89% CLV to market-underdog
picks (+38% mean CLV, -8.4% real ROI) — the favorite-longshot signature of
proportional de-vig. This recomputes the de-vigged market probability three
ways and re-measures. If the underdog CLV collapses under power/Shin, it was
a methodology artifact, not model skill.

| method | segment | n | mean edge | mean CLV | CLV 95% CI | ROI (n edge>0) |
|---|---|---:|---:|---:|---|---|
{chr(10).join(rows)}

## Reading this

Proportional de-vig splits the bookmaker margin evenly across both sides;
power and Shin both shift more of the margin onto the underdog, lowering its
de-vigged probability. If underdog mean CLV falls sharply from proportional
to Shin while ROI stays negative, the +38% underdog CLV was manufactured by
the de-vig, and match_winner has no demonstrated edge under a fair margin
model. Shin is the most defensible of the three for bookmaker prices.
"""
    REPORT_PATH.write_text(report)
    print(report)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
