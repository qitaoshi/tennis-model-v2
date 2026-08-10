"""Flat-stake ROI on total games, at prices you could actually have taken.

Under the current goal this is a BENCHMARK, not the objective: a model can be
the best available forecaster and still lose money, because the bookmaker's
margin sits between being right and getting paid. The number is here to say
how far the totals edge is from clearing that margin, not to justify betting.

Method, chosen to avoid flattering the result:

* Flat one unit per bet. No Kelly, no staking plan — those amplify an edge
  that has to exist first.
* A bet is placed when the model's probability exceeds the probability
  implied by the RAW decimal odds, i.e. the price actually on offer with the
  vig in it. Betting against the de-vigged probability instead would count a
  bet as positive-edge when the real price is negative-edge, which is how
  paper edges are manufactured.
* Profit is ``decimal_odds - 1`` on a win, ``-1`` on a loss, ``0`` on a push,
  with the push still in the denominator.
* Bootstrap CI over matches, not selections — a match's ladder rungs are one
  correlated cluster, and resampling selections would treat them as
  independent evidence and shrink the interval to fiction.

SCOPE: TUNE only (2024-01-01 .. 2025-06-30). The odds file also covers TEST
and HOLDOUT and both are spent.

Run:  python -m scripts.totals_roi
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import constants as C
from scripts.model_vs_market import ODDS
from scripts.multi_market_clv import _actual, _join_odds, _pair, _selections
from scripts import panel as P

REPORT = C.REPORTS_DIR / "totals_roi.md"

EDGE_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)


def _boot(profit: np.ndarray, groups: np.ndarray, n: int = 2000) -> tuple:
    """Bootstrap the mean, resampling whole matches."""
    keys = pd.unique(groups)
    if len(keys) < 20:
        return float("nan"), float("nan")
    idx = {k: np.where(groups == k)[0] for k in keys}
    rng = np.random.default_rng(C.MC_SEED)
    means = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sel = np.concatenate([idx[k] for k in pick])
        means.append(profit[sel].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _unfiltered(j: pd.DataFrame) -> str:
    """One bet per match on the primary line, no edge filter at all.

    This is the question "what if it just bet every match". It matters
    because it separates two things the filtered numbers confound: whether
    the model has any signal, and whether the edge filter is doing all the
    work. Betting blindly at fair prices loses exactly the bookmaker's
    margin, so **the margin is the benchmark, not zero** — beating it means
    the model's side selection carries information, even while losing money.

    One bet per match, so the correlated-ladder problem disappears and each
    observation is independent.
    """
    out = []
    for book in ("bet365", "best"):
        g = j[j["bookmaker"] == "bet365"] if book == "bet365" else j
        # The primary line is the one the book quotes most; ties go to the
        # line nearest the median quoted total, which is what "the" line means.
        rows = []
        for mid, m in g.groupby("match_id"):
            counts = m.groupby("line").size()
            top = counts[counts == counts.max()].index
            med = m["line"].astype(float).median()
            line = min(top, key=lambda x: abs(float(x) - med))
            at = m[m["line"] == line]
            # Need both sides at that line to know the price and the margin.
            sides = {}
            for sel, s in at.groupby("selection"):
                best = s.loc[s["decimal_odds"].idxmax()]
                sides[sel] = best
            if len(sides) != 2:
                continue
            over = 1.0 / sum(1.0 / s["decimal_odds"] for s in sides.values())
            pick = max(sides.values(), key=lambda s: s["model_p"])
            rows.append({"match_id": mid,
                         "profit": 0.0 if pick["push"] > 0 else
                                   (pick["decimal_odds"] - 1.0
                                    if pick["y"] > 0 else -1.0),
                         "margin": 1.0 - over, "y": pick["y"],
                         "odds": pick["decimal_odds"]})
        d = pd.DataFrame(rows)
        if len(d) < 100:
            continue
        roi = float(d["profit"].mean())
        lo, hi = _boot(d["profit"].to_numpy(), d["match_id"].to_numpy())
        margin = float(d["margin"].mean())
        out.append({"book": book, "n_matches": len(d), "roi_pct": 100 * roi,
                    "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
                    "hit_rate": float((d["y"] > 0).mean()),
                    "mean_odds": float(d["odds"].mean()),
                    "book_margin_pct": 100 * margin,
                    "vs_margin_pp": 100 * (roi + margin)})
    tab = pd.DataFrame(out)
    print("\nUNFILTERED — one bet per match, primary line, no edge filter")
    print(tab.to_string(index=False))
    return ("## Betting every match, no edge filter\n\nOne bet per match on "
            "the primary line, taking whichever side the model prefers, "
            "regardless of whether it beats the price. One bet per match, so "
            "these observations are independent — the ladder-correlation "
            "caveat above does not apply here.\n\n"
            "`book_margin_pct` is the bookmaker's actual overround on that "
            "line, and `vs_margin_pp` is ROI plus margin: **how much better "
            "than blind betting the model's side selection is**. Betting "
            "blind at these prices loses the margin by construction, so zero "
            "on that column means no signal and positive means real "
            "information, even while losing money.\n\n"
            + tab.to_markdown(index=False) + "\n")


def main() -> None:
    pan = P.build(splits=("tune",))
    rec = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet").set_index("match_id")
    pan = pan[rec.loc[pan["match_id"], "tour"].eq("atp").to_numpy()].copy()
    for col in ("date", "winner_name", "loser_name", "tourney_name", "surface",
                "winner_games", "loser_games", "n_sets", "score"):
        pan[col] = rec.loc[pan["match_id"], col].to_numpy()
    pan["pair_key"] = [_pair(a, b) for a, b in
                       zip(pan["winner_name"], pan["loser_name"])]

    sels = _selections(pan)
    odds = pd.read_parquet(ODDS)
    odds["date"] = pd.to_datetime(odds["date"]).dt.date
    odds = odds[(odds["date"] >= C.TUNE_START) & (odds["date"] <= C.TUNE_END)]
    odds = odds[odds["market"] == "total_games"]
    j = _join_odds(sels, odds)
    print(f"joined total-games quotes on TUNE: {len(j):,}")

    out = [_actual(r) for _, r in j.iterrows()]
    j["y"] = [a for a, _ in out]
    j["push"] = [float(p) for _, p in out]
    j = j[np.isfinite(j["decimal_odds"]) & (j["decimal_odds"] > 1.0)].copy()

    # Edge against the RAW price on offer, not against the de-vigged number.
    j["implied_raw"] = 1.0 / j["decimal_odds"]
    j["edge"] = j["model_p"] - j["implied_raw"]
    j["profit"] = np.where(j["push"] > 0, 0.0,
                           np.where(j["y"] > 0, j["decimal_odds"] - 1.0, -1.0))

    rows = []
    for book in ["bet365", "ALL BOOKS (best price)"]:
        if book == "bet365":
            g0 = j[j["bookmaker"] == "bet365"]
        else:
            # Best available price per selection, which is the most generous
            # honest assumption: it assumes you always shopped perfectly.
            g0 = j.sort_values("decimal_odds", ascending=False).drop_duplicates(
                ["match_id", "selection"])
        for thr in EDGE_THRESHOLDS:
            g = g0[g0["edge"] > thr]
            if len(g) < 50:
                continue
            roi = float(g["profit"].mean())
            lo, hi = _boot(g["profit"].to_numpy(), g["match_id"].to_numpy())
            rows.append({"book": book, "min_edge": thr, "n_bets": len(g),
                         "n_matches": g["match_id"].nunique(),
                         "roi_pct": 100 * roi, "ci_lo_pct": 100 * lo,
                         "ci_hi_pct": 100 * hi,
                         "hit_rate": float((g["y"] > 0).mean()),
                         "mean_odds": float(g["decimal_odds"].mean())})
    tab = pd.DataFrame(rows)
    print("\n" + tab.to_string(index=False))

    best = tab.loc[tab["roi_pct"].idxmax()] if len(tab) else None
    any_clear = bool((tab["ci_lo_pct"] > 0).any()) if len(tab) else False
    print(f"\nany configuration with a CI clear of zero: {any_clear}")

    unf = _unfiltered(j)

    lines = [
        "# Flat-stake ROI on total games\n",
        "A **benchmark, not the objective**. The model can be the best "
        "available forecaster and still lose money, because the bookmaker's "
        "margin sits between being right and getting paid. This says how far "
        "the totals edge is from clearing that margin.\n",
        f"\n**Scope: TUNE only ({C.TUNE_START} .. {C.TUNE_END}), ATP main "
        "tour.** TEST and HOLDOUT are spent and were not read.\n",
        "\n## Method\n",
        "- Flat one unit per bet; no staking plan, which would amplify an "
        "edge that has to exist first.\n"
        "- A bet is placed when the model's probability beats the probability "
        "implied by the **raw** decimal odds — the price actually on offer, "
        "vig included. Comparing against the de-vigged number instead counts "
        "bets as positive-edge when the real price is not, which is how paper "
        "edges get manufactured.\n"
        "- Push returns the stake and stays in the denominator.\n"
        "- Bootstrap resamples **whole matches**, because one match's ladder "
        "rungs are a correlated cluster and treating them as independent "
        "would shrink the interval to fiction.\n",
        "\n## Results\n",
        tab.to_markdown(index=False),
        f"\n\n**Any configuration whose confidence interval clears zero: "
        f"{any_clear}.**\n",
        "\n## Reading this\n",
        "`ALL BOOKS (best price)` assumes you always found the best quote of "
        "the seven books in the file. That is the most generous honest "
        "assumption available and still not achievable in practice — it "
        "ignores limits, closing lines and the accounts being restricted.\n",
        "\nRaising the edge threshold trades sample for selectivity. If ROI "
        "does not improve as the threshold rises, the model's edge estimate "
        "carries no ordering information, which matters more than the "
        "headline number: it means the model cannot tell its good bets from "
        "its bad ones.\n",
        "\n" + unf,
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
