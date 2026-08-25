"""Does blending the model with the market beat either one alone?

The accuracy question `reports/model_vs_market.md` left open. That report
scored the model and the bookmakers as rival forecasters and found the model
behind on Brier and log-loss but ahead on ECE — worse at ranking, better at
knowing its own confidence. Two forecasters whose errors differ that way are
the textbook case for combining rather than choosing.

The combination is a weighted average **in log-odds space**, which is the
right space for probabilities: a straight average of 0.90 and 0.50 is 0.70,
which throws away most of the first forecast's confidence, while the log-odds
average lands at 0.75. One parameter, `w`, the weight on the model.

    logit(p) = w * logit(p_model) + (1 - w) * logit(p_market)

`w = 1` is the model unchanged, `w = 0` is the market unchanged, so the fit
cannot do worse than the better of the two except by sampling noise — which
is exactly why the honest number here is the cross-validated one, not the
in-sample one. Both are reported.

A second, freer form is fitted alongside it: an unconstrained pair of
coefficients on the same two log-odds, with no intercept. Dropping the
intercept is deliberate, not laziness — an intercept would make the forecast
asymmetric, so swapping which player is "player A" would change the answer.
The free form can shrink both inputs toward a coin flip, which the convex
blend cannot, and that matters if both forecasts are overconfident.

**SCOPE: TUNE only (2024-01-01 .. 2025-06-30), ATP main tour.** `w` is a
hyperparameter and TUNE is the set entitled to select it. TEST and HOLDOUT
are not read. Logged to `reports/tune_ledger.json` per ground rule 2.

**This is not a shipped change.** Whether a projection that reads the
bookmaker's price still counts as "an accurate match projection" under the
2026-08-08 goal is a judgment call for the human, not for this script. What
the script settles is only the empirical half: how much accuracy the blend
buys, if it is allowed at all.

Run:  python -m scripts.fit_market_blend
      python -m scripts.fit_market_blend --self-check
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from model import constants as C
from model import recalibrate as RC
from scripts.model_vs_market import build_joined, consensus

REPORT = C.REPORTS_DIR / "market_blend.md"
LEDGER = C.REPORTS_DIR / "tune_ledger.json"

#: Probabilities are clipped before the log-odds transform. A de-vigged quote
#: of exactly 0 or 1 would send logit to infinity and take the fit with it.
EPS = 1e-6

N_FOLDS = 5
SEED = 20260825


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _mirror(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both orientations of every match, so the outcome is not all ones.

    Every joined row is the quote on the player who actually won, so `y` is 1
    by construction. Scoring that alone would reward a forecaster for saying
    "1.0" every time. Mirroring each match into its losing orientation
    restores a real base rate without inventing any data.
    """
    x_model = np.concatenate([logit(a), logit(1 - a)])
    x_market = np.concatenate([logit(b), logit(1 - b)])
    y = np.concatenate([np.ones(len(a)), np.zeros(len(a))])
    return x_model, x_market, y


def _fit_convex(xm: np.ndarray, xk: np.ndarray, y: np.ndarray) -> float:
    """The single weight `w` on the model, minimising log-loss."""
    def loss(w: float) -> float:
        return RC.log_loss(sigmoid(w * xm + (1 - w) * xk), y)
    return float(minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded").x)


def _fit_market_scale(xk: np.ndarray, y: np.ndarray) -> float:
    """One scale on the market's log-odds, with the model absent entirely.

    The control the free stack needs. If scaling the market alone recovers
    most of the free stack's gain, that gain is a statement about the
    bookmakers being slightly under- or overconfident, not about the model
    contributing information — and only the second would be a finding.
    """
    def loss(a: float) -> float:
        return RC.log_loss(sigmoid(a * xk), y)
    return float(minimize_scalar(loss, bounds=(0.1, 3.0), method="bounded").x)


def _fit_free(xm: np.ndarray, xk: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unconstrained coefficients on both log-odds, no intercept."""
    def loss(c: np.ndarray) -> float:
        return RC.log_loss(sigmoid(c[0] * xm + c[1] * xk), y)
    return minimize(loss, x0=np.array([0.5, 0.5]), method="Nelder-Mead").x


def _scores(p: np.ndarray, y: np.ndarray) -> dict:
    return {"brier": RC.brier(p, y), "logloss": RC.log_loss(p, y),
            "ece": RC.calibration_error(p, y)}


def _cross_validated(cons: pd.DataFrame) -> tuple[dict, dict, dict, list[float]]:
    """Out-of-fold scores, folded on matches rather than on mirrored rows.

    Splitting after the mirror would put a match's two orientations in
    different folds, which leaks the answer: knowing p for one orientation
    gives 1 - p for the other. Folds are drawn on match index first, then
    each side is mirrored within its own fold.
    """
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(cons))
    folds = np.array_split(order, N_FOLDS)
    model_p, market_p = cons.p_model_cal.to_numpy(), cons.market_p.to_numpy()

    oof_convex, oof_free, oof_scale, oof_y, ws = [], [], [], [], []
    for k in range(N_FOLDS):
        held = folds[k]
        rest = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        xm_tr, xk_tr, y_tr = _mirror(model_p[rest], market_p[rest])
        xm_te, xk_te, y_te = _mirror(model_p[held], market_p[held])
        w = _fit_convex(xm_tr, xk_tr, y_tr)
        c = _fit_free(xm_tr, xk_tr, y_tr)
        a = _fit_market_scale(xk_tr, y_tr)
        ws.append(w)
        oof_convex.append(sigmoid(w * xm_te + (1 - w) * xk_te))
        oof_free.append(sigmoid(c[0] * xm_te + c[1] * xk_te))
        oof_scale.append(sigmoid(a * xk_te))
        oof_y.append(y_te)
    y = np.concatenate(oof_y)
    return (_scores(np.concatenate(oof_convex), y),
            _scores(np.concatenate(oof_free), y),
            _scores(np.concatenate(oof_scale), y), ws)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _log_to_ledger(entry: dict) -> None:
    ledger = json.loads(LEDGER.read_text())
    ledger.append(entry)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")


def self_check() -> None:
    """The endpoints have to reproduce the inputs, or the algebra is wrong."""
    p_model = np.array([0.9, 0.4, 0.65, 0.2])
    p_market = np.array([0.5, 0.55, 0.7, 0.35])
    xm, xk, _ = _mirror(p_model, p_market)
    both = np.concatenate([p_model, 1 - p_model])
    assert np.allclose(sigmoid(1.0 * xm + 0.0 * xk), both, atol=1e-5)
    market_both = np.concatenate([p_market, 1 - p_market])
    assert np.allclose(sigmoid(0.0 * xm + 1.0 * xk), market_both, atol=1e-5)
    # A blend must sit between its inputs, never outside them.
    mid = sigmoid(0.5 * xm + 0.5 * xk)
    assert np.all(mid >= np.minimum(both, market_both) - 1e-9)
    assert np.all(mid <= np.maximum(both, market_both) + 1e-9)
    # Mirroring must be symmetric: the two orientations sum to one.
    n = len(p_model)
    assert np.allclose(mid[:n] + mid[n:], 1.0, atol=1e-9)
    print("self-check OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return

    self_check()
    mg, _ = build_joined()
    cons = consensus(mg)
    print(f"\nconsensus matches: {len(cons):,}")

    model_p, market_p = cons.p_model_cal.to_numpy(), cons.market_p.to_numpy()
    xm, xk, y = _mirror(model_p, market_p)

    s_model, s_market = _scores(sigmoid(xm), y), _scores(sigmoid(xk), y)
    w = _fit_convex(xm, xk, y)
    c = _fit_free(xm, xk, y)
    s_convex = _scores(sigmoid(w * xm + (1 - w) * xk), y)
    s_free = _scores(sigmoid(c[0] * xm + c[1] * xk), y)
    a = _fit_market_scale(xk, y)
    s_scale = _scores(sigmoid(a * xk), y)
    cv_convex, cv_free, cv_scale, ws = _cross_validated(cons)

    print(f"\nfitted w (weight on the model): {w:.4f}")
    print(f"free coefficients: model {c[0]:.4f}, market {c[1]:.4f}")
    print(f"market-only scale (the control): {a:.4f}")
    print(f"per-fold w: {', '.join(f'{v:.3f}' for v in ws)}")
    tab = pd.DataFrame([
        {"forecaster": "market (median of books)", **s_market},
        {"forecaster": "model (calibrated)", **s_model},
        {"forecaster": f"blend, w={w:.3f} (in-sample)", **s_convex},
        {"forecaster": "blend (5-fold out-of-fold)", **cv_convex},
        {"forecaster": "free stack (in-sample)", **s_free},
        {"forecaster": "free stack (5-fold out-of-fold)", **cv_free},
        {"forecaster": f"market scaled x{a:.3f}, no model (in-sample)", **s_scale},
        {"forecaster": "market scaled, no model (5-fold out-of-fold)", **cv_scale},
    ])
    print("\n" + tab.to_string(index=False))

    gain_ll = s_market["logloss"] - cv_convex["logloss"]
    gain_br = s_market["brier"] - cv_convex["brier"]
    print(f"\nblend vs market, out-of-fold: log-loss {gain_ll:+.5f}, "
          f"Brier {gain_br:+.5f} (positive means the blend is better)")

    REPORT.write_text("\n".join([
        "# Blending the model with the market\n",
        "One weight `w` on the model, applied in log-odds space, fitted by "
        "log-loss. `w = 1` is the model alone, `w = 0` is the market alone.\n",
        f"\n**Scope: TUNE only ({C.TUNE_START} .. {C.TUNE_END}), ATP main "
        f"tour, {len(cons):,} matches.** TEST and HOLDOUT were not read. `w` "
        "is a selected hyperparameter and is logged to `tune_ledger.json`.\n",
        f"\n**Fitted `w` = {w:.4f}** — the weight the data puts on the model, "
        f"the remaining {1 - w:.4f} on the market. Free-form coefficients, "
        f"fitted without the convexity constraint and without an intercept: "
        f"model {c[0]:.4f}, market {c[1]:.4f}. The control — one scale on the "
        f"market's log-odds with the model absent — fits {a:.4f}.\n",
        "\n## Scores\n",
        "\n" + tab.to_markdown(index=False, floatfmt=".5f") + "\n",
        "\n## Reading this\n",
        "\nThe in-sample blend rows cannot lose to their own inputs — the "
        "endpoints are inside the search space — so they are reported for "
        "completeness, not as evidence. The out-of-fold rows are the claim: "
        "the weight is fitted on four folds and scored on the fifth, five "
        "times over, folded on matches so that a match's two orientations "
        "never straddle a fold.\n",
        f"\nAgainst the market alone, out-of-fold: log-loss {gain_ll:+.5f}, "
        f"Brier {gain_br:+.5f}. Positive means the blend is better.\n",
        f"\nPer-fold `w`: {', '.join(f'{v:.3f}' for v in ws)}. A weight that "
        "moves a lot between folds is a weight fitted on noise; a stable one "
        "is a real division of labour between the two forecasts.\n",
        "\n## What this does not settle\n",
        "\nWhether a projection that reads the bookmaker's price still counts "
        "as an accurate *match projection* under the 2026-08-08 goal. A blend "
        "with a low `w` is mostly the market repeated back, and it cannot "
        "price a match no bookmaker has quoted. That is a judgment call for "
        "the human. This report settles only how much accuracy is on the "
        "table.\n",
        "\nThe usual caveat on the odds source applies: these are OddsPortal "
        "2024-2025 quotes, several books, de-vigged. The paper-trading path "
        "now reads PointsBet alone, which is one book and not a consensus.\n",
    ]) + "\n")
    print(f"\nwrote {REPORT}")

    _log_to_ledger({
        "stage": "market_blend",
        "run_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{_git_sha()}",
        "metric": "tune_logloss_out_of_fold",
        "baseline": "market consensus alone (median de-vigged across books)",
        "selected": {"w_model": w, "free_model": float(c[0]),
                     "free_market": float(c[1])},
        "tune_gain": gain_ll,
        "tune_metric_value": cv_convex["logloss"],
        "fit_rolling_origin_gain_folds": ws,
        "fit_rolling_origin_gain_mean": float(np.mean(ws)),
        "fit_rolling_origin_gain_std": float(np.std(ws)),
        "baselines": {"tune_logloss": s_market["logloss"],
                      "tune_brier": s_market["brier"],
                      "tune_ece": s_market["ece"],
                      "model_alone_logloss": s_model["logloss"],
                      "model_alone_brier": s_model["brier"],
                      "market_scaled_oof_logloss": cv_scale["logloss"],
                      "market_scaled_oof_brier": cv_scale["brier"],
                      "free_stack_oof_logloss": cv_free["logloss"],
                      "free_stack_oof_brier": cv_free["brier"],
                      "market_scale": a},
        "frozen": False,
        "notes": (
            f"exploratory: {len(cons):,} TUNE ATP matches with match_winner "
            "quotes. NOT shipped — a blend consumes the bookmaker's price, "
            "and whether that is admissible under the 2026-08-08 projection "
            "goal is an open human decision. `fit_rolling_origin_gain_folds` "
            "holds the per-fold fitted w, not a gain, because the fold "
            "structure here is 5-fold on TUNE matches rather than a "
            "rolling origin on FIT."),
    })
    print(f"logged to {LEDGER}")


if __name__ == "__main__":
    main()
