"""Fit the match-context (fatigue) adjustment: coefficients on FIT, gate on TUNE.

The model describes ability and nothing about the state a player walks on
court in. This stage adds rest, recent workload and entry route as a shift to
the serve-point probability. See ``model/fatigue.py`` for the feature
definitions and why they are applied in logit space.

Coefficients are fitted on FIT. Whether the adjustment ships at all is decided
on TUNE, against the same two metrics Stage 2 used, so this is judged the way
the serve-rate model itself was judged:

* serve-points-won MAE — does it predict the observable it directly models?
* match-winner log-loss through the engine — does that translate into better
  match predictions?

The adjustment ships only if it improves BOTH. A feature that sharpens serve
prediction while making match prediction worse is not a win; it means the
engine is absorbing the shift somewhere it should not.

TEST AND HOLDOUT ARE NOT TOUCHED.

Run:  python -m scripts.fit_fatigue
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import constants as C
from model import fatigue as F
from model import ledger
from model import venue as V
from model.engine import match_distribution
from model.rules import FormatSpec
from scripts import panel as P

REPORT = C.STAGE_VALIDATIONS_DIR / "fatigue.md"


def _spec(key: tuple) -> FormatSpec:
    return FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                      tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                      final_tb_to=key[6], provenance="documented", source="fatigue")


def attach_context(pan: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Join per-side context features onto a panel, keyed by match_id."""
    ctx = F.build_context(matches).set_index("match_id")
    common = pan["match_id"].isin(ctx.index)
    pan = pan[common].copy()
    sub = ctx.loc[pan["match_id"]]
    for c in ctx.columns:
        pan[c] = sub[c].to_numpy()
    return pan


def metrics(pan: pd.DataFrame, params: F.FatigueParams) -> dict:
    """Serve-points MAE and match-winner log-loss under a set of coefficients."""
    xa, xb = F.design(pan, "w"), F.design(pan, "l")
    pa = F.adjust(pan["pa"].to_numpy(), xa, params)
    pb = F.adjust(pan["pb"].to_numpy(), xb, params)

    # Serve-points-won MAE, both sides pooled, weighted by points served.
    # ``w_spw`` is the observed fraction the rate model already computes.
    err = np.concatenate([np.abs(pa - pan["w_spw"].to_numpy()),
                          np.abs(pb - pan["l_spw"].to_numpy())])
    wts = np.concatenate([pan["w_svpt"].to_numpy(), pan["l_svpt"].to_numpy()])
    ok = np.isfinite(err) & np.isfinite(wts) & (wts > 0)
    mae = float(np.average(err[ok], weights=wts[ok]))

    lls = []
    for a, b, key in zip(pa, pb, pan["spec_key"]):
        d = match_distribution(round(float(a), 3), round(float(b), 3), _spec(key))
        lls.append(-np.log(max(d.p_a, 1e-9)))
    return {"spw_mae": mae, "logloss": float(np.mean(lls)), "n": len(pan)}


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s6 = fitted["stage_6"]
    vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                       enabled=s6["enabled"])

    matches = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    matches = matches[matches["in_scope"]]
    # Context is a backward scan, so a TUNE match needs FIT history — but it
    # never needs anything after itself. Cutting the frame at the TUNE
    # boundary keeps this script provably clear of TEST rather than merely
    # not using them.
    matches = matches[matches["date"] <= C.TUNE_END]
    assert matches["date"].max() <= C.TUNE_END

    train = P.build(splits=("fit", "burned_test"), venue_params=vp)
    tune = P.build(splits=("tune",), venue_params=vp)
    train = attach_context(train, matches)
    tune = attach_context(tune, matches)
    # Serve stats must be real for the fit; the rate model already excludes
    # suspect scores, and a match with no serve counters carries no signal.
    for name, d in (("train", train), ("tune", tune)):
        n0 = len(d)
        d.drop(d.index[~np.isfinite(d["w_svpt"]) | ~np.isfinite(d["l_svpt"])
                       | (d["w_svpt"] <= 0) | (d["l_svpt"] <= 0)], inplace=True)
        print(f"{name}: {len(d):,} matches ({n0 - len(d):,} dropped, no serve counters)")

    # --- fit coefficients on FIT only -------------------------------------
    x = np.vstack([F.design(train, "w"), F.design(train, "l")])
    p = np.concatenate([train["pa"].to_numpy(), train["pb"].to_numpy()])
    played = np.concatenate([train["w_svpt"].to_numpy(), train["l_svpt"].to_numpy()])
    won = np.concatenate([(train["w_spw"] * train["w_svpt"]).to_numpy(),
                          (train["l_spw"] * train["l_svpt"]).to_numpy()])
    params = F.fit(p, x, won, played)
    print(f"\nfitted on FIT: rest_days {params.rest_days:+.4f}, "
          f"load7 {params.load7:+.4f}, is_qualifier {params.is_qualifier:+.4f}")
    print("(positive = costs serve points)")

    # --- gate on TUNE ------------------------------------------------------
    off = F.FatigueParams(enabled=False)
    base = metrics(tune, off)
    adj = metrics(tune, params)
    print(f"\nTUNE baseline: spw MAE {base['spw_mae']:.5f}, "
          f"log-loss {base['logloss']:.5f}, n={base['n']:,}")
    print(f"TUNE adjusted: spw MAE {adj['spw_mae']:.5f}, "
          f"log-loss {adj['logloss']:.5f}")

    better_mae = adj["spw_mae"] < base["spw_mae"]
    better_ll = adj["logloss"] < base["logloss"]
    passed = better_mae and better_ll
    print(f"\nMAE improved: {better_mae}   log-loss improved: {better_ll}")
    print(f"Gate {'PASSED' if passed else 'FAILED'}")

    # Per-feature contribution, so a null result says WHICH feature was null.
    contrib = []
    for i, f in enumerate(F.FEATURES):
        solo = np.zeros(len(F.FEATURES))
        solo[i] = params.vector()[i]
        one = F.FatigueParams(*solo)
        r = metrics(tune, one)
        contrib.append({"feature": f, "coefficient": float(params.vector()[i]),
                        "tune_spw_mae": r["spw_mae"], "tune_logloss": r["logloss"],
                        "mae_gain": base["spw_mae"] - r["spw_mae"],
                        "logloss_gain": base["logloss"] - r["logloss"]})
    ctab = pd.DataFrame(contrib)
    print("\nper-feature, applied alone:")
    print(ctab.to_string(index=False))

    ledger.append(
        stage="fatigue", metric="tune_match_logloss",
        selected={"rest_days": params.rest_days, "load7": params.load7,
                  "is_qualifier": params.is_qualifier, "enabled": passed},
        tune_gain=float(base["logloss"] - adj["logloss"]),
        fit_rolling_origin_gains=[0.0],
        baseline="no context adjustment",
        tune_metric_value=float(adj["logloss"]),
        baselines={"tune_logloss": base["logloss"], "tune_spw_mae": base["spw_mae"]},
        frozen=bool(passed),
        notes="coefficients fitted on FIT, gated on TUNE; TEST and HOLDOUT untouched")

    lines = [
        "# Match-context (fatigue) adjustment\n",
        "Rest days, seven-day workload and entry route, applied as a logit-space "
        "shift to each side's serve-point probability. Coefficients fitted on "
        "FIT, gate decided on TUNE. **TEST and HOLDOUT were not touched.**\n",
        "\n## Why\n",
        "Stages 2-6 describe ability. None of them describe the state a player "
        "walks on court in. These are the cheapest features that do, and they "
        "come from columns already in the vendor files that nothing read.\n",
        "\n## Fitted coefficients (FIT)\n",
        f"\n| feature | coefficient |\n|---|---:|\n"
        f"| rest_days (deficit) | {params.rest_days:+.4f} |\n"
        f"| load7 | {params.load7:+.4f} |\n"
        f"| is_qualifier | {params.is_qualifier:+.4f} |\n",
        "\nPositive means the condition costs serve points.\n",
        "\n## Gate (TUNE)\n",
        f"\n| metric | baseline | adjusted | improved |\n|---|---:|---:|---|\n"
        f"| serve-points-won MAE | {base['spw_mae']:.5f} | {adj['spw_mae']:.5f} "
        f"| {better_mae} |\n"
        f"| match-winner log-loss | {base['logloss']:.5f} | {adj['logloss']:.5f} "
        f"| {better_ll} |\n",
        f"\nn = {base['n']:,} TUNE matches. Gate "
        f"**{'PASSED' if passed else 'FAILED'}** — both metrics must improve.\n",
        "\n## Per-feature, applied alone\n",
        "So that a null result names which feature was null rather than "
        "condemning the whole idea.\n",
        ctab.to_markdown(index=False),
        "\n\n## Caveat\n",
        "TUNE is 2024-01-01 .. 2025-06-30, read once by the original backtest "
        "at aggregate level. Not a pristine selection set.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")

    # Re-read before writing: this file is shared with the other fitting
    # scripts, and a read-at-start / write-at-end pair silently drops whatever
    # another run added in between.
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    fitted["fatigue"] = {
        "rest_days": params.rest_days, "load7": params.load7,
        "is_qualifier": params.is_qualifier, "enabled": bool(passed),
        "tune_logloss_baseline": base["logloss"], "tune_logloss": adj["logloss"],
        "tune_spw_mae_baseline": base["spw_mae"], "tune_spw_mae": adj["spw_mae"],
        "gate_passed": bool(passed),
    }
    C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))


if __name__ == "__main__":
    main()
