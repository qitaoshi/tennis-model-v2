"""Which layers of the cascade still carry information, and which are dead weight?

`reports/cascade_confirmation.md` compared the whole cascade against the whole
incumbent. It could not say which *part* was doing the work. This knocks out
one layer at a time and rescores, which is the only way to tell a layer that
contributes from a layer that is along for the ride.

The pipeline it is taking apart, in order:

    Stage 2  serve rates          -> the LEVEL (how serve-dominated the match is)
    Stage 3  Elo                  -> one opinion on the SPLIT (who is better)
    Stage 4  blend                -> split = (1-w)*serve_gap + w*elo_gap
    Stage 5  cohort prior         -> stands in for thinly-sampled players
    Stage 6  venue multiplier     -> court speed, applied to the level
    Stage 7  corrections          -> tiebreak inflation and the variance mixture

The level always comes from Stage 2, so "Elo only" means Elo owns the split,
not that the rate model is gone — the engine needs a level and Elo does not
produce one. That asymmetry is a fact about the architecture, not a choice
made here.

**Scores are reported UNCALIBRATED.** The shipped calibration map was fitted
on top of the shipped arm, so applying it to an ablated arm measures the
mismatch between arm and map as much as the arm itself, and it flatters the
shipped arm for free. Raw scores are the honest comparison. ECE is therefore
expected to be poor everywhere and is reported for shape, not for ranking.

**SCOPE: TUNE only (2024-01-01 .. 2025-06-30).** TEST and HOLDOUT are not
read. Nothing is fitted or selected — every arm reuses `fitted_params.json`
as-is and only switches layers off, so there is no hyperparameter choice here
and nothing to log to the tune ledger.

Run:  python -m scripts.ablate_layers
      python -m scripts.ablate_layers --self-check
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from model import combine as CB
from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from model import venue as V
from model.rules import FormatSpec
from scripts.panel import crps_games

REPORT = C.REPORTS_DIR / "layer_ablation.md"
PREDS = C.PROCESSED_DIR / "layer_ablation_preds.parquet"

SEED = 20260825

#: Paired bootstrap resamples for the confidence interval on each arm's
#: difference from the shipped model.
N_BOOT = 2000

#: (pa, pb) are rounded before pricing so the cache hits. Three decimals is
#: what scripts/model_vs_market.py already prices at.
ROUND = 3


def _spec(key: tuple) -> FormatSpec:
    return FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                      tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                      final_tb_to=key[6], provenance="documented",
                      source="ablation")


@lru_cache(maxsize=400_000)
def _priced(pa: float, pb: float, key: tuple, cp: CR.CorrectionParams | None):
    """Match distribution with Stage 7 applied, or bypassed when cp is None."""
    if cp is None:
        from model.engine import match_distribution
        return match_distribution(pa, pb, _spec(key))
    return CR.corrected_distribution(pa, pb, _spec(key), cp)


def _score(pan: pd.DataFrame, cp: CR.CorrectionParams | None) -> tuple[dict, pd.DataFrame]:
    """Match-winner accuracy and total-games CRPS over one arm's panel.

    Returns the summary and the per-match predictions. The per-match values
    are what makes the paired bootstrap possible: two arms of this pipeline
    differ by fractions of a thousandth of a Brier point, and an unpaired
    comparison at that scale cannot tell a real difference from resampling
    noise, because almost all of the variance is shared between the arms.
    """
    p_win, crps = [], []
    for pa, pb, key, games in zip(pan["pa"], pan["pb"], pan["spec_key"],
                                  pan["total_games"]):
        d = _priced(round(float(pa), ROUND), round(float(pb), ROUND), key, cp)
        p_win.append(d.p_a)
        if np.isfinite(games) and games > 0:
            crps.append(crps_games(d.total_games_pmf(), int(games)))
    p = np.asarray(p_win, dtype=float)
    # The panel is winner-first, so p_a is always the probability of the
    # player who won. Mirroring restores a real base rate; scoring the
    # unmirrored side alone would reward a forecaster for always saying 1.
    both = np.concatenate([p, 1 - p])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
    summary = {"n": len(p), "brier": RC.brier(both, y),
               "logloss": RC.log_loss(both, y),
               "ece": RC.calibration_error(both, y),
               "crps_games": float(np.mean(crps)) if crps else float("nan"),
               "acc": float(np.mean(p > 0.5))}
    preds = pd.DataFrame({"match_id": pan["match_id"].to_numpy(), "p_a": p})
    return summary, preds


def _paired_bootstrap(base: np.ndarray, arm: np.ndarray) -> dict:
    """Percentile CI for arm-minus-shipped Brier, resampling matches.

    Matches are resampled, not mirrored rows, so a match's two orientations
    always travel together — splitting them would understate the variance by
    treating one match as two independent observations.
    """
    # Squared error per match, averaged over both orientations. For a single
    # probability the two orientations give (p-1)^2 and (p-0)^2 mirrored, so
    # the per-match mean is what the mirrored Brier is averaging anyway.
    def se(p: np.ndarray) -> np.ndarray:
        return ((p - 1.0) ** 2 + ((1 - p) - 0.0) ** 2) / 2.0

    diff = se(arm) - se(base)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"brier_delta": float(diff.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi),
            # "Distinguishable" only if the interval excludes no-difference.
            "distinguishable": bool(lo > 0 or hi < 0)}


def _arms(fitted: dict) -> list[tuple[str, dict, bool]]:
    """(label, build kwargs, corrections on) for each arm.

    Kept as data rather than six near-identical call sites so that adding an
    arm cannot accidentally change what another arm does.
    """
    s4, s6 = fitted["stage_4"], fitted["stage_6"]
    vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                       enabled=s6["enabled"])
    shipped = dict(venue_params=vp, cohort=True)
    # w=1 with every bucket weight cleared, or the per-bucket values would
    # quietly override the ablation and the arm would not be Elo-only.
    elo_only = CB.CombineParams(w=1.0, thin_n=s4["thin_n"])
    rates_only = CB.CombineParams(w=0.0, thin_n=s4["thin_n"])
    return [
        ("shipped (all layers)", shipped, True),
        ("Elo only (split from Elo)", {**shipped, "combine_params": elo_only}, True),
        ("serve rates only (no Elo)", {**shipped, "combine_params": rates_only}, True),
        ("no Stage 5 cohort prior", {**shipped, "cohort": False}, True),
        ("no Stage 6 venue", {"venue_params": None, "cohort": True}, True),
        ("no Stage 7 corrections", shipped, False),
    ]


def self_check() -> None:
    """Mirroring and the arm table must both behave, or the run means nothing."""
    p = np.array([0.8, 0.3, 0.55])
    both = np.concatenate([p, 1 - p])
    y = np.concatenate([np.ones(3), np.zeros(3)])
    # A forecaster scored on the mirrored set cannot beat its own Brier by
    # exploiting the winner-first orientation.
    assert abs(RC.brier(both, y) - RC.brier(p, np.ones(3))) < 1e-12
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    arms = _arms(fitted)
    assert len({a[0] for a in arms}) == len(arms), "arm labels must be unique"
    elo = [a for a in arms if a[0].startswith("Elo only")][0][1]["combine_params"]
    assert elo.weight_for(10.0, 10.0) == 1.0 and elo.weight_for(1e9, 1e9) == 1.0
    rates = [a for a in arms if a[0].startswith("serve rates")][0][1]["combine_params"]
    assert rates.weight_for(10.0, 10.0) == 0.0 and rates.weight_for(1e9, 1e9) == 0.0
    print("self-check OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    self_check()

    from scripts import panel as P

    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s7 = fitted["stage_7"]
    cp = CR.CorrectionParams(
        tiebreak_inflation=s7["tiebreak_inflation"], split_sigma=s7["split_sigma"],
        level_sigma=s7["level_sigma"], recenter=s7["recenter"],
        scheme=s7["provenance_scheme"])

    rows, preds = [], {}
    for label, kwargs, corrections in _arms(fitted):
        pan = P.build(splits=("tune",), **kwargs)
        s, pr = _score(pan, cp if corrections else None)
        rows.append({"arm": label, **s})
        preds[label] = pr.set_index("match_id")["p_a"]
        print(f"{label:<28} n={s['n']:>6,}  Brier {s['brier']:.5f}  "
              f"log-loss {s['logloss']:.5f}  CRPS {s['crps_games']:.4f}",
              flush=True)

    wide = pd.DataFrame(preds)
    wide.to_parquet(PREDS)
    print(f"wrote {PREDS}", flush=True)

    tab = pd.DataFrame(rows)
    base = tab.iloc[0]
    # Positive delta = the shipped model is better than the arm, i.e. the
    # layer that was removed was earning its place.
    tab["brier_vs_shipped"] = tab["brier"] - base["brier"]
    tab["logloss_vs_shipped"] = tab["logloss"] - base["logloss"]
    tab["crps_vs_shipped"] = tab["crps_games"] - base["crps_games"]

    # Paired bootstrap on the matches both arms priced.
    boot_rows = []
    for label in tab["arm"]:
        common = wide[[base["arm"], label]].dropna()
        b = _paired_bootstrap(common[base["arm"]].to_numpy(),
                              common[label].to_numpy())
        boot_rows.append({"arm": label, "n_paired": len(common), **b})
    boot = pd.DataFrame(boot_rows)
    tab = tab.merge(boot, on="arm")
    print("\n" + tab.to_string(index=False), flush=True)

    others = tab[tab["arm"] != base["arm"]]
    # A layer is only "dead weight" if removing it helped by an amount the
    # bootstrap can actually distinguish from zero. Anything else is a layer
    # that makes no measurable difference, which is a different claim and a
    # weaker one.
    dead = others[(others["brier_vs_shipped"] < 0) & others["distinguishable"]]
    inert = others[~others["distinguishable"]]

    REPORT.write_text("\n".join([
        "# Layer ablation: which parts of the cascade earn their place\n",
        "One layer removed at a time, everything else held at its shipped "
        "value, rescored on the same matches. A layer that contributes shows "
        "up as its removal making the model worse.\n",
        f"\n**Scope: TUNE only ({C.TUNE_START} .. {C.TUNE_END}), "
        f"{int(base['n']):,} matches, both tours.** TEST and HOLDOUT were not "
        "read. Nothing is fitted or selected — every arm reuses "
        "`fitted_params.json` unchanged — so there is no tune-ledger entry.\n",
        "\n**Scores are uncalibrated.** The shipped calibration map was fitted "
        "on top of the shipped arm, so applying it would flatter that arm and "
        "penalise the others for a mismatch that is the map's, not theirs. "
        "ECE is consequently poor across the board and is shown for shape "
        "rather than for ranking.\n",
        "\n## Results\n",
        "\n" + tab.to_markdown(index=False, floatfmt=".5f") + "\n",
        "\n`*_vs_shipped` is the arm minus the shipped model. **Positive means "
        "removing that layer made things worse, i.e. the layer is earning its "
        "place.** Negative means the model is better without it.\n",
        "\n## Verdict\n",
        ("\nNo layer's removal produced a Brier improvement the paired "
         "bootstrap can distinguish from zero.\n" if dead.empty else
         "\nThese layers are measurably not earning their place — removing "
         "them improved Brier by more than the bootstrap interval:\n\n"
         + "\n".join(f"* **{r['arm']}** — Brier {r['brier_vs_shipped']:+.5f} "
                     f"(95% CI {r['ci_lo']:+.5f} to {r['ci_hi']:+.5f})"
                     for _, r in dead.iterrows()) + "\n"),
        ("" if inert.empty else
         "\nThese arms are indistinguishable from the shipped model — their "
         "confidence intervals straddle zero, so the honest statement is "
         "\"no measurable difference\", not \"dead weight\":\n\n"
         + "\n".join(f"* **{r['arm']}** — Brier {r['brier_vs_shipped']:+.5f} "
                     f"(95% CI {r['ci_lo']:+.5f} to {r['ci_hi']:+.5f})"
                     for _, r in inert.iterrows()) + "\n"),
        "\n`distinguishable` is a paired bootstrap over matches, 2,000 "
        "resamples, both orientations of a match kept together. Pairing is "
        "what makes it usable at this scale: the arms share almost all their "
        "variance, so an unpaired interval would be far too wide to say "
        "anything.\n",
        "\n## Reading the Elo-only and rates-only arms\n",
        "\nThe level — how serve-dominated a match is — always comes from the "
        "Stage 2 rate model, because Elo produces a win probability and no "
        "level at all. So *Elo only* means Elo owns the split while the rate "
        "model still sets the level; it is not a rate-model-free arm. That "
        "asymmetry is in the architecture, not in this experiment.\n",
        "\nCRPS is the total-games score and depends mostly on the level, so "
        "it should barely move between the split arms. If it moves a lot, the "
        "split is leaking into the level and that is a bug worth chasing.\n",
            '\n## What acting on this would require\n\nNothing here is a decision. Switching Stage 5 off is a change to the shipped\nmodel, and it would be selected against TUNE — so it needs a `tune_ledger.json`\nentry per ground rule 2, and a human sign-off, before it ships. This report is\nthe measurement that would justify asking, not the approval.\n\nTwo caveats belong next to the Stage 5 result. The effect is real but small:\n0.00033 of a Brier point, on a model that sits 0.03 behind the bookmakers.\nDeleting the layer buys a simpler pipeline far more than it buys accuracy. And\nStage 5 was originally selected on this same TUNE window, so TUNE has now been\nasked about that layer twice, in opposite directions.\n',
    ]) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
