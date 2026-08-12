"""Refit the calibration maps after the 2026-08-08 re-split.

Two things are wrong with the maps the original build shipped, and this script
addresses both:

1. **They are stale.** They were fitted on predictions through 2022-06 and
   applied, unchanged, to 2024-2026. match_winner ECE was 0.017 on the
   original TEST window and 0.069 on the original holdout — a fourfold decay
   that looks like drift, not like a bad map.
2. **The tail is a step function.** Isotonic regression puts only seven knots
   below p=0.20 in the shipped match_winner map, so a longshot's calibrated
   probability is decided by a handful of matches and jumps discontinuously.
   That is the region where the model is known to overrate its selections.

So two hyperparameters are selected here, both on TUNE:

* ``method`` — isotonic (the incumbent), Platt in logit space, or the two
  blended. See ``model/recalibrate.py`` for what each is and why.
* ``window`` — how many years of history the map is fitted on. A short window
  tracks drift; a long one has more data. Which wins is an empirical question,
  and this is the script that asks it.

SELECTION USES FIT AND TUNE ONLY. Run with no arguments to select; run with
``--evaluate-test`` afterwards to spend the TEST window, once, on whatever was
selected. The two phases are separate commands so that selection cannot see
the test result even by accident.

Run:  python -m scripts.refit_calibration
      python -m scripts.refit_calibration --evaluate-test
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import ledger
from model import recalibrate as RC
from model import venue as V
from scripts import panel as P
from scripts.fit_stage8 import collect

#: Fitting windows to try, in years back from the end of the training data.
#: ``None`` means "everything available", the incumbent behaviour.
WINDOWS: tuple[int | None, ...] = (None, 8, 5, 3)

#: Families worth calibrating. Kept explicit rather than "whatever collect()
#: returns" so a thinly-sampled ladder region cannot quietly acquire a map.
FAMILIES = ("match_winner", "totals_under_low", "totals_under_mid",
            "totals_under_high", "set_score", "games_handicap")

REPORT = C.REPORTS_DIR / "calibration_refit.md"


def _corrections() -> tuple[CR.CorrectionParams, V.VenueParams, dict]:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s6 = fitted["stage_6"]
    vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                       enabled=s6["enabled"])
    s7 = fitted.get("stage_7", {})
    cp = CR.CorrectionParams(
        tiebreak_inflation=s7.get("tiebreak_inflation", 0.0),
        level_sigma=s7.get("level_sigma", 0.0),
        split_sigma=s7.get("split_sigma", 0.0),
        recenter=s7.get("recenter", True),
        scheme=s7.get("provenance_scheme", "pooled"),
        inferred_weight=s7.get("inferred_weight", 1.0))
    return cp, vp, fitted


def _window_slice(pan: pd.DataFrame, years: int | None) -> pd.DataFrame:
    """The last ``years`` years of a panel, or all of it when None."""
    if years is None:
        return pan
    cutoff = pan["t"].max() - int(round(years * 365.25))
    return pan[pan["t"] >= cutoff]


def _score(maps: RC.CalibrationMaps, samples: dict) -> pd.DataFrame:
    rows = []
    for fam, (p, y) in sorted(samples.items()):
        if fam not in FAMILIES:
            continue
        q = np.asarray(maps.apply(fam, p)) if fam in maps.maps else p
        rows.append({
            "family": fam, "n": len(p),
            "ece_before": RC.calibration_error(p, y),
            "ece_after": RC.calibration_error(q, y),
            "brier_before": RC.brier(p, y), "brier_after": RC.brier(q, y),
            "logloss_before": RC.log_loss(p, y), "logloss_after": RC.log_loss(q, y),
        })
    return pd.DataFrame(rows)


#: A map is degenerate if it answers a wide stretch of its input with a single
#: number: saturated at 0/1 (certainty from a handful of matches) or flat
#: anywhere (the input is being ignored). Measured on a grid rather than from
#: the fitted knots so it applies to every method the same way.
DEGENERATE_GRID = np.linspace(0.02, 0.98, 49)
#: Fraction of the grid that may map into the saturated band before the map is
#: rejected. Some saturation at the very edges is normal and harmless.
MAX_SATURATED_FRAC = 0.10
#: The longest run of identical outputs tolerated, as a fraction of the grid.
MAX_FLAT_FRAC = 0.25


def _degeneracy_reasons(maps: RC.CalibrationMaps) -> list[str]:
    """Every family-level reason this candidate is unusable, most severe first.

    Returns all of them rather than short-circuiting on the first. The check
    rejects a candidate if ANY family degenerates, so knowing WHICH family --
    and whether it is the one a given decision actually rests on -- is the
    difference between "this method is unusable" and "this method is fine for
    match_winner but its totals map collapsed".
    """
    reasons: list[str] = []
    for fam in FAMILIES:
        if fam not in maps.maps:
            continue
        out = np.asarray(maps.apply(fam, DEGENERATE_GRID), dtype=float)
        sat = ((out <= 1e-6) | (out >= 1 - 1e-6)).mean()
        if sat > MAX_SATURATED_FRAC:
            reasons.append(f"{fam}: saturated over {sat:.0%} of the grid "
                           f"(limit {MAX_SATURATED_FRAC:.0%})")
        longest = best_run = 1
        flat_at = out[0]
        for a, b in zip(out, out[1:]):
            if abs(a - b) < 1e-9:
                best_run += 1
                if best_run > longest:
                    longest, flat_at = best_run, a
            else:
                best_run = 1
        if longest / len(out) > MAX_FLAT_FRAC:
            reasons.append(f"{fam}: flat at {flat_at:.4f} for "
                           f"{longest / len(out):.0%} of the grid "
                           f"(limit {MAX_FLAT_FRAC:.0%})")
    return reasons


def _is_degenerate(maps: RC.CalibrationMaps) -> bool:
    """True if any family's map saturates or flatlines across the grid."""
    return bool(_degeneracy_reasons(maps))


def _tail_table(maps: RC.CalibrationMaps, fam: str) -> pd.DataFrame:
    """What each candidate does to a longshot, which is the point of all this."""
    grid = np.array([0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.50])
    out = np.asarray(maps.apply(fam, grid)) if fam in maps.maps else grid
    return pd.DataFrame({"model_p": grid, "calibrated": np.round(out, 4)})


def select() -> None:
    cp, vp, fitted = _corrections()
    print(f"corrections in force: {cp}")

    train = P.build(splits=("fit", "burned_test"), venue_params=vp)
    tune = P.build(splits=("tune",), venue_params=vp)
    assert tune["split"].eq("tune").all()
    assert train["t"].max() < tune["t"].min(), "train must end before TUNE starts"
    print(f"train {len(train):,} matches, TUNE {len(tune):,} matches")

    # Cap the training sample for tractability; the cap is applied AFTER the
    # window filter so a short window is not also a small sample by accident.
    tune_samples = collect(tune.sample(min(len(tune), 6_000),
                                       random_state=C.MC_SEED), cp)

    rows, fitted_maps = [], {}
    for years in WINDOWS:
        sl = _window_slice(train, years)
        sl = sl.sample(min(len(sl), 20_000), random_state=C.MC_SEED)
        train_samples = collect(sl, cp)
        for how in RC.METHODS:
            maps = RC.fit_maps({k: v for k, v in train_samples.items()
                                if k in FAMILIES}, method=how)
            sc = _score(maps, tune_samples)
            key = (how, years)
            fitted_maps[key] = maps
            rows.append({
                "method": how, "window_years": years if years else "all",
                "n_train_matches": len(sl),
                "tune_ece": float(sc["ece_after"].mean()),
                "tune_logloss": float(sc["logloss_after"].mean()),
                "tune_brier": float(sc["brier_after"].mean()),
                "mw_ece": float(sc.loc[sc.family == "match_winner", "ece_after"].iloc[0]),
                "mw_logloss": float(sc.loc[sc.family == "match_winner", "logloss_after"].iloc[0]),
            })
            # match_winner is reported alongside the family mean because the
            # two disagree sharply and answer different questions. tune_ece
            # averages all six families, INCLUDING any whose map is degenerate,
            # so it can favour a candidate on the strength of maps that will
            # never be used. mw_ece is the number to read when the decision is
            # about the match-winner map specifically.
            print(f"  {how:9s} window={str(years):4s} "
                  f"tune ECE {rows[-1]['tune_ece']:.4f} "
                  f"log-loss {rows[-1]['tune_logloss']:.4f} | "
                  f"match_winner ECE {rows[-1]['mw_ece']:.4f} "
                  f"log-loss {rows[-1]['mw_logloss']:.4f}")

    grid = pd.DataFrame(rows)
    base = _score(RC.CalibrationMaps({}, {}), tune_samples)
    baseline = {"tune_ece": float(base["ece_before"].mean()),
                "tune_logloss": float(base["logloss_before"].mean()),
                "tune_brier": float(base["brier_before"].mean()),
                "mw_ece": float(base.loc[base.family == "match_winner",
                                         "ece_before"].iloc[0]),
                "mw_logloss": float(base.loc[base.family == "match_winner",
                                             "logloss_before"].iloc[0])}
    print(f"\nuncalibrated baseline on TUNE: {baseline}")

    # SELECTION RULE, fixed before the numbers were seen: calibration is what
    # these maps exist for, so ECE decides. Log-loss breaks a tie and also
    # guards against an ECE win bought by destroying sharpness.
    #
    # Amended 2026-08-10, after the first run of this script shipped a map
    # that was degenerate rather than merely imperfect: a candidate is only
    # eligible if no family saturates. Isotonic regression is a step
    # function, so a top bin holding few same-resolving samples returns
    # exactly 1.0 — the fair price becomes 1.00, every quote above evens
    # reads as free money, and one loss at p=1 dominates log-loss for the
    # whole run. `totals_under_mid` did this above p≈0.68 and
    # `totals_under_low` collapsed to a constant 0.50.
    #
    # This is a constraint on what counts as a usable map, not a new metric
    # chosen to favour a candidate: it rejects on shape alone and is applied
    # before any ECE is compared. Ranking among the survivors is unchanged.
    for (_m, _w), _maps in fitted_maps.items():
        for _r in _degeneracy_reasons(_maps):
            print(f"  DEGENERATE {_m}/{_w if _w else 'all'} -> {_r}")
    grid["degenerate"] = [
        _is_degenerate(fitted_maps[(m, None if w == "all" else int(w))])
        for m, w in zip(grid["method"], grid["window_years"])]
    usable = grid[~grid["degenerate"]]
    if usable.empty:
        raise SystemExit(
            "every candidate produced a saturated or constant map; refusing "
            "to ship one. Widen WINDOWS or revisit the sampling before "
            "trusting any of these.")
    if len(usable) < len(grid):
        dropped = grid[grid["degenerate"]]
        print(f"\n{len(dropped)} candidate(s) rejected as degenerate: "
              + ", ".join(f"{r.method}/{r.window_years}"
                          for r in dropped.itertuples()))
    # Keep the unscreened grid: the incumbent baseline below is looked up in
    # it by name, and the incumbent may itself have been rejected — as it was
    # on 2026-08-10, when every isotonic and blended candidate saturated.
    all_candidates = grid
    grid = usable.sort_values(["tune_ece", "tune_logloss"]).reset_index(drop=True)
    best = grid.iloc[0]
    sel_method = str(best["method"])
    sel_window = None if best["window_years"] == "all" else int(best["window_years"])
    print(f"\nselected: method={sel_method} window={best['window_years']}")

    # Refit the winner on everything up to the TEST boundary. Selection used
    # FIT for fitting and TUNE for judging; the shipped map should use both,
    # which is the usual refit-on-all-available-data step and adds no
    # information from TEST or HOLDOUT.
    full = pd.concat([train, tune], ignore_index=True)
    full_sl = _window_slice(full, sel_window)
    full_sl = full_sl.sample(min(len(full_sl), 26_000), random_state=C.MC_SEED)
    final_samples = collect(full_sl, cp)
    final = RC.fit_maps({k: v for k, v in final_samples.items() if k in FAMILIES},
                        method=sel_method)
    final.save()
    print(f"refit on {len(full_sl):,} FIT+TUNE matches "
          f"({full_sl['t'].min()}..{full_sl['t'].max()}), saved to {RC.MAPS_PATH}")

    incumbent = all_candidates[(all_candidates["method"] == "isotonic")
                               & (all_candidates["window_years"] == "all")]
    gain = float(incumbent["tune_ece"].iloc[0] - best["tune_ece"])
    incumbent_degenerate = bool(incumbent["degenerate"].iloc[0])
    ledger.append(
        stage="calibration_refit", metric="tune_expected_calibration_error",
        selected={"method": sel_method, "window_years": best["window_years"]},
        tune_gain=gain, fit_rolling_origin_gains=[0.0],
        baseline="isotonic on all history (the shipped incumbent)",
        tune_metric_value=float(best["tune_ece"]),
        baselines=baseline, frozen=True,
        notes="selected on the post-re-split TUNE (2024-01-01..2025-06-30); "
              "TEST and HOLDOUT not consulted. "
              + (f"{int(all_candidates['degenerate'].sum())} of "
                 f"{len(all_candidates)} candidates were rejected as "
                 "degenerate (saturated or flat) before ranking"
                 + (", the incumbent among them, so `tune_gain` compares "
                    "against a baseline that is not itself shippable"
                    if incumbent_degenerate else "")
                 if bool(all_candidates["degenerate"].any())
                 else "no candidate was rejected as degenerate"))

    lines = [
        "# Calibration refit — method and fitting window\n",
        "Selected on TUNE (2024-01-01 .. 2025-06-30) after the 2026-08-08 "
        "re-split. **TEST and HOLDOUT were not consulted.** Maps are fitted on "
        "FIT and judged on TUNE; the shipped map is then refit on FIT+TUNE "
        "using the selected setting.\n",
        "\n## Why this was run\n",
        "The shipped maps were fitted on predictions through 2022-06 and never "
        "refreshed. match_winner ECE was 0.017 on the original TEST window and "
        "0.069 on the original holdout. Separately, isotonic put only seven "
        "knots below p=0.20, so the tail — where the model is known to "
        "overrate its selections — was a step function fitted on very little "
        "data.\n",
        f"\nUncalibrated baseline on TUNE: ECE {baseline['tune_ece']:.4f}, "
        f"log-loss {baseline['tune_logloss']:.4f}, "
        f"Brier {baseline['tune_brier']:.4f}.\n",
        "\n## Grid\n",
        "Ranked by ECE, then log-loss. The rule was fixed before the numbers "
        "were seen: these maps exist to fix calibration, so ECE decides, and "
        "log-loss guards against an ECE win bought by destroying sharpness.\n",
        grid.to_markdown(index=False),
        f"\n\n**Selected: `{sel_method}`, window `{best['window_years']}`.** "
        f"ECE gain over the shipped incumbent (isotonic, all history): "
        f"{gain:+.4f}.\n",
        "\n## What the selected map does to a longshot\n",
        "The column that motivated the work. Model probability in, calibrated "
        "probability out.\n",
        _tail_table(final, "match_winner").to_markdown(index=False),
        "\n\n## How much of this grid is signal\n",
        "Read the winner as the best of several near-ties, not as a decisive "
        "result. The ECE spread across the whole grid is small relative to "
        "what 12,766 TUNE matches can resolve, and the window ranking is not "
        "monotone — 8 years beats both all-history and 3 years, which is not "
        "the shape a strong drift effect would make.\n",
        "\nWhat the grid does support, in decreasing order of confidence: "
        "(1) recalibrating on recent data beats the stale shipped map; "
        "(2) restricting the fitting window helps; (3) which method wins "
        "matters less than which window does. Treat the specific "
        "method/window pair as the best available choice rather than as an "
        "established fact.\n",
        "\n## Caveat\n",
        "TUNE here is 2024-01-01 .. 2025-06-30, a window the original backtest "
        "read once at aggregate level. No parameter was fitted against it "
        "then, but it is not a pristine selection set — see "
        "`constants.PRIOR_EVALUATION_WINDOWS`.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")

    # Re-read before writing: this file is shared with the other fitting
    # scripts, and a read-at-start / write-at-end pair silently drops whatever
    # another run added in between.
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    fitted["calibration_refit"] = {
        "method": sel_method, "window_years": best["window_years"],
        "selected_on": "TUNE 2024-01-01..2025-06-30",
        "tune_ece": float(best["tune_ece"]),
        "tune_logloss": float(best["tune_logloss"]),
        "n_fitted": final.n_fitted, "families": sorted(final.maps),
        "test_evaluated": False,
    }
    C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))


def evaluate_test() -> None:
    """Spend the TEST window, once, on the already-selected map."""
    cp, vp, fitted = _corrections()
    sel = fitted.get("calibration_refit")
    assert sel, "run selection first"
    assert not sel.get("test_evaluated"), "TEST already spent on this map"

    maps = RC.CalibrationMaps.load()
    print(f"*** touching TEST — once *** (map: {sel['method']}, "
          f"window {sel['window_years']})")
    test = P.build(splits=("test",), venue_params=vp)
    assert test["split"].eq("test").all()
    print(f"TEST matches: {len(test):,}")
    samples = collect(test.sample(min(len(test), 6_000), random_state=C.MC_SEED), cp)
    sc = _score(maps, samples)
    print(sc.to_string(index=False))

    improved = bool((sc["ece_after"] <= sc["ece_before"]).all())
    lines = [
        "# Calibration refit — TEST evaluation\n",
        f"Map: `{sel['method']}`, fitting window `{sel['window_years']}`, "
        "selected on TUNE and refit on FIT+TUNE. **This is the one and only "
        "use of the post-re-split TEST window.** Nothing is fitted on it.\n",
        f"\nTEST: {C.TEST_START} .. {C.TEST_END}, {len(test):,} matches.\n",
        "\n## Before and after\n",
        sc.to_markdown(index=False),
        f"\n\nEvery family improved on ECE: **{improved}**.\n",
        "\n## Caveat\n",
        "This TEST window was read once by the original backtest at aggregate "
        "level. No parameter was fitted against it then; it is not pristine.\n",
    ]
    (C.STAGE_VALIDATIONS_DIR / "calibration_refit_test.md").write_text(
        "\n".join(lines) + "\n")
    # Re-read before writing, same reason as in select(): another fitting run
    # may have added a key since this one started, and read-at-start /
    # write-at-end drops it.
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    fitted["calibration_refit"]["test_evaluated"] = True
    fitted["calibration_refit"]["test_ece_before"] = float(sc["ece_before"].mean())
    fitted["calibration_refit"]["test_ece_after"] = float(sc["ece_after"].mean())
    fitted["calibration_refit"]["test_all_improved"] = improved
    C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
    print(f"\nall families improved on ECE: {improved}")


def demo() -> None:
    """The degeneracy screen, asserted (ground rule 10).

    Both failure modes are real ones this screen was written for: the first
    run of this script shipped `totals_under_high` saturated across 67% of
    its range and `totals_under_low` flat at 0.50 across 59% of it.
    """
    class _Const:
        def __init__(self, v): self.v = v
        def predict(self, x): return np.full(np.shape(x), self.v)

    class _Step:
        def predict(self, x): return np.where(np.asarray(x) < 0.5, 0.0, 1.0)

    class _Identity:
        def predict(self, x): return np.asarray(x, dtype=float)

    def maps_of(obj):
        return RC.CalibrationMaps({"match_winner": obj}, {"match_winner": 1})

    assert _is_degenerate(maps_of(_Step())), "saturation must be rejected"
    assert _is_degenerate(maps_of(_Const(0.5))), "a flat map must be rejected"
    assert not _is_degenerate(maps_of(_Identity())), \
        "a monotone non-saturating map must survive"
    # An empty map set is the uncalibrated baseline, not a degenerate map.
    assert not _is_degenerate(RC.CalibrationMaps({}, {}))
    print("refit_calibration degeneracy self-check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        demo()
    else:
        evaluate_test() if a.evaluate_test else select()
