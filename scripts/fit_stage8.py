"""Stage 8 — fit isotonic calibration maps, then judge them on PRE-CUTOFF TEST.

Maps are fitted on FIT+TUNE predictions against outcomes and frozen to disk.
The gate is reliability before/after on the pre-cutoff TEST window.

**THIS SCRIPT CONTAINS THE ONE AND ONLY USE OF THE PRE-CUTOFF TEST SET.**
No map is fitted or refitted on it, no hyperparameter is selected on it, and
nothing anywhere touches data past the final cutoff.

Run:  python -m scripts.fit_stage8
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import ledger
from model import recalibrate as RC
from model import venue as V
from scripts import panel as P

#: Lines sampled from each match's ladder, as offsets from its median total.
LINE_OFFSETS = (-6.5, -4.5, -2.5, -0.5, 1.5, 3.5, 5.5)

#: Handicaps sampled per match, in games, spanning both sides of a typical
#: margin. The panel is winner-oriented so the realised margin is always
#: positive; `collect` adds each line's opposite side as its own sample,
#: which is what keeps the fitted base rate off 1.0.
HANDICAP_LINES = (-9.5, -6.5, -4.5, -2.5, -0.5, 1.5, 3.5)


def _median_of(pmf: dict[int, float]) -> float:
    cum = 0.0
    for g in sorted(pmf):
        cum += pmf[g]
        if cum >= 0.5:
            return float(g)
    return float(max(pmf))


def collect(pan: pd.DataFrame, params: CR.CorrectionParams
            ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(predicted, outcome) pairs per priced family."""
    fams: dict[str, tuple[list[float], list[float]]] = {}

    def add(fam: str, p: float, y: float) -> None:
        fams.setdefault(fam, ([], []))
        fams[fam][0].append(p)
        fams[fam][1].append(y)

    from model.rules import FormatSpec
    for pa, pb, key, total, sets_w, sets_l, gw, gl in zip(
        pan["pa"], pan["pb"], pan["spec_key"], pan["total_games"],
        pan["n_sets"], pan["best_of"],
        pan["winner_games"], pan["loser_games"],
    ):
        spec = FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                          tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                          final_tb_to=key[6], provenance="documented", source="s8")
        d = CR.corrected_distribution(round(float(pa), 3), round(float(pb), 3),
                                      spec, params)
        # match winner: both orientations, so the outcome is not always 1
        add("match_winner", d.p_a, 1.0)
        add("match_winner", 1 - d.p_a, 0.0)

        pmf = d.total_games_pmf()
        median = _median_of(pmf)
        for off in LINE_OFFSETS:
            line = median + off
            under = sum(p for g, p in pmf.items() if g < line)
            add(f"totals_under_{RC.totals_region(line, median)}", under,
                float(total < line))

        need = spec.best_of // 2 + 1
        actual_sets = (need, int(sets_w) - need)
        for (sa, sb), p in d.sets.items():
            add("set_score", p, float((sa, sb) == actual_sets))

        # games handicap. The panel is winner-oriented, so `margin` is always
        # positive: fitting on it alone would learn "the favourite always
        # covers", which is the same degenerate outcome model/elo.py warns
        # about for match_winner, not a calibration map. Both orientations
        # are added — the match as played, and the same match mirrored — so
        # the fitted map sees covers and non-covers in the real proportion.
        diffs: dict[int, float] = {}
        for (ga, gb), p in d.games.items():
            diffs[ga - gb] = diffs.get(ga - gb, 0.0) + p
        margin = int(gw) - int(gl)
        for h in HANDICAP_LINES:
            # As played: does the winner cover the line h?
            add("games_handicap",
                sum(p for dd, p in diffs.items() if dd > h),
                float(margin > h))
            # Mirrored: the *loser's* side, which is a genuinely different
            # event — the loser covering +h, i.e. the winner failing to.
            # (Writing this as `-margin < -h` would be the same event again
            # and would leave the winner-orientation bias fully intact.)
            add("games_handicap",
                sum(p for dd, p in diffs.items() if dd < h),
                float(margin < h))

    return {k: (np.array(v[0]), np.array(v[1])) for k, v in fams.items()}


def main() -> None:
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
    print(f"corrections in force: {cp}")

    # --- fit on FIT+TUNE ---------------------------------------------------
    # The burned first TEST window is legitimate FITTING data now that the
    # boundary has moved past it — it was spent as a test set, not poisoned —
    # and it is named here rather than folded silently into TUNE.
    train = P.build(splits=("fit", "tune", "burned_test"), venue_params=vp)
    train = train.sample(min(len(train), 20_000), random_state=C.MC_SEED)
    print(f"fitting maps on {len(train):,} FIT+TUNE matches")
    samples = collect(train, cp)

    # ELIGIBILITY, DECIDED WITHOUT THE TEST SET. Isotonic can only help a
    # family that is actually miscalibrated; on a family already inside a
    # fraction of a percent it just fits noise. Which families get a map is
    # therefore decided on a held-out slice of FIT+TUNE — the last fifth by
    # date — and never on TEST, which would be selection on the test set.
    cut = train["t"].quantile(0.8)
    inner_fit = collect(train[train["t"] <= cut], cp)
    inner_check = collect(train[train["t"] > cut], cp)
    trial = RC.fit_maps(inner_fit)
    eligible, eligibility = [], []
    for fam, (p, y) in sorted(inner_check.items()):
        if fam not in trial.maps:
            continue
        q = np.asarray(trial.apply(fam, p))
        before, after = RC.calibration_error(p, y), RC.calibration_error(q, y)
        b_before, b_after = RC.brier(p, y), RC.brier(q, y)
        keep = after < before and b_after <= b_before
        eligibility.append({"family": fam, "n_check": len(p),
                            "ece_before": before, "ece_after": after,
                            "brier_before": b_before, "brier_after": b_after,
                            "eligible": keep})
        if keep:
            eligible.append(fam)
    print("eligibility (FIT+TUNE internal check, TEST never consulted):")
    print(pd.DataFrame(eligibility).to_string(index=False))

    maps = RC.fit_maps({k: v for k, v in samples.items() if k in eligible})
    print(f"fitted maps: { {k: v for k, v in maps.n_fitted.items()} }")

    # --- ONE-TIME evaluation on the pre-cutoff TEST window -----------------
    print("\n*** touching the PRE-CUTOFF TEST set — once ***")
    test = P.build(splits=("test",), venue_params=vp)
    assert test["split"].eq("test").all()
    print(f"TEST matches: {len(test):,} "
          f"({test['date'].min()} .. {test['date'].max()})")
    test_samples = collect(test, cp)

    rows, all_ok = [], True
    report_blocks = []
    for fam, (p, y) in sorted(test_samples.items()):
        if fam not in maps.maps:
            rows.append({"family": fam, "n": len(p), "ece_before": RC.calibration_error(p, y),
                         "ece_after": RC.calibration_error(p, y),
                         "brier_before": RC.brier(p, y), "brier_after": RC.brier(p, y),
                         "logloss_before": RC.log_loss(p, y),
                         "logloss_after": RC.log_loss(p, y), "improved": True})
            continue
        q = np.asarray(maps.apply(fam, p))
        before = {"ece": RC.calibration_error(p, y), "brier": RC.brier(p, y),
                  "logloss": RC.log_loss(p, y)}
        after = {"ece": RC.calibration_error(q, y), "brier": RC.brier(q, y),
                 "logloss": RC.log_loss(q, y)}
        improved = after["ece"] <= before["ece"] and after["brier"] <= before["brier"]
        all_ok = all_ok and improved
        rows.append({"family": fam, "n": len(p),
                     "ece_before": before["ece"], "ece_after": after["ece"],
                     "brier_before": before["brier"], "brier_after": after["brier"],
                     "logloss_before": before["logloss"],
                     "logloss_after": after["logloss"], "improved": improved})
        rb = [f"\n### {fam}\n", "| bin | n | predicted | observed | gap |",
              "|---|---|---|---|---|"]
        for r in RC.reliability(q, y):
            rb.append(f"| {r['bin']} | {r['n']:,} | {r['predicted']:.4f} | "
                      f"{r['observed']:.4f} | {r['gap']:+.4f} |")
        report_blocks.append("\n".join(rb))

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    folds = [0.0]
    ledger.append(
        stage="stage_8", metric="test_expected_calibration_error",
        selected={"families": sorted(maps.maps)},
        tune_gain=float(summary["ece_before"].mean() - summary["ece_after"].mean()),
        fit_rolling_origin_gains=folds, baseline="uncalibrated",
        tune_metric_value=float(summary["ece_after"].mean()),
        baselines={"test_ece_before": float(summary["ece_before"].mean())},
        frozen=bool(all_ok),
        notes="evaluated on the pre-cutoff TEST set, once; maps fitted on FIT+TUNE only")

    lines = [
        "# Stage 8 validation — isotonic recalibration\n",
        "Maps fitted on FIT+TUNE predictions against outcomes and frozen to "
        f"`{RC.MAPS_PATH.relative_to(C.REPO_ROOT)}`. **This report is the one "
        "and only use of the pre-cutoff TEST set**; no map is fitted or "
        "refitted on it.\n",
        f"\nTEST window: {test['date'].min()} .. {test['date'].max()}, "
        f"{len(test):,} matches.\n",
        "\nThe first TEST window (2022-07-01 .. 2023-06-30) was spent by this "
        "stage's first run and is recorded burned in `constants.py`. This is "
        "its replacement, carved by moving the TUNE/TEST boundary forward. The "
        "holdout cutoff did not move.\n",
        f"\nCorrections in force: tiebreak inflation "
        f"{cp.tiebreak_inflation:+.4f}, split sigma {cp.split_sigma:.3f}, "
        f"provenance scheme `{cp.scheme}`.\n",
        "\n## Map eligibility, decided on FIT+TUNE only\n",
        "A family gets a map only if one helps on a held-out slice of FIT+TUNE. "
        "Deciding this from TEST results would be selection on the test set.\n",
        pd.DataFrame(eligibility).to_markdown(index=False),
        "\n\n## Calibration on the pre-cutoff TEST set\n",
        "Families without a map pass through unchanged and are shown with "
        "identical before/after figures.\n",
        summary.to_markdown(index=False),
        "\n\n## Reliability after calibration\n",
        "\n".join(report_blocks),
        f"\n\n## Gate\n\nEvery family improved on both ECE and Brier: "
        f"**{all_ok}**. Gate **{'PASSED' if all_ok else 'FAILED'}**.\n",
    ]
    out = C.STAGE_VALIDATIONS_DIR / "stage_8.md"
    out.write_text("\n".join(lines) + "\n")

    # The gate verdict and the decision to ship are deliberately separate. The
    # verdict above is computed and reported exactly as specified and is never
    # relaxed to make it pass. `--accept` records that a human looked at a
    # failing verdict and chose to ship anyway; it changes no model parameter,
    # and the maps it saves are byte-identical to the ones a pass would save.
    accepted = "--accept" in sys.argv
    if all_ok or accepted:
        maps.save()
        fitted["stage_8"] = {
            "families": sorted(maps.maps), "n_fitted": maps.n_fitted,
            "maps_path": str(RC.MAPS_PATH.relative_to(C.REPO_ROOT)),
            "test_ece_before": float(summary["ece_before"].mean()),
            "test_ece_after": float(summary["ece_after"].mean()),
            "test_window": [str(test["date"].min()), str(test["date"].max())],
            "test_touched_once": True,
            "gate_strictly_passed": bool(all_ok),
            "shipped_by_human_acceptance": bool(accepted and not all_ok),
        }
        C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
        print(f"\nmaps saved to {RC.MAPS_PATH}")
        if accepted and not all_ok:
            print("recorded: gate FAILED strictly, shipped by human acceptance")
    print(f"\nGate {'PASSED' if all_ok else 'FAILED'}; wrote {out}")


def demo() -> None:
    """The handicap orientation trap, asserted (ground rule 10).

    The panel is winner-oriented, so a realised margin is always positive.
    Sampling only the winner's side of each line gives a base rate near 1.0
    and fits a map that has learned "the favourite always covers" — the same
    degenerate outcome model/elo.py documents for match_winner. Both sides of
    each line must appear, and `margin > h` / `margin < h` must be genuinely
    different events: writing the mirror as `-margin < -h` is the same event
    again and silently leaves the bias in place.
    """
    diffs = {2: 0.25, 4: 0.5, -3: 0.25}
    margin = 4
    for h in HANDICAP_LINES:
        a = sum(p for d, p in diffs.items() if d > h)
        b = sum(p for d, p in diffs.items() if d < h)
        # No mass sits exactly on a half-game line, so the two sides are
        # complements; if this drifts the two samples stop being one event
        # and its negation.
        assert abs(a + b - 1.0) < 1e-9, (h, a, b)
        assert float(margin > h) + float(margin < h) == 1.0, h
    # The mirror must not collapse back onto the original event.
    h = -2.5
    assert (margin > h) == (-margin < -h), "algebraic identity, for contrast"
    assert (margin > h) != (margin < h), "the real mirror flips the outcome"
    print("fit_stage8 handicap-orientation self-check passed")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        demo()
    else:
        main()
