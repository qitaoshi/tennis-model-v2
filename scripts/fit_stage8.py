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
    for pa, pb, key, total, sets_w, sets_l in zip(
        pan["pa"], pan["pb"], pan["spec_key"], pan["total_games"],
        pan["n_sets"], pan["best_of"],
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
    train = P.build(splits=("fit", "tune"), venue_params=vp)
    train = train.sample(min(len(train), 20_000), random_state=C.MC_SEED)
    print(f"fitting maps on {len(train):,} FIT+TUNE matches")
    samples = collect(train, cp)
    maps = RC.fit_maps(samples)
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
        f"\nCorrections in force: tiebreak inflation "
        f"{cp.tiebreak_inflation:+.4f}, split sigma {cp.split_sigma:.3f}, "
        f"provenance scheme `{cp.scheme}`.\n",
        "\n## Calibration on the pre-cutoff TEST set\n",
        summary.to_markdown(index=False),
        "\n\n## Reliability after calibration\n",
        "\n".join(report_blocks),
        f"\n\n## Gate\n\nEvery family improved on both ECE and Brier: "
        f"**{all_ok}**. Gate **{'PASSED' if all_ok else 'FAILED'}**.\n",
    ]
    out = C.STAGE_VALIDATIONS_DIR / "stage_8.md"
    out.write_text("\n".join(lines) + "\n")

    if all_ok:
        maps.save()
        fitted["stage_8"] = {
            "families": sorted(maps.maps), "n_fitted": maps.n_fitted,
            "maps_path": str(RC.MAPS_PATH.relative_to(C.REPO_ROOT)),
            "test_ece_before": float(summary["ece_before"].mean()),
            "test_ece_after": float(summary["ece_after"].mean()),
            "test_window": [str(test["date"].min()), str(test["date"].max())],
            "test_touched_once": True,
        }
        C.FITTED_PARAMS_PATH.write_text(json.dumps(fitted, indent=1))
        print(f"\nmaps saved to {RC.MAPS_PATH}")
    print(f"\nGate {'PASSED' if all_ok else 'FAILED'}; wrote {out}")


if __name__ == "__main__":
    main()
