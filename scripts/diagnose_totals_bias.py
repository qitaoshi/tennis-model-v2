"""Where does the total-games over-prediction come from?

`reports/totals_ladder_check.md` found that at all seven rungs of the totals
ladder the actual under-rate exceeds the predicted one, by 0.5-3.5pp with no
exceptions. Seven out of seven in one direction is a location shift: the
model expects more games than matches actually produce.

Total games is a function of two things, and they need opposite fixes:

* the serve probabilities (pa, pb) that Stages 2-6 produce, and
* the engine that turns them into a games distribution.

If the serve rates are biased high, the engine is innocent and the fix is
upstream — the model thinks players hold more often than they do, and holds
make long sets. If the serve rates are unbiased and the games are still too
long, the rates are right and the IID POINT ASSUMPTION is wrong: real breaks
cluster (a player who drops serve is more likely to drop the next one), while
iid points spread breaks evenly and manufacture 6-4s where reality produces
6-2s. That is Stage 7's mandate, not Stage 2's.

This measures both on the same matches, so the answer is a comparison rather
than a guess.

Measured on TUNE. Nothing is fitted. TEST and HOLDOUT are not touched.

Run:  python -m scripts.diagnose_totals_bias
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import venue as V
from model.rules import FormatSpec
from scripts import panel as P

REPORT = C.REPORTS_DIR / "totals_bias_diagnosis.md"


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s6, s7 = fitted["stage_6"], fitted["stage_7"]
    vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                       enabled=s6["enabled"])
    cp = CR.CorrectionParams(
        tiebreak_inflation=s7["tiebreak_inflation"], split_sigma=s7["split_sigma"],
        level_sigma=s7["level_sigma"], recenter=s7["recenter"],
        scheme=s7["provenance_scheme"])

    pan = P.build(splits=("tune",), venue_params=vp)
    rec = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet").set_index("match_id")
    for col in ("level_label", "surface", "tour"):
        pan[col] = rec.loc[pan["match_id"], col].to_numpy()
    pan = pan.sample(min(len(pan), 6_000), random_state=C.MC_SEED)
    print(f"TUNE panel: {len(pan):,} matches")

    exp_games, exp_no_corr = [], []
    for r in pan.itertuples(index=False):
        k = r.spec_key
        spec = FormatSpec(best_of=k[0], games_to_win_set=k[1], tb_at=k[2],
                          tb_to=k[3], final_set=k[4], final_tb_at=k[5],
                          final_tb_to=k[6], provenance="documented", source="dtb")
        pa, pb = round(float(r.pa), 3), round(float(r.pb), 3)
        pmf = CR.corrected_distribution(pa, pb, spec, cp).total_games_pmf()
        exp_games.append(sum(g * p for g, p in pmf.items()))
        # Same thing with Stage 7 switched off, to see whether the corrections
        # are causing the shift or already partly absorbing it.
        raw = CR.corrected_distribution(
            pa, pb, spec, CR.CorrectionParams(
                tiebreak_inflation=0.0, split_sigma=0.0, level_sigma=0.0,
                recenter=True, scheme=cp.scheme)).total_games_pmf()
        exp_no_corr.append(sum(g * p for g, p in raw.items()))
    pan["exp_games"] = exp_games
    pan["exp_games_no_corr"] = exp_no_corr

    # --- 1. the games bias -------------------------------------------------
    d = pan["exp_games"] - pan["total_games"]
    d0 = pan["exp_games_no_corr"] - pan["total_games"]
    print(f"\nGAMES: predicted mean {pan['exp_games'].mean():.3f} vs actual "
          f"{pan['total_games'].mean():.3f} -> bias {d.mean():+.3f} games")
    print(f"  with Stage 7 off: bias {d0.mean():+.3f} games")

    # --- 2. the serve-rate bias -------------------------------------------
    # pa/pb are the model's serve-point probabilities; w_spw/l_spw are what
    # actually happened. If these agree, the rates are not the problem.
    pred = np.concatenate([pan["pa"], pan["pb"]])
    act = np.concatenate([pan["w_spw"], pan["l_spw"]])
    ok = np.isfinite(pred) & np.isfinite(act)
    spw_bias = float(np.mean(pred[ok] - act[ok]))
    print(f"\nSERVE POINTS WON: predicted {pred[ok].mean():.5f} vs actual "
          f"{act[ok].mean():.5f} -> bias {spw_bias:+.5f}")
    # Scale it: how many games would this much serve bias explain? A rough
    # sensitivity, not a claim - recompute expected games with pa/pb shifted
    # down by the measured bias and see how far the games bias moves.
    shifted = []
    for r in pan.head(1500).itertuples(index=False):
        k = r.spec_key
        spec = FormatSpec(best_of=k[0], games_to_win_set=k[1], tb_at=k[2],
                          tb_to=k[3], final_set=k[4], final_tb_at=k[5],
                          final_tb_to=k[6], provenance="documented", source="dtb")
        pmf = CR.corrected_distribution(round(float(r.pa) - spw_bias, 3),
                                        round(float(r.pb) - spw_bias, 3),
                                        spec, cp).total_games_pmf()
        shifted.append(sum(g * p for g, p in pmf.items()))
    head = pan.head(1500)
    resid = float(np.mean(np.array(shifted) - head["total_games"].to_numpy()))
    print(f"  removing that serve bias leaves a games bias of {resid:+.3f} "
          f"(was {float((head['exp_games'] - head['total_games']).mean()):+.3f})")

    # --- 3. by segment -----------------------------------------------------
    seg = []
    for name, col in (("level", "level_label"), ("surface", "surface"),
                      ("best_of", "best_of")):
        for v, g in pan.groupby(col):
            if len(g) < 150:
                continue
            gp = np.concatenate([g["pa"], g["pb"]])
            ga = np.concatenate([g["w_spw"], g["l_spw"]])
            m = np.isfinite(gp) & np.isfinite(ga)
            seg.append({"cut": name, "value": str(v), "n": len(g),
                        "games_bias": float((g["exp_games"] - g["total_games"]).mean()),
                        "spw_bias": float(np.mean(gp[m] - ga[m]))})
    segs = pd.DataFrame(seg)
    print("\nby segment (positive = model predicts too high):")
    print(segs.to_string(index=False))

    verdict = ("SERVE RATES — fix upstream in Stage 2"
               if abs(resid) < abs(d.mean()) * 0.4
               else "THE ENGINE / IID ASSUMPTION — fix in Stage 7")
    print(f"\nVERDICT: {verdict}")

    lines = [
        "# Where the total-games over-prediction comes from\n",
        "`totals_ladder_check.md` found the model expects more games than "
        "matches produce, at all seven rungs of the ladder with no "
        "exceptions. Total games is a function of the serve probabilities and "
        "of the engine that turns them into a distribution, and those need "
        "opposite fixes — so this measures both on the same matches.\n",
        "\nTUNE only. Nothing fitted.\n",
        "\n## The two candidates\n",
        f"\n| quantity | predicted | actual | bias |\n|---|---:|---:|---:|\n"
        f"| total games | {pan['exp_games'].mean():.3f} "
        f"| {pan['total_games'].mean():.3f} | {d.mean():+.3f} |\n"
        f"| total games, Stage 7 off | {pan['exp_games_no_corr'].mean():.3f} "
        f"| {pan['total_games'].mean():.3f} | {d0.mean():+.3f} |\n"
        f"| serve points won | {pred[ok].mean():.5f} | {act[ok].mean():.5f} "
        f"| {spw_bias:+.5f} |\n",
        f"\nShifting `pa`/`pb` down by the measured serve bias leaves a games "
        f"bias of {resid:+.3f}, against "
        f"{float((head['exp_games'] - head['total_games']).mean()):+.3f} "
        "before. If most of the games bias survives that shift, the serve "
        "rates are not the cause and the engine's iid point assumption is.\n",
        "\n## By segment\n",
        "Positive means the model predicts too high.\n",
        segs.to_markdown(index=False),
        f"\n\n## Verdict\n\n**{verdict}.**\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
