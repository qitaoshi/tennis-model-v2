"""Can more Stage 7 correction remove the total-games over-prediction?

`reports/totals_bias_diagnosis.md` established:

* the model expects 0.428 more games than matches produce;
* with Stage 7 switched off that becomes 2.218, so the corrections already
  remove 80% of the problem and the mechanism is right;
* serve rates explain almost none of it (0.14pp bias, worth 0.03 games).

`split_sigma` is the lever: it wobbles the skill split within a match, which
produces lopsided sets, and lopsided sets are shorter. It ships at 0.08.

**The obvious objection, checked here rather than assumed away:** Stage 7's
own grid already ran to 0.12 and chose 0.08, so a bigger value was available
and rejected. It was selected on PIT deviation, not on mean bias, so it is
entirely possible that fixing the mean costs the distribution shape Stage 7
was gated on. This sweep therefore reports the games bias AND all three of
Stage 7's gate metrics together, so the trade is visible instead of implied.

Gate criteria carried over from Stage 7, unchanged:
  |tiebreak gap| < 0.01, central-80% coverage in [0.75, 0.85], PIT deviation
  as low as possible.

Measured on TUNE. Nothing is fitted or written. TEST and HOLDOUT untouched.

Run:  python -m scripts.sweep_split_sigma
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
from scripts.fit_stage7 import COVERAGE_BAND, MAX_TB_GAP, measure, predict

#: Extends past Stage 7's own grid (which stopped at 0.12) so the shape of
#: the trade is visible rather than being cut off at the incumbent's edge,
#: and is fine between 0.08 and 0.12 because BOTH the games bias and the
#: tiebreak gap change sign inside that interval — a coarse grid jumps over
#: the only region where the two can be near zero together.
SIGMAS = (0.0, 0.08, 0.09, 0.095, 0.10, 0.105, 0.11, 0.12, 0.16, 0.24)

REPORT = C.REPORTS_DIR / "split_sigma_sweep.md"


def main() -> None:
    fitted = json.loads(C.FITTED_PARAMS_PATH.read_text())
    s6, s7 = fitted["stage_6"], fitted["stage_7"]
    vp = V.VenueParams(shrink_n0=s6["shrink_n0"], use_indoor=s6["use_indoor"],
                       enabled=s6["enabled"])
    infl = s7["tiebreak_inflation"]

    pan = P.build(splits=("tune",), venue_params=vp)
    pan = pan.sample(min(len(pan), 6_000), random_state=C.MC_SEED)
    prov = pan["format_provenance"].to_numpy()
    w = CR.provenance_weights(prov, CR.CorrectionParams(
        scheme=s7["provenance_scheme"], inferred_weight=s7.get("inferred_weight", 1.0)))
    print(f"TUNE rows {len(pan):,}, weighted {int((w > 0).sum()):,}")

    rows = []
    for sigma in SIGMAS:
        pred = predict(pan, infl, sigma)
        m = measure(pan, pred, w)
        # Mean predicted total games, for the bias the sweep exists to fix.
        exp = []
        for pa, pb, key in zip(pan["pa"], pan["pb"], pan["spec_key"]):
            spec = FormatSpec(best_of=key[0], games_to_win_set=key[1],
                              tb_at=key[2], tb_to=key[3], final_set=key[4],
                              final_tb_at=key[5], final_tb_to=key[6],
                              provenance="documented", source="sweep")
            pmf = CR.corrected_distribution(
                round(float(pa), 3), round(float(pb), 3), spec,
                CR.CorrectionParams(tiebreak_inflation=infl, split_sigma=sigma,
                                    scheme=s7["provenance_scheme"])
            ).total_games_pmf()
            exp.append(sum(g * p for g, p in pmf.items()))
        bias = float(np.mean(np.array(exp) - pan["total_games"].to_numpy()))
        gate = (abs(m["tb_gap"]) < MAX_TB_GAP
                and COVERAGE_BAND[0] <= m["coverage80"] <= COVERAGE_BAND[1])
        rows.append({"split_sigma": sigma, "games_bias": bias,
                     "tb_gap": m["tb_gap"], "pit_dev": m["pit_dev"],
                     "coverage80": m["coverage80"], "gate_ok": gate})
        # .3f, not .2f: the grid has 0.095 and 0.105 in it and two decimals
        # renders both as "0.10", which makes the console output disagree with
        # the table it is supposed to summarise.
        print(f"  sigma {sigma:.3f}  bias {bias:+.3f}  tb_gap {m['tb_gap']:+.4f}  "
              f"pit_dev {m['pit_dev']:.5f}  cov {m['coverage80']:.3f}  "
              f"gate {'OK' if gate else 'FAIL'}")

    tab = pd.DataFrame(rows)
    inc = tab[tab["split_sigma"] == 0.08].iloc[0]
    feasible = tab[tab["gate_ok"]]
    best = (feasible.iloc[feasible["games_bias"].abs().argmin()]
            if len(feasible) else None)

    print(f"\nincumbent sigma 0.08: bias {inc['games_bias']:+.3f}, "
          f"pit_dev {inc['pit_dev']:.5f}, coverage {inc['coverage80']:.3f}")
    if best is not None:
        print(f"smallest |bias| that still passes the gate: sigma "
              f"{best['split_sigma']:.3f}, bias {best['games_bias']:+.3f}, "
              f"pit_dev {best['pit_dev']:.5f}")
        improves = bool(abs(best["games_bias"]) < abs(inc["games_bias"]) - 0.02
                        and best["pit_dev"] <= inc["pit_dev"] + 0.002)
    else:
        improves = False
    print(f"\nA better setting exists that keeps Stage 7's gate: {improves}")

    lines = [
        "# Sweeping Stage 7's `split_sigma` against the total-games bias\n",
        "The model expects 0.428 more games than matches produce; with Stage "
        "7 off that is 2.218, so the corrections already remove 80% and the "
        "mechanism is right. Serve rates explain almost none of it. "
        "`split_sigma` wobbles the skill split, which makes sets lopsided, "
        "and lopsided sets are shorter.\n",
        "\n**The objection this checks:** Stage 7's own grid ran to 0.12 and "
        "chose 0.08. A larger value was available and rejected — on PIT "
        "deviation, not on mean bias. So fixing the mean may cost the "
        "distribution shape Stage 7 was gated on, and the table below reports "
        "both so the trade is visible rather than implied.\n",
        f"\nGate carried over unchanged: `|tb_gap| < {MAX_TB_GAP}`, coverage "
        f"in {COVERAGE_BAND}. TUNE only; nothing fitted or written.\n",
        "\n## Sweep\n",
        tab.to_markdown(index=False),
        f"\n\nIncumbent `split_sigma` 0.08: bias {inc['games_bias']:+.3f}, "
        f"PIT deviation {inc['pit_dev']:.5f}, coverage {inc['coverage80']:.3f}.\n",
        (f"\nSmallest absolute bias that still passes the gate: "
         f"`split_sigma` {best['split_sigma']:.3f}, bias "
         f"{best['games_bias']:+.3f}, PIT deviation {best['pit_dev']:.5f}, "
         f"coverage {best['coverage80']:.3f}.\n" if best is not None
         else "\nNo setting in the sweep passes the gate.\n"),
        f"\n**A better setting exists that keeps Stage 7's gate: "
        f"{improves}.**\n",
        "\n## If this is taken further\n",
        "Changing `split_sigma` means re-running Stage 7's gate properly (not "
        "just the two metrics here), and the calibration maps sit downstream "
        "and would need refitting — they were fitted against the current "
        "corrections. And there is still no untouched window to confirm the "
        "result on. This sweep says whether the fix is available, not whether "
        "it is proven.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
