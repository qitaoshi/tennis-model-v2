"""Restate run 2's day-1 picks under the staking rule adopted 2026-08-16.

One-shot, and safe only because it runs before anything settled. Both stakes
are pure functions of values already on each row:

    stake_flat  = FLAT_STAKE
    stake_kelly = min(KELLY_SCALE * kelly_full * bankroll, KELLY_CAP)

`kelly_full` is stored per row, and with zero settled bets every row's Kelly
bankroll is still BANKROLL_START — nothing has compounded. So the rewritten
rows are byte-for-byte what the run would have written had it started under
this rule, which is the only thing that makes editing an append-only ledger
defensible. Run this after a settlement and it would be fabricating history.

It also rewrites `fingerprint.staking` in each run_state, because the staking
rule genuinely moved and `check_unchanged` must not be left asserting the old
one. That is the guard being updated deliberately, in a script that says so —
not silenced.

Run:  python -m scripts.backfill_run2_staking [--dry-run]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from scripts import paper_trade as P


def restate(variant: str, dry_run: bool) -> None:
    P.select_variant(variant)
    led = pd.read_csv(P.LEDGER, dtype=str, keep_default_na=False)

    settled = led[led["row_type"].isin(("settle", "board_settle"))]
    if len(settled):
        raise SystemExit(
            f"{P.LEDGER} already has {len(settled)} settled row(s). Kelly "
            "stakes depend on a bankroll that has compounded, so they can no "
            "longer be recomputed from the row alone. Restart the run instead.")

    picks = led["row_type"] == "pick"
    before = led.loc[picks, ["stake_flat", "stake_kelly"]].copy()

    led.loc[picks, "stake_flat"] = f"{P.FLAT_STAKE}"
    led.loc[picks, "bankroll_kelly"] = f"{P.BANKROLL_START}"
    led.loc[picks, "stake_kelly"] = [
        f"{round(min(P.KELLY_SCALE * float(k) * P.BANKROLL_START, P.KELLY_CAP), 4)}"
        for k in led.loc[picks, "kelly_full"]
    ]

    kelly = led.loc[picks, "stake_kelly"].astype(float)
    print(f"{variant:9s} {picks.sum():3d} picks · "
          f"flat {before['stake_flat'].iloc[0]} -> {P.FLAT_STAKE} · "
          f"kelly {kelly.min():.2f}..{kelly.max():.2f}u "
          f"({(kelly >= P.KELLY_CAP).sum()} at the cap)")

    state = json.loads(P.STATE.read_text())
    state["bankroll_start"] = P.BANKROLL_START
    state["flat_stake"] = P.FLAT_STAKE
    state["fingerprint"] = P.model_fingerprint()
    state["restated_on"] = "2026-08-16"
    state["restated_why"] = (
        "Flat staking restored alongside Kelly and the Kelly scale moved to a "
        "25u bankroll so the 1u cap stops binding on every bet. Day-1 picks "
        "were recomputed from kelly_full before anything settled; no result "
        "informed the change.")

    if dry_run:
        print(f"  [dry-run] would rewrite {P.LEDGER.name} and {P.STATE.name}")
        return
    led.to_csv(P.LEDGER, index=False)
    P.STATE.write_text(json.dumps(state, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for variant in sorted(P.VARIANTS):
        restate(variant, args.dry_run)


if __name__ == "__main__":
    main()
