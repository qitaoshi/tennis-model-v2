"""How close is the model to the bookmakers, as a forecaster?

Every previous odds report asked a betting question — edge, CLV, ROI. Under
the current goal that is the wrong question. This one asks the accuracy
question directly: **on the same matches, who predicts better, the model or
the market?**

The market's de-vigged probability is treated as a rival forecast and scored
with exactly the metrics the model is scored with. No stake, no price, no
margin: just Brier and log-loss on identical matches.

That makes the bookmaker a benchmark rather than an opponent, which is the
right frame for a projection model. A bookmaker's closing line is close to
the best public forecast that exists for a tennis match, so "how far behind
the market" is a meaningful yardstick in a way that "better than a coin
flip" is not.

SCOPE: TUNE only (2024-01-01 .. 2025-06-30). The odds file runs to 2026-05
and therefore covers TEST and HOLDOUT too, and both are already spent — so
this deliberately cuts at the TUNE boundary rather than quietly reading them.

Nothing is fitted or selected here.

Run:  python -m scripts.model_vs_market
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from model import venue as V
from scripts import panel as P
from scripts.clv_backtest import _name_key
from scripts.multi_market_clv import (_actual, _alias_map, _join_odds,
                                      _pair, _selections)

ODDS = C.PROCESSED_DIR / "odds_portal_flat_2024.parquet"
REPORT = C.REPORTS_DIR / "model_vs_market.md"

#: A tournament's rows are dated by its START date, so a quote can sit up to
#: this many days after the match date the model carries.
TOURNAMENT_SPAN_DAYS = 16


def _scores(p: np.ndarray, y: np.ndarray) -> dict:
    return {"n": int(len(p)), "brier": RC.brier(p, y),
            "logloss": RC.log_loss(p, y), "ece": RC.calibration_error(p, y)}


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
    pan = pan[rec.loc[pan["match_id"], "tour"].eq("atp").to_numpy()].copy()
    for col in ("date", "winner_name", "loser_name"):
        pan[col] = rec.loc[pan["match_id"], col].to_numpy()
    print(f"TUNE ATP panel: {len(pan):,} matches")

    # Model probability for the player who actually won, through corrections.
    from model.rules import FormatSpec
    p_model = []
    for pa, pb, key in zip(pan["pa"], pan["pb"], pan["spec_key"]):
        spec = FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                          tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                          final_tb_to=key[6], provenance="documented", source="mvm")
        p_model.append(CR.corrected_distribution(
            round(float(pa), 3), round(float(pb), 3), spec, cp).p_a)
    pan["p_model_raw"] = p_model
    maps = RC.CalibrationMaps.load()
    pan["p_model_cal"] = np.asarray(maps.apply("match_winner",
                                               pan["p_model_raw"].to_numpy()))
    pan["winner_key"] = [_name_key(n) for n in pan["winner_name"]]
    pan["loser_key"] = [_name_key(n) for n in pan["loser_name"]]
    pan["pair_key"] = ["|".join(sorted(p)) for p in
                       zip(pan["winner_key"], pan["loser_key"])]

    odds = pd.read_parquet(ODDS)
    odds = odds[odds["market"] == "match_winner"].copy()
    odds["date"] = pd.to_datetime(odds["date"]).dt.date
    odds = odds[(odds["date"] >= C.TUNE_START) & (odds["date"] <= C.TUNE_END)]
    print(f"match_winner quotes inside TUNE: {len(odds):,}")

    alias = _alias_map(pd.unique(pd.concat([odds.player_1, odds.player_2])),
                       pd.unique(pd.concat([pan.winner_name, pan.loser_name])))
    odds["p1_key"] = [alias.get(n, _name_key(n, True)) for n in odds.player_1]
    odds["p2_key"] = [alias.get(n, _name_key(n, True)) for n in odds.player_2]
    odds["pair_key"] = ["|".join(sorted(p)) for p in zip(odds.p1_key, odds.p2_key)]

    matches = pan.drop_duplicates("match_id")[
        ["match_id", "pair_key", "date", "winner_key",
         "p_model_raw", "p_model_cal"]]
    mg = odds.merge(matches, on="pair_key", suffixes=("_odds", ""))
    delta = (pd.to_datetime(mg.date_odds) - pd.to_datetime(mg.date)).dt.days
    mg = mg[delta.between(-1, TOURNAMENT_SPAN_DAYS)].copy()
    mg["_d"] = delta[mg.index].abs()
    mg = mg.sort_values("_d").drop_duplicates(
        ["match_link", "side", "bookmaker"])
    print(f"joined quote rows: {len(mg):,}")

    # Keep only the quote on the player who actually won, so model and market
    # are scored on the same event with the same outcome (y == 1).
    side_key = np.where(mg["side"].eq("home"), mg["p1_key"], mg["p2_key"])
    mg = mg[side_key == mg["winner_key"]].copy()
    mg = mg[np.isfinite(mg["market_p"]) & mg["market_p"].between(0.01, 0.99)]
    print(f"quotes on the actual winner: {len(mg):,}")

    rows = []
    for book, g in mg.groupby("bookmaker"):
        if len(g) < 300:
            continue
        # Both orientations, or every outcome is a 1 by construction.
        y = np.concatenate([np.ones(len(g)), np.zeros(len(g))])
        mk = np.concatenate([g["market_p"], 1 - g["market_p"]])
        mr = np.concatenate([g["p_model_raw"], 1 - g["p_model_raw"]])
        mc = np.concatenate([g["p_model_cal"], 1 - g["p_model_cal"]])
        rows.append({"bookmaker": book, "n_matches": len(g),
                     **{f"market_{k}": v for k, v in _scores(mk, y).items()
                        if k != "n"},
                     **{f"model_{k}": v for k, v in _scores(mc, y).items()
                        if k != "n"},
                     "model_raw_brier": RC.brier(mr, y),
                     "brier_gap": RC.brier(mc, y) - RC.brier(mk, y),
                     "logloss_gap": RC.log_loss(mc, y) - RC.log_loss(mk, y)})
    tab = pd.DataFrame(rows).sort_values("n_matches", ascending=False)
    print("\n" + tab.to_string(index=False))

    # Consensus: median de-vigged probability across books for each match.
    cons = mg.groupby("match_id").agg(
        market_p=("market_p", "median"),
        p_model_cal=("p_model_cal", "first"),
        p_model_raw=("p_model_raw", "first"), n_books=("bookmaker", "nunique"))
    y = np.concatenate([np.ones(len(cons)), np.zeros(len(cons))])
    mk = np.concatenate([cons.market_p, 1 - cons.market_p])
    mc = np.concatenate([cons.p_model_cal, 1 - cons.p_model_cal])
    mr = np.concatenate([cons.p_model_raw, 1 - cons.p_model_raw])
    c_market, c_model, c_raw = _scores(mk, y), _scores(mc, y), _scores(mr, y)
    # Where they disagree most is where the model's claim is really tested.
    disagree = (cons.p_model_cal - cons.market_p).abs()
    big = disagree > 0.10
    yb = np.concatenate([np.ones(int(big.sum())), np.zeros(int(big.sum()))])
    mkb = np.concatenate([cons.market_p[big], 1 - cons.market_p[big]])
    mcb = np.concatenate([cons.p_model_cal[big], 1 - cons.p_model_cal[big]])

    print(f"\nCONSENSUS ({len(cons):,} matches, median across books)")
    print(f"  market: Brier {c_market['brier']:.5f} log-loss "
          f"{c_market['logloss']:.5f} ECE {c_market['ece']:.5f}")
    print(f"  model : Brier {c_model['brier']:.5f} log-loss "
          f"{c_model['logloss']:.5f} ECE {c_model['ece']:.5f}")
    print(f"  gap   : Brier {c_model['brier'] - c_market['brier']:+.5f}, "
          f"log-loss {c_model['logloss'] - c_market['logloss']:+.5f}")
    print(f"\n  disagreements >10pp: {int(big.sum()):,} matches — "
          f"market Brier {RC.brier(mkb, yb):.5f} vs model {RC.brier(mcb, yb):.5f}")

    lines = [
        "# Model versus the bookmakers, as forecasters\n",
        "The market's de-vigged probability scored as a rival forecast, with "
        "the same metrics as the model, on the same matches. No stake, no "
        "price, no margin — this is an accuracy comparison, not a betting "
        "one.\n",
        f"\n**Scope: TUNE only ({C.TUNE_START} .. {C.TUNE_END}), ATP main "
        "tour.** The odds file runs to 2026-05 and so covers TEST and HOLDOUT "
        "as well; both are already spent, so this cuts at the TUNE boundary "
        "rather than quietly reading them. Nothing is fitted here.\n",
        "\n## Consensus of all books\n",
        f"\n| forecaster | Brier | log-loss | ECE |\n|---|---:|---:|---:|\n"
        f"| market (median of books) | {c_market['brier']:.5f} "
        f"| {c_market['logloss']:.5f} | {c_market['ece']:.5f} |\n"
        f"| model (calibrated) | {c_model['brier']:.5f} "
        f"| {c_model['logloss']:.5f} | {c_model['ece']:.5f} |\n"
        f"| model (uncalibrated) | {c_raw['brier']:.5f} "
        f"| {c_raw['logloss']:.5f} | {c_raw['ece']:.5f} |\n",
        f"\n{len(cons):,} matches. Gap to market: Brier "
        f"{c_model['brier'] - c_market['brier']:+.5f}, log-loss "
        f"{c_model['logloss'] - c_market['logloss']:+.5f}. Positive means the "
        "model is worse.\n",
        "\n## Per bookmaker\n",
        tab.to_markdown(index=False),
        "\n\n## Where they disagree\n",
        f"\nOn the {int(big.sum()):,} matches where model and market differ by "
        f"more than 10 points, market Brier {RC.brier(mkb, yb):.5f} against "
        f"model {RC.brier(mcb, yb):.5f}. This is the honest test: agreeing "
        "with the market costs nothing and proves nothing, so the model's "
        "value shows up only where it takes a different view.\n",
        "\n## Reading this\n",
        "A bookmaker's price is close to the best public forecast that exists "
        "for a tennis match — it aggregates sharp money, injury news and "
        "team information the model has none of. So the target is not to beat "
        "it. The useful question is how much is given up, and whether that "
        "gap is small enough for the model's projections to be worth acting "
        "on where no line exists.\n",
        "\n" + _other_markets(pan),
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


def _other_markets(pan: pd.DataFrame) -> str:
    """Same comparison for the line-based markets.

    match_winner is one probability per match, so it can be handled directly.
    Totals and handicaps are a ladder of lines per match, and orienting a
    quote onto the right rung — whose handicap, which sign, which side — is
    fiddly enough that it is worth reusing the logic the CLV work already
    debugged rather than writing a second copy of it.
    """
    rec = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet").set_index("match_id")
    p = pan.copy()
    for col in ("tourney_name", "surface", "winner_games", "loser_games",
                "n_sets", "score"):
        p[col] = rec.loc[p["match_id"], col].to_numpy()
    p["pair_key"] = [_pair(a, b) for a, b in zip(p["winner_name"], p["loser_name"])]

    sels = _selections(p)
    odds = pd.read_parquet(ODDS)
    odds["date"] = pd.to_datetime(odds["date"]).dt.date
    odds = odds[(odds["date"] >= C.TUNE_START) & (odds["date"] <= C.TUNE_END)]
    joined = _join_odds(sels, odds)
    if joined.empty:
        return "## Other markets\n\nNo quotes joined.\n"

    out = [_actual(r) for _, r in joined.iterrows()]
    joined["y"] = [a for a, _ in out]
    joined["push"] = [pu for _, pu in out]
    # A push is neither a win nor a loss, so it cannot be scored as either.
    joined = joined[~joined["push"]]
    joined = joined[np.isfinite(joined["market_p"])
                    & joined["market_p"].between(0.01, 0.99)]

    rows = []
    for market, g in joined.groupby("market"):
        # One consensus quote per selection, so heavily-quoted lines do not
        # dominate by weight of bookmakers rather than weight of evidence.
        c = g.groupby(["match_id", "selection"]).agg(
            market_p=("market_p", "median"), model_p=("model_p", "first"),
            y=("y", "first")).reset_index()
        if len(c) < 300:
            continue
        y = c["y"].to_numpy(dtype=float)
        mk, md = c["market_p"].to_numpy(), c["model_p"].to_numpy()
        # Orientation tripwire. If a market's quotes are being bound to the
        # wrong side or the wrong sign, model and market probabilities stop
        # agreeing in the aggregate long before anyone notices the metrics
        # look odd. A well-oriented join correlates strongly and has a mean
        # difference near zero; a mis-oriented one does not. Handicaps are the
        # real risk here — the pricer names "A +6.5" for A winning BY MORE
        # THAN 6.5, the opposite sign to a bookmaker's -6.5 for the same bet.
        rows.append({
            "market": market, "n_selections": len(c),
            "n_matches": c["match_id"].nunique(),
            "market_brier": RC.brier(mk, y), "model_brier": RC.brier(md, y),
            "market_logloss": RC.log_loss(mk, y),
            "model_logloss": RC.log_loss(md, y),
            "market_ece": RC.calibration_error(mk, y),
            "model_ece": RC.calibration_error(md, y),
            "brier_gap": RC.brier(md, y) - RC.brier(mk, y),
            "corr": float(np.corrcoef(mk, md)[0, 1]),
            "mean_diff": float(np.mean(md - mk)),
            "base_rate": float(np.mean(y)),
        })
    tab = pd.DataFrame(rows).sort_values("n_selections", ascending=False)
    print("\nOTHER MARKETS (consensus per selection)")
    print(tab.to_string(index=False))
    return ("## Other markets\n\nConsensus quote per selection, pushes "
            "excluded (a push is neither a win nor a loss and cannot be "
            "scored as either). Positive `brier_gap` means the model is "
            "worse.\n\n" + tab.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
