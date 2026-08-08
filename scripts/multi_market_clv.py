"""Frozen-model multi-market CLV and Bet365 raw-odds ROI evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from model import constants as C
from model import corrections as CR
from model import recalibrate as RC
from model import price as PZ
from model import rules as R
from scripts import panel as P
from scripts.clv_backtest import _name_key, _text_key

BOOTSTRAP_N = 10_000
# Longest span a single tournament covers, used to pair a match date against
# its tournament's start date.
TOURNAMENT_SPAN_DAYS = 20
FAMILIES = ("match_winner", "total_games", "total_sets",
            "games_handicap", "set_betting")


def _bootstrap_mean(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(C.MC_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_N, len(values))).mean(axis=1)
    return (float(values.mean()), float(np.quantile(samples, .025)),
            float(np.quantile(samples, .975)))


def _pair(a: object, b: object, odds_style: bool = False) -> str:
    return "|".join(sorted((_name_key(a, odds_style), _name_key(b, odds_style))))


def _keys(name: object, odds_style: bool = False) -> tuple[str, frozenset]:
    """Given-name initial plus the set of surname tokens.

    The two sources disagree about how many name parts to print:
    OddsPortal writes ``Alcaraz Garfia C.`` where the match data has
    ``Carlos Alcaraz``, and ``Herbert P.`` where it has
    ``Pierre-Hugues Herbert``. A concatenated key cannot satisfy both
    directions, so match on initial plus any shared surname token.
    """
    tokens = re.findall(r"[a-z]+", str(name).lower().replace("'", ""))
    if not tokens:
        return "", frozenset()
    if odds_style:
        surname, initials = list(tokens), []
        while len(surname) > 1 and len(surname[-1]) == 1:
            initials.insert(0, surname.pop())
        return (initials[0] if initials else surname[-1][0]), frozenset(surname)
    return tokens[0][0], frozenset(tokens[1:])


def _alias_map(odds_names, model_names) -> dict[str, str]:
    """Map each OddsPortal display name to the match data's canonical key.

    Ambiguous names (two model players compatible with one odds name) are
    left out rather than guessed.
    """
    index: dict[str, list[tuple[frozenset, str]]] = {}
    for name in model_names:
        initial, surnames = _keys(name)
        index.setdefault(initial, []).append((surnames, _name_key(name)))
    alias = {}
    for name in odds_names:
        initial, surnames = _keys(name, odds_style=True)
        hits = {key for model_surnames, key in index.get(initial, ())
                if model_surnames & surnames}
        if len(hits) == 1:
            alias[name] = hits.pop()
    return alias


def _model_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = P.build(splits=("holdout",))
    holdout = pd.read_parquet(C.PROCESSED_DIR / "matches_with_holdout.parquet")
    meta = holdout.set_index("match_id")
    panel = panel[meta.loc[panel["match_id"], "tour"].eq("atp").to_numpy()].copy()
    for col in ("date", "tourney_name", "surface", "winner_name", "loser_name",
                "winner_games", "loser_games", "n_sets", "score"):
        panel[col] = meta.loc[panel["match_id"], col].to_numpy()
    panel["pair_key"] = [_pair(a, b) for a, b in
                         zip(panel["winner_name"], panel["loser_name"])]
    return panel, holdout


def _selections(panel: pd.DataFrame) -> pd.DataFrame:
    fitted = P.load_fitted()
    s7 = fitted["stage_7"]
    params = CR.CorrectionParams(
        tiebreak_inflation=s7["tiebreak_inflation"],
        split_sigma=s7["split_sigma"], level_sigma=s7["level_sigma"],
        recenter=s7["recenter"], scheme=s7["provenance_scheme"])
    maps = RC.CalibrationMaps.load()
    light = PZ.Pricer.__new__(PZ.Pricer)
    light.maps = maps
    rows = []
    for r in panel.itertuples(index=False):
        k = r.spec_key
        spec = R.FormatSpec(
            best_of=k[0], games_to_win_set=k[1], tb_at=k[2], tb_to=k[3],
            final_set=k[4], final_tb_at=k[5], final_tb_to=k[6],
            provenance=r.format_provenance, source="multi_market_clv")
        dist = CR.corrected_distribution(round(float(r.pa), 3),
                                          round(float(r.pb), 3), spec, params)
        sels = light._selections(dist, spec)
        # Total-sets prices are a direct, unfitted derivation from the frozen
        # set PMF; no new calibration or threshold is introduced.
        set_pmf: dict[int, float] = {}
        for (a, b), prob in dist.sets.items():
            set_pmf[a + b] = set_pmf.get(a + b, 0.0) + prob
        for twice in range(2, 2 * spec.best_of + 2):
            line = twice / 2
            over = sum(p for n, p in set_pmf.items() if n > line)
            under = sum(p for n, p in set_pmf.items() if n < line)
            push = set_pmf.get(int(line), 0.0) if line.is_integer() else 0.0
            sels.extend([
                PZ.Selection("total_sets", f"over {line}", over,
                             (1 - push) / over if over else 1e9, push),
                PZ.Selection("total_sets", f"under {line}", under,
                             (1 - push) / under if under else 1e9, push),
            ])
        score = str(r.score)
        sets = [(int(a), int(b)) for a, b in
                re.findall(r"(\d+)-(\d+)", score)]
        score_a = sum(a > b for a, b in sets)
        score_b = sum(b > a for a, b in sets)
        for s in sels:
            rows.append({
                "match_id": r.match_id, "date": r.date,
                "pair_key": r.pair_key, "model_a": r.winner_name,
                "model_b": r.loser_name, "market": s.market,
                "selection": s.selection, "model_p": s.probability,
                "push_probability": s.push_probability,
                "actual_total_games": r.winner_games + r.loser_games,
                "actual_total_sets": r.n_sets,
                "actual_game_diff": r.winner_games - r.loser_games,
                "actual_score": f"{score_a}-{score_b}",
            })
    return pd.DataFrame(rows)


def _orient_selection(row: pd.Series, side: str) -> str:
    home_is_a = row.player_1_key == _name_key(row.model_a)
    if row.market == "match_winner":
        return "A" if (side == "home") == home_is_a else "B"
    if row.market in ("total_games", "total_sets"):
        return f"{side} {float(row.line):g}"
    if row.market == "games_handicap":
        # The quoted line is the HOME player's handicap and the away side of
        # the same quote carries its negation. The sign follows the side being
        # priced; only WHICH of the model's two players it is depends on who
        # was at home. Deriving the sign from that relabelling instead prices
        # one side and settles the other.
        # The pricer names a handicap by the margin threshold itself, so its
        # "A +6.5" is A winning BY MORE THAN 6.5 -- the opposite sign to the
        # bookmaker's -6.5 for the same bet. Convert rather than relabel.
        h = float(row.line)
        if side == "home":
            who, value = ("A" if home_is_a else "B"), -h
        else:
            who, value = ("B" if home_is_a else "A"), h
        return f"{who} {value:+g}"
    if row.market == "set_betting":
        score = str(row.line).replace(":", "-")
        if home_is_a:
            return score
        a, b = score.split("-")
        return f"{b}-{a}"
    raise ValueError(row.market)


def _actual(row: pd.Series) -> tuple[float, bool]:
    s = row.selection
    if row.market == "match_winner":
        return (1.0 if s == "A" else 0.0), False
    if row.market in ("total_games", "total_sets"):
        actual = (row.actual_total_games if row.market == "total_games"
                  else row.actual_total_sets)
        line = float(row.line)
        return (float(actual > line) if s.startswith("over")
                else float(actual < line),
                float(actual) == line)
    if row.market == "games_handicap":
        # The pricer prices "A h" as P(margin > h) but labels the other side
        # of the same threshold "B -h", so B's label has to be negated back
        # before it can be settled.
        who, value = s.split()
        diff = float(row.actual_game_diff)
        limit = float(value) if who == "A" else -float(value)
        return (float(diff > limit) if who == "A" else float(diff < limit),
                diff == limit)
    a, b = map(int, s.split("-"))
    actual_a, actual_b = (int(x) for x in row.actual_score.split("-"))
    return float((a, b) == (actual_a, actual_b)), False


def _join_odds(model: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    odds = odds.copy()
    alias = _alias_map(
        pd.unique(pd.concat([odds.player_1, odds.player_2])),
        pd.unique(pd.concat([model.model_a, model.model_b])))
    odds["player_1_key"] = [alias.get(n, _name_key(n, True))
                            for n in odds.player_1]
    odds["player_2_key"] = [alias.get(n, _name_key(n, True))
                            for n in odds.player_2]
    odds["pair_key"] = ["|".join(sorted(p)) for p in
                        zip(odds.player_1_key, odds.player_2_key)]
    odds["date"] = pd.to_datetime(odds.date).dt.date
    model = model[model.pair_key.isin(set(odds.pair_key))].copy()
    model["date"] = pd.to_datetime(model.date).dt.date
    # Bind each quote to its match first, on the match-level columns only.
    # Joining the priced selections in at this point would pair every quote
    # with every selection of the match, and collapsing that back down again
    # would discard the one selection the quote is actually about.
    matches = model.drop_duplicates("match_id")[
        ["match_id", "pair_key", "date", "model_a"]]
    merged = odds.merge(matches, on="pair_key", suffixes=("_odds", ""))
    # The match data dates a match by its tournament's START date (every
    # Wimbledon 2024 row reads 2024-07-01, final included), while the odds
    # carry the real match date. So the offset is one-sided and as wide as
    # a tournament, not the +/-1 day a match-date comparison would want.
    delta = (pd.to_datetime(merged.date_odds)
             - pd.to_datetime(merged.date)).dt.days
    merged = merged[delta.between(-1, TOURNAMENT_SPAN_DAYS)].copy()
    if merged.empty:
        return merged
    merged["_date_delta"] = delta[merged.index].abs()
    # A pair can meet more than once inside the window; bind each quote to the
    # single nearest match rather than counting it against every candidate.
    merged = merged.sort_values("_date_delta").drop_duplicates(
        ["match_link", "market", "line", "side", "bookmaker"])
    # Now the quote knows its match, so it can name the selection it is a
    # price for, and that name is what carries the model's probability over.
    merged["model_selection"] = [_orient_selection(row, row.side)
                                 for _, row in merged.iterrows()]
    priced = [c for c in ("match_id", "market", "selection", "model_p",
                          "push_probability", "model_b", "actual_total_games",
                          "actual_total_sets", "actual_game_diff",
                          "actual_score") if c in model.columns]
    return merged.merge(model[priced],
                        left_on=["match_id", "market", "model_selection"],
                        right_on=["match_id", "market", "selection"])


def reconcile(odds: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for path in sorted((C.PROCESSED_DIR / "raw").glob("202[4-6].xlsx")):
        x = pd.read_excel(path, engine="openpyxl")
        x["date"] = pd.to_datetime(x["Date"], errors="coerce").dt.date
        x = x[x.date >= C.HOLDOUT_CUTOFF]
        for _, r in x.iterrows():
            frames += [{"date": r.date, "pair_key": _pair(r.Winner, r.Loser),
                        "player_key": _name_key(r.Winner),
                        "xlsx_side": "winner", "xlsx_odds": r.B365W},
                       {"date": r.date, "pair_key": _pair(r.Winner, r.Loser),
                        "player_key": _name_key(r.Loser),
                        "xlsx_side": "loser", "xlsx_odds": r.B365L}]
    ref = pd.DataFrame(frames)
    scraped = odds[odds.bookmaker_tag.eq("bet365")
                   & odds.market.eq("match_winner")].copy()
    scraped["pair_key"] = [_pair(a, b) for a, b in
                           zip(scraped.player_1, scraped.player_2)]
    scraped["player_key"] = [
        _name_key(a if side == "home" else b)
        for a, b, side in zip(scraped.player_1, scraped.player_2,
                              scraped.side)]
    out = scraped.merge(ref, on=["pair_key", "player_key"], how="inner")
    delta = (pd.to_datetime(out.date_x) - pd.to_datetime(out.date_y)).abs().dt.days
    out = out[delta <= 1].copy()
    out["abs_diff"] = (out.decimal_odds - pd.to_numeric(out.xlsx_odds,
                                                         errors="coerce")).abs()
    return out.dropna(subset=["abs_diff"])


def _report(recon: pd.DataFrame, results: dict[str, dict]) -> str:
    rec = "no overlap"
    if not recon.empty:
        rec = (f"{len(recon):,} side comparisons; agreement within 0.05 "
               f"decimal odds {recon.abs_diff.le(.05).mean():.1%}; median "
               f"absolute difference {recon.abs_diff.median():.3f}; worst "
               f"{recon.abs_diff.max():.3f}")
    text = f"""# Multi-market CLV and ROI backtest

This is evaluation only; no model parameter was fitted, tuned, or selected.
ROI uses **raw Bet365 decimal odds** (`decimal_odds - 1` on a win), while
de-vigged `market_p` is used only for edge and CLV. A push returns zero profit
and remains in the placed-bet denominator.

## Known-good Bet365 reconciliation

{rec}

The exact OddsPortal bookmaker tag is `bet365`. The reconciliation is a gate:
other families must not be interpreted until the scraped match-winner prices
agree closely with the independent B365W/B365L reference.

## Family results
"""
    for family in FAMILIES:
        r = results.get(family, {})
        if not r:
            text += f"\n### {family}\nNo matched odds were available.\n"
            continue
        text += f"""
### {family}

The model's preferred selections had mean CLV {r['clv']:+.2%}
(bootstrap 95% CI {r['clv_lo']:+.2%} to {r['clv_hi']:+.2%}) versus the sharp
quote. Betting positive-Bet365-edge selections returned {r['roi']:+.2%}
per unit staked at Bet365's raw price (bootstrap 95% CI
{r['roi_lo']:+.2%} to {r['roi_hi']:+.2%}).

- coverage: {r['matched']:,} matched, {r['bet365']:,} with Bet365,
  {r['bets']:,} bets, {r['excluded']:,} Bet365 exclusions
- verdict: **{r['verdict']}**
"""
    return text


def evaluate(odds: pd.DataFrame, output: Path) -> None:
    model, _ = _model_table()
    selections = _selections(model)
    joined = _join_odds(selections, odds)
    results = {}
    for family in FAMILIES:
        g = joined[joined.market.eq(family)].copy()
        if g.empty:
            continue
        g["model_selection"] = [
            _orient_selection(row, row.side) for _, row in g.iterrows()]
        g = g[g.selection.eq(g.model_selection)]
        # Keep one model-preferred side per offered line, not both sides.
        g = g.sort_values("model_p", ascending=False).drop_duplicates(
            ["match_id", "market", "line", "bookmaker_tag"])
        sharp = g[g.bookmaker_tag.eq("sharp")][
            ["match_id", "market", "line", "model_p", "market_p"]
        ].rename(columns={"market_p": "sharp_p"})
        b365 = g[g.bookmaker_tag.eq("bet365")][
            ["match_id", "market", "line", "selection", "model_p",
             "market_p", "decimal_odds", "actual_total_games",
             "actual_total_sets", "actual_game_diff", "actual_score",
             "push_probability"]
        ].rename(columns={"market_p": "bet365_p"})
        both = b365.merge(sharp, on=["match_id", "market", "line", "model_p"],
                          how="left")
        if both.empty:  # family with no Bet365 quotes at all
            continue
        both["edge"] = both.model_p - both.bet365_p
        both["clv"] = both.model_p / both.sharp_p - 1
        outcomes = both.apply(_actual, axis=1)
        both["won"], both["push"] = zip(*outcomes)
        positive = both[both.edge > 0].copy()
        positive["profit"] = np.where(positive.push, 0.0,
                                      np.where(positive.won,
                                               positive.decimal_odds - 1, -1))
        cm, clo, chi = _bootstrap_mean(both.clv.dropna().to_numpy())
        rm, rlo, rhi = _bootstrap_mean(positive.profit.to_numpy())
        verdict = ("demonstrated edge" if rlo > 0
                   else "no demonstrated edge")
        results[family] = dict(clv=cm, clv_lo=clo, clv_hi=chi, roi=rm,
                               roi_lo=rlo, roi_hi=rhi,
                               matched=int(len(g)), bet365=int(len(b365)),
                               bets=int(len(positive)),
                               excluded=int(len(g) - len(b365)),
                               verdict=verdict)
    recon = reconcile(odds)
    output.write_text(_report(recon, results))
    print(f"wrote {output}")


def self_check() -> None:
    odds = np.array([2.0, 2.0])
    assert np.allclose((1 / odds) / (1 / odds).sum(), [.5, .5])
    row = pd.Series({"market": "total_games", "selection": "over",
                     "line": 22, "actual_total_games": 22})
    won, push = _actual(row)
    assert won == 0 and push is True
    assert _orient_selection(pd.Series({
        "market": "match_winner", "player_1_key": _name_key("Harry Home"),
        "model_a": "Away Aaron", "line": "",}), "home") == "B"
    assert _orient_selection(pd.Series({
        "market": "match_winner", "player_1_key": _name_key("Harry Home"),
        "model_a": "Harry Home", "line": "",}), "home") == "A"
    # A handicap quote keeps its sign whichever of the model's players is at
    # home; only the A/B label changes.
    giving = pd.Series({"market": "games_handicap", "line": "-6.5",
                        "player_1_key": _name_key("Harry Home"),
                        "model_a": "Harry Home"})
    # Bookmaker -6.5 on the home player is the pricer's +6.5 on whoever that
    # is: win by more than 6.5.
    assert _orient_selection(giving, "home") == "A +6.5"
    assert _orient_selection(giving, "away") == "B -6.5"
    losing_host = pd.Series({"market": "games_handicap", "line": "-6.5",
                             "player_1_key": _name_key("Harry Home"),
                             "model_a": "Away Aaron"})
    assert _orient_selection(losing_host, "home") == "B +6.5"
    assert _orient_selection(losing_host, "away") == "A -6.5"
    # Settlement has to read those labels the same way the pricer wrote them,
    # including B's negation.
    def _hcp(sel, diff):
        return _actual(pd.Series({"market": "games_handicap",
                                  "selection": sel, "actual_game_diff": diff}))
    assert _hcp("A +6.5", 7) == (1.0, False)
    assert _hcp("A +6.5", 6) == (0.0, False)
    assert _hcp("B -6.5", 6) == (1.0, False)   # priced as P(margin < 6.5)
    assert _hcp("B -6.5", 7) == (0.0, False)
    assert _hcp("A +6", 6) == (0.0, True)      # whole line can push
    assert _hcp("B -6", 6) == (0.0, True)
    # The two sides of one quote must never both win.
    for margin in (0, 3, 6, 7, 12):
        a_won, _ = _hcp("A +6.5", margin)
        b_won, _ = _hcp("B -6.5", margin)
        assert a_won + b_won == 1.0, margin
    # A correct-score bet settles on the whole score, not just the winner's
    # set count.
    score = {"market": "set_betting", "actual_score": "3-2"}
    assert _actual(pd.Series({**score, "selection": "3-2"})) == (1.0, False)
    assert _actual(pd.Series({**score, "selection": "3-1"})) == (0.0, False)
    assert _actual(pd.Series({**score, "selection": "3-0"})) == (0.0, False)
    assert _pair("Frances Tiafoe", "Holger Rune") == _pair(
        "Tiafoe F.", "Rune H.", odds_style=True)
    # The two sources print different numbers of name parts; every pair here
    # is the same player and must resolve to one key.
    same = [("Tiafoe F.", "Frances Tiafoe"),
            ("Etcheverry T. M.", "Tomas Martin Etcheverry"),
            ("Struff J-L.", "Jan-Lennard Struff"),
            ("Wolf J.J.", "J.J. Wolf"),
            ("Galan D. E.", "Daniel Elahi Galan"),
            ("Herbert P.", "Pierre-Hugues Herbert"),
            ("Alcaraz Garfia C.", "Carlos Alcaraz")]
    alias = _alias_map([o for o, _ in same], [m for _, m in same])
    for odds_name, model_name in same:
        assert alias.get(odds_name) == _name_key(model_name), odds_name
    # A name compatible with two different players must not be guessed.
    assert "Cerundolo J." not in _alias_map(
        ["Cerundolo J."], ["Juan Manuel Cerundolo", "Juan Pablo Cerundolo"])
    # A late-round match still belongs to its tournament, whose start date is
    # what the match data carries.
    model = pd.DataFrame([{
        "pair_key": _pair("Carlos Alcaraz", "Novak Djokovic"),
        "market": "match_winner", "date": "2024-07-01", "match_id": "w24",
        "model_a": "Carlos Alcaraz", "model_b": "Novak Djokovic",
        "selection": "A", "model_p": 0.6}])
    quote = {"player_1": "Alcaraz Garfia C.", "player_2": "Djokovic N.",
             "market": "match_winner", "line": "", "side": "home",
             "bookmaker": "bet365", "match_link": "L1"}
    final = _join_odds(model, pd.DataFrame([{**quote, "date": "2024-07-14"}]))
    assert len(final) == 1, "final should join to its tournament start date"
    assert final.iloc[0].player_1_key == _name_key("Carlos Alcaraz")
    stale = _join_odds(model, pd.DataFrame([{**quote, "date": "2024-09-01"}]))
    assert stale.empty, "a match far outside the tournament must not join"
    # A quote must carry its OWN line's probability across. A ladder offers
    # many lines, and picking any other one silently prices the wrong bet.
    ladder = pd.DataFrame([
        {"pair_key": _pair("Carlos Alcaraz", "Novak Djokovic"),
         "market": "total_games", "date": "2024-07-01", "match_id": "w24",
         "model_a": "Carlos Alcaraz", "model_b": "Novak Djokovic",
         "selection": f"{side} {line}", "model_p": p}
        for side, line, p in (("over", 21.5, 0.55), ("under", 21.5, 0.45),
                              ("over", 22.5, 0.40), ("under", 22.5, 0.60))])
    got = _join_odds(ladder, pd.DataFrame([{
        "player_1": "Alcaraz Garfia C.", "player_2": "Djokovic N.",
        "market": "total_games", "line": "22.5", "side": "over",
        "bookmaker": "bet365", "match_link": "L2", "date": "2024-07-08"}]))
    assert len(got) == 1, "one quote must match exactly one priced selection"
    assert got.iloc[0].selection == "over 22.5", got.iloc[0].selection
    assert got.iloc[0].model_p == 0.40, got.iloc[0].model_p
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odds", type=Path)
    parser.add_argument("--output", type=Path,
                        default=C.REPORTS_DIR / "multi_market_clv.md")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.odds is None:
        raise SystemExit("--odds is required")
    odds = (pd.read_parquet(args.odds) if args.odds.suffix == ".parquet"
            else pd.read_csv(args.odds))
    evaluate(odds, args.output)


if __name__ == "__main__":
    main()
