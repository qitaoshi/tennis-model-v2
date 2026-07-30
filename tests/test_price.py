"""price.py — the product's contract.

Checks the structure and the metadata the spec requires, plus the as-of
guarantee: a pricer built for a date has never seen a later match.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from model import constants as C
from model import price as PZ


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")


@pytest.fixture(scope="module")
def sample(matches: pd.DataFrame) -> pd.Series:
    ok = matches[(matches["split"] == "tune") & matches["in_scope"]
                 & matches["outcome"].eq("completed")]
    return ok.iloc[500]


@pytest.fixture(scope="module")
def priced(sample: pd.Series) -> PZ.PricedMatch:
    return PZ.price_match(sample["winner_id"], sample["loser_id"],
                          sample["tourney_code"], int(sample["season_file"]),
                          sample["surface"], sample["tourney_name"],
                          sample["date"])


def test_every_market_is_priced(priced: PZ.PricedMatch) -> None:
    markets = {s.market for s in priced.selections}
    assert markets == {"match_winner", "set_betting", "total_games",
                       "games_handicap", "tiebreak", "player_games_A",
                       "player_games_B"}
    assert len(priced.selections) > 100


def test_probabilities_and_prices_are_consistent(priced: PZ.PricedMatch) -> None:
    for s in priced.selections:
        assert 0 < s.probability <= 1, s
        assert s.fair_decimal >= 1.0, s
        expected = (1 - s.push_probability) / s.probability
        assert s.fair_decimal == pytest.approx(expected, rel=1e-9)


def test_two_way_markets_sum_to_one(priced: PZ.PricedMatch) -> None:
    mw = {s.selection: s.probability for s in priced.market("match_winner")}
    assert sum(mw.values()) == pytest.approx(1.0, abs=1e-6)
    tb = {s.selection: s.probability for s in priced.market("tiebreak")}
    assert sum(tb.values()) == pytest.approx(1.0, abs=1e-6)


def test_set_betting_is_a_distribution(priced: PZ.PricedMatch) -> None:
    total = sum(s.probability for s in priced.market("set_betting"))
    assert total == pytest.approx(1.0, abs=1e-3)


def test_totals_ladder_includes_push_on_whole_lines(priced: PZ.PricedMatch) -> None:
    whole = [s for s in priced.market("total_games")
             if float(s.selection.split()[-1]).is_integer()]
    half = [s for s in priced.market("total_games")
            if not float(s.selection.split()[-1]).is_integer()]
    assert any(s.push_probability > 0 for s in whole)
    assert all(s.push_probability == 0 for s in half)
    for s in whole:
        line = float(s.selection.split()[-1])
        other = next(o for o in whole
                     if float(o.selection.split()[-1]) == line and o is not s)
        assert s.probability + other.probability + s.push_probability == pytest.approx(
            1.0, abs=1e-3)


def test_metadata_carries_everything_the_spec_requires(priced: PZ.PricedMatch) -> None:
    md = priced.metadata
    for key in ("pa", "pb", "player_a_effective_n", "player_b_effective_n",
                "elo_serve_disagreement_pp", "venue_multiplier",
                "venue_measured", "format", "corrections", "calibration_maps",
                "thin_data_a", "thin_data_b", "truncation_point_games",
                "lines_beyond_truncation_not_priced"):
        assert key in md, key
    assert md["format"]["provenance"] in ("documented", "inferred")
    assert md["format"]["source"]
    assert md["player_a_effective_n"] > 0


def test_format_comes_from_rules_not_a_caller_flag(matches: pd.DataFrame) -> None:
    """Same players, different tournament-year, different deciding-set rule."""
    m = matches[(matches["tourney_code"] == "540") & matches["in_scope"]]
    old = m[m["season_file"] == 2015].iloc[0]
    new = m[m["season_file"] == 2022].iloc[0]
    a = PZ.price_match(old["winner_id"], old["loser_id"], "540", 2015,
                       "Grass", "Wimbledon", old["date"])
    b = PZ.price_match(new["winner_id"], new["loser_id"], "540", 2022,
                       "Grass", "Wimbledon", new["date"])
    assert a.metadata["format"]["final_set"] == "advantage"
    assert b.metadata["format"]["final_set"] == "tiebreak"
    assert b.metadata["format"]["final_tb_to"] == 10
    assert a.metadata["lines_beyond_truncation_not_priced"] is True
    assert b.metadata["lines_beyond_truncation_not_priced"] is False


def test_advantage_format_reports_its_truncation_point(matches: pd.DataFrame) -> None:
    m = matches[(matches["tourney_code"] == "540")
                & (matches["season_file"] == 2015) & matches["in_scope"]].iloc[0]
    p = PZ.price_match(m["winner_id"], m["loser_id"], "540", 2015, "Grass",
                       "Wimbledon", m["date"])
    md = p.metadata
    assert md["truncation_point_games"] > 60
    assert md["truncated_mass"] < C.TAIL_MASS_THRESHOLD
    lines = [float(s.selection.split()[-1]) for s in p.market("total_games")]
    assert max(lines) <= md["truncation_point_games"] + 1


def test_as_of_guarantee(matches: pd.DataFrame, sample: pd.Series) -> None:
    """The pricer's history contains nothing on or after its as-of date."""
    pricer = PZ.Pricer(sample["date"])
    assert pricer.history["date"].max() < sample["date"]


def test_pricing_a_holdout_date_is_refused() -> None:
    with pytest.raises(ValueError, match="holdout"):
        PZ.Pricer(C.HOLDOUT_CUTOFF + timedelta(days=30))
    with pytest.raises(ValueError):
        PZ.price_match("A", "B", "339", 2024, "Hard", None,
                       C.HOLDOUT_CUTOFF + timedelta(days=1))


def test_no_bookmaker_odds_anywhere_in_the_output(priced: PZ.PricedMatch) -> None:
    blob = str(priced.metadata) + str(priced.to_frame().to_dict())
    for token in ("B365", "PSW", "PSL", "MaxW", "AvgW", "bookmaker", "market_odds"):
        assert token not in blob


def test_unsupported_format_returns_no_selections(matches: pd.DataFrame) -> None:
    m = matches[matches["tourney_code"] == "7696"].iloc[0]
    p = PZ.price_match(m["winner_id"], m["loser_id"], "7696",
                       int(m["season_file"]), m["surface"], "Next Gen", m["date"])
    assert p.selections == []
    assert not p.metadata["format"]["supported"]
    assert "unsupported_format" in p.metadata


def test_determinism(sample: pd.Series) -> None:
    a = PZ.price_match(sample["winner_id"], sample["loser_id"],
                       sample["tourney_code"], int(sample["season_file"]),
                       sample["surface"], None, sample["date"])
    b = PZ.price_match(sample["winner_id"], sample["loser_id"],
                       sample["tourney_code"], int(sample["season_file"]),
                       sample["surface"], None, sample["date"])
    assert a.to_frame().equals(b.to_frame())
