"""Stage 1 validation gate — the closed form must be right, not plausible.

Every check below either enumerates the point space by brute force or
compares against point-level Monte Carlo from simulate.py. Nothing is
asserted from memory of what the numbers "should" be.
"""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from model import constants as C
from model import engine as E
from model import simulate as S
from model.rules import FormatSpec, rules_for

#: One spec per distinct deciding-set rule in the data.
FORMATS = {
    "advantage_bo5": rules_for("540", 2010),
    "tb12_bo5": rules_for("540", 2019),
    "tb10_bo5": rules_for("540", 2023),
    "tb7_bo5": rules_for("560", 2015),
    "tb7_bo3": rules_for("339", 2019),
}


# --- p_hold ---------------------------------------------------------------

def brute_force_game(p: float, max_points: int = 400) -> float:
    """P(server wins a game) by expanding the point space state by state.

    Independent of the closed form under test: no binomial coefficients and
    no deuce formula, just every reachable score with its probability, run
    out until the undecided mass is below 1e-15. (Expanding distinct point
    *sequences* instead would be 2^400 paths for the same answer.)
    """
    live = {(0, 0): 1.0}
    won = 0.0
    for _ in range(max_points):
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), prob in live.items():
            for na, nb, q in ((a + 1, b, prob * p), (a, b + 1, prob * (1 - p))):
                if na >= 4 and na - nb >= 2:
                    won += q
                elif nb >= 4 and nb - na >= 2:
                    pass
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + q
        live = nxt
        if sum(live.values()) < 1e-15:
            break
    return won


def test_p_hold_half_is_exactly_half() -> None:
    assert E.p_hold(0.5) == 0.5


def test_p_hold_monotonic() -> None:
    vals = [E.p_hold(p / 100) for p in range(1, 100)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("p", [0.35, 0.5, 0.55, 0.62, 0.7, 0.8])
def test_p_hold_matches_brute_force(p: float) -> None:
    assert E.p_hold(p) == pytest.approx(brute_force_game(p), abs=1e-12)


def test_p_hold_62_is_in_the_known_region() -> None:
    """~0.78, verified against enumeration rather than recalled."""
    assert E.p_hold(0.62) == pytest.approx(brute_force_game(0.62), abs=1e-12)
    assert 0.77 < E.p_hold(0.62) < 0.80


# --- tiebreak -------------------------------------------------------------

def brute_force_tiebreak(pa: float, pb: float, first_to: int,
                         max_points: int = 400) -> float:
    """Same state expansion for a tiebreak, serve rotation included."""
    live = {(0, 0): 1.0}
    won = 0.0
    for _ in range(max_points):
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), prob in live.items():
            p_point = pa if E._tb_server_is_a(a + b, True) else 1 - pb
            for na, nb, q in ((a + 1, b, prob * p_point),
                              (a, b + 1, prob * (1 - p_point))):
                if na >= first_to and na - nb >= 2:
                    won += q
                elif nb >= first_to and nb - na >= 2:
                    pass
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + q
        live = nxt
        if sum(live.values()) < 1e-15:
            break
    return won


@pytest.mark.parametrize("first_to", [7, 10])
@pytest.mark.parametrize("pa,pb", [(0.5, 0.5), (0.62, 0.6), (0.7, 0.55), (0.55, 0.75)])
def test_tiebreak_matches_brute_force(pa: float, pb: float, first_to: int) -> None:
    assert E.p_tiebreak(pa, pb, first_to) == pytest.approx(
        brute_force_tiebreak(pa, pb, first_to), abs=1e-12
    )


def test_tiebreak_symmetry() -> None:
    assert E.p_tiebreak(0.6, 0.6, 7) == pytest.approx(0.5, abs=1e-12)
    assert E.p_tiebreak(0.62, 0.58, 7) + E.p_tiebreak(0.58, 0.62, 7) == pytest.approx(
        1.0, abs=1e-12
    )


def test_serve_rotation() -> None:
    """A, BB, AA, BB, … — the standard tiebreak rotation."""
    got = [E._tb_server_is_a(i, True) for i in range(9)]
    assert got == [True, False, False, True, True, False, False, True, True]


# --- sets and matches -----------------------------------------------------

@pytest.mark.parametrize("name", list(FORMATS))
def test_set_distribution_is_a_distribution(name: str) -> None:
    spec = FORMATS[name]
    if not spec.supported:
        pytest.skip("short-set format is not priced")
    for deciding in (False, True):
        out = E.set_distribution(0.63, 0.6, spec, deciding=deciding)
        assert sum(out.scores.values()) + out.truncated_mass == pytest.approx(1.0, abs=1e-9)
        for (ga, gb, _), p in out.scores.items():
            assert p >= 0 and max(ga, gb) >= spec.games_to_win_set


@pytest.mark.parametrize("name", list(FORMATS))
@pytest.mark.parametrize("pa,pb", [(0.62, 0.60), (0.70, 0.55)])
def test_match_matches_monte_carlo(name: str, pa: float, pb: float) -> None:
    """Point-level MC (>=10^7 points) must reproduce the closed form.

    70,000 matches is 1.0-1.8 x 10^7 simulated points depending on format.
    """
    spec = FORMATS[name]
    n = 70_000
    sim = S.simulate_match(pa, pb, spec, n=n)
    exact = E.match_distribution(pa, pb, spec)

    assert sim.p_a == pytest.approx(exact.p_a, abs=4 * sim.se)

    exact_mean = sum(g * p for g, p in exact.total_games_pmf().items())
    games_se = sim.total_games.std() / n**0.5
    assert sim.total_games.mean() == pytest.approx(exact_mean, abs=4 * games_se)

    tb_se = (exact.tiebreak_any * (1 - exact.tiebreak_any) / n) ** 0.5
    assert sim.tiebreak_any.mean() == pytest.approx(exact.tiebreak_any, abs=4 * tb_se)


@pytest.mark.parametrize("name", list(FORMATS))
def test_match_symmetry(name: str) -> None:
    spec = FORMATS[name]
    pa, pb = 0.66, 0.58
    ab = E.match_distribution(pa, pb, spec)
    ba = E.match_distribution(pb, pa, spec, a_serves_first=False)
    assert ab.p_a + ba.p_a == pytest.approx(1.0, abs=1e-9)
    # swapping players mirrors the joint games distribution
    for (ga, gb), p in ab.games.items():
        assert ba.games.get((gb, ga), 0.0) == pytest.approx(p, abs=1e-9)


def test_equal_players_are_even() -> None:
    for spec in FORMATS.values():
        d = E.match_distribution(0.62, 0.62, spec)
        assert d.p_a == pytest.approx(0.5, abs=1e-9)


def test_more_sets_favour_the_better_player() -> None:
    bo3, bo5 = FORMATS["tb7_bo3"], FORMATS["tb7_bo5"]
    assert E.p_match(0.66, 0.60, bo5) > E.p_match(0.66, 0.60, bo3)


# --- tails, ladder, prices ------------------------------------------------

@pytest.mark.parametrize("name", list(FORMATS))
def test_minimum_games_has_no_mass_below_it(name: str) -> None:
    spec = FORMATS[name]
    d = E.match_distribution(0.62, 0.60, spec)
    pmf = d.total_games_pmf()
    need = spec.best_of // 2 + 1
    assert min(pmf) == need * spec.games_to_win_set
    rows = E.ladder(0.62, 0.60, spec)
    below = [r for r in rows if r["line"] < min(pmf)]
    assert all(r["p_under"] == 0.0 for r in below)


def test_advantage_truncation_is_below_threshold() -> None:
    spec = FORMATS["advantage_bo5"]
    for pa, pb in [(0.5, 0.5), (0.62, 0.6), (0.75, 0.55)]:
        d = E.match_distribution(pa, pb, spec)
        assert d.truncated_mass < C.TAIL_MASS_THRESHOLD
        assert sum(d.games.values()) + d.truncated_mass == pytest.approx(1.0, abs=1e-12)


def test_ladder_is_monotone_in_the_line() -> None:
    rows = E.ladder(0.63, 0.60, FORMATS["tb7_bo3"])
    overs = [r["p_over"] for r in rows]
    unders = [r["p_under"] for r in rows]
    assert all(b <= a + 1e-15 for a, b in zip(overs, overs[1:]))
    assert all(b >= a - 1e-15 for a, b in zip(unders, unders[1:]))


def test_ladder_probabilities_close_with_push() -> None:
    for r in E.ladder(0.63, 0.60, FORMATS["tb7_bo3"]):
        assert r["p_over"] + r["p_under"] + r["p_push"] == pytest.approx(1.0, abs=1e-9)
        if float(r["line"]).is_integer():
            assert r["p_push"] >= 0.0
        else:
            assert r["p_push"] == 0.0


def test_fair_price_push_handling() -> None:
    # no push: plain reciprocal
    assert E.fair_price(0.5) == pytest.approx(2.0)
    # a stake returned on a push must be priced out of the remaining book
    assert E.fair_price(0.45, push=0.10) == pytest.approx(0.9 / 0.45)


def test_set_and_games_distributions_agree() -> None:
    spec = FORMATS["tb7_bo3"]
    d = E.match_distribution(0.63, 0.60, spec)
    assert sum(d.sets.values()) + d.truncated_mass == pytest.approx(1.0, abs=1e-9)
    assert sum(d.games.values()) + d.truncated_mass == pytest.approx(1.0, abs=1e-9)
    assert d.p_a == pytest.approx(
        sum(p for (sa, sb), p in d.sets.items() if sa > sb), abs=1e-12
    )
    a_pmf = d.player_games_pmf(True)
    assert sum(a_pmf.values()) + d.truncated_mass == pytest.approx(1.0, abs=1e-9)


# --- historical targets ---------------------------------------------------

def test_historical_targets_exclude_suspect_matches() -> None:
    """Ground rule 8: a flagged score never becomes a validation target."""
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    targets = m[
        m["outcome"].eq("completed")
        & ~m["score_string_suspect"]
        & m["tourney_code"].eq("540")
        & m["season_file"].eq(2010)
    ]
    assert len(targets) > 100
    assert not targets["score_string_suspect"].any()

    # every observed set score must be reachable under the assigned format
    spec = rules_for("540", 2010)
    out = E.set_distribution(0.65, 0.62, spec, deciding=True)
    reachable = {(ga, gb) for (ga, gb, _) in out.scores}
    isner = targets[targets["score"].str.contains("70-68", na=False)]
    assert len(isner) == 1
    assert (70, 68) in reachable or (68, 70) in reachable


def test_unsupported_format_is_not_priced() -> None:
    spec = rules_for("7696", 2019)
    assert not spec.supported


def test_determinism() -> None:
    a = E.match_distribution(0.63, 0.6, FORMATS["tb7_bo3"])
    b = E.match_distribution(0.63, 0.6, FORMATS["tb7_bo3"])
    assert a.games == b.games and a.p_a == b.p_a
    s1 = S.simulate_match(0.63, 0.6, FORMATS["tb7_bo3"], n=2000)
    s2 = S.simulate_match(0.63, 0.6, FORMATS["tb7_bo3"], n=2000)
    assert s1.p_a == s2.p_a


def test_spec_drives_the_deciding_set_not_a_global_flag() -> None:
    """Same (pa, pb); only the deciding-set rule differs."""
    pa, pb = 0.64, 0.62
    adv = E.match_distribution(pa, pb, FORMATS["advantage_bo5"])
    tb10 = E.match_distribution(pa, pb, FORMATS["tb10_bo5"])
    assert adv.max_total_games > tb10.max_total_games
    custom = FormatSpec(best_of=3, final_set="tiebreak", final_tb_at=6,
                        final_tb_to=10, provenance="documented", source="test")
    assert E.p_match(pa, pb, custom) != E.p_match(pa, pb, FORMATS["tb7_bo3"])
