"""Stage 1 — closed-form tennis scoring math (Barnett & Clarke style).

Inputs are ``pa`` and ``pb``, each player's probability of winning a point on
their OWN serve, plus a :class:`~model.rules.FormatSpec`. Points are assumed
iid within a match; the systematic ways that assumption fails are corrected in
Stage 7, not here.

Everything is closed form — recursions and exact dynamic programming over
score states, no simulation. ``simulate.py`` exists to verify this module, not
to produce its numbers.

Advantage-set formats have an unbounded games tail. It is truncated once the
remaining mass falls below ``constants.TAIL_MASS_THRESHOLD``. **That
truncation point is the ladder's upper bound**: there is no further "true"
tail beyond it, and any requested line past it is reported as not priced
rather than silently omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from model import constants as C
from model.rules import FormatSpec

#: Probabilities below this are dropped from a distribution during dynamic
#: programming. Far below the tail threshold, because thousands of pruned
#: states accumulate into the truncated mass that must stay under it.
_PRUNE = C.TAIL_MASS_THRESHOLD / 1e5


@lru_cache(maxsize=4096)
def p_hold(p: float) -> float:
    """P(server wins a game) given point-win probability ``p`` on serve.

    Sum over the ways to reach 4-0/4-1/4-2 plus the deuce branch, where from
    deuce the server wins with p² / (p² + (1-p)²) — the standard two-point
    block recursion.
    """
    q = 1.0 - p
    deuce = 0.0 if p in (0.0, 1.0) else p * p / (p * p + q * q)
    return (
        p**4                      # 4-0
        + 4 * p**4 * q            # 4-1
        + 10 * p**4 * q**2        # 4-2
        + 20 * p**3 * q**3 * deuce  # from deuce
    )


def _tb_server_is_a(points_played: int, a_serves_first: bool) -> bool:
    """Who serves point ``points_played`` (0-based) of a tiebreak.

    One point, then alternating pairs: A, BB, AA, BB, … when A opens.
    """
    a_turn = ((points_played + 1) // 2) % 2 == 0
    return a_turn == a_serves_first


@lru_cache(maxsize=8192)
def p_tiebreak(pa: float, pb: float, first_to: int = 7,
               a_serves_first: bool = True) -> float:
    """P(A wins a tiebreak), win by two, accounting for serve rotation.

    ``first_to`` is 7 or 10. From the level score (``first_to - 1`` each) the
    continuation is closed form: over any two consecutive points each player
    serves exactly one, so A wins the tiebreak with w / (w + l) where
    w = pa(1-pb) and l = (1-pa)pb — independent of who serves first from the
    level score.
    """
    w = pa * (1.0 - pb)
    l = (1.0 - pa) * pb
    level = 0.5 if w + l == 0 else w / (w + l)

    memo: dict[tuple[int, int], float] = {}

    def rec(a: int, b: int) -> float:
        if a >= first_to and a - b >= 2:
            return 1.0
        if b >= first_to and b - a >= 2:
            return 0.0
        if a >= first_to - 1 and b >= first_to - 1:
            return level
        key = (a, b)
        if key in memo:
            return memo[key]
        p_point = pa if _tb_server_is_a(a + b, a_serves_first) else 1.0 - pb
        memo[key] = p_point * rec(a + 1, b) + (1.0 - p_point) * rec(a, b + 1)
        return memo[key]

    return rec(0, 0)


@dataclass(frozen=True)
class SetOutcome:
    """Distribution over a single set's outcomes.

    ``scores`` maps (games won by A, games won by B, tiebreak played) to
    probability. ``truncated_mass`` is the advantage-set tail dropped past
    ``constants.TAIL_MASS_THRESHOLD``.
    """

    scores: dict[tuple[int, int, bool], float]
    truncated_mass: float = 0.0

    @property
    def p_a(self) -> float:
        return sum(p for (ga, gb, _), p in self.scores.items() if ga > gb)

    @property
    def p_tiebreak(self) -> float:
        return sum(p for (_, _, tb), p in self.scores.items() if tb)


def set_distribution(pa: float, pb: float, spec: FormatSpec,
                     a_serves_first: bool = True,
                     deciding: bool = False) -> SetOutcome:
    """Exact distribution over set scores by DP over game states.

    The deciding set uses the format's deciding-set rule — advantage
    continuation or a tiebreak at the format's game score, to the format's
    point target — rather than a global flag.
    """
    hold_a, hold_b = p_hold(pa), p_hold(pb)
    target = spec.games_to_win_set
    tb_at = spec.final_tb_at if deciding else spec.tb_at
    tb_to = spec.final_tb_to if deciding else spec.tb_to
    advantage = deciding and spec.final_set == "advantage"
    # Where a tiebreak is played, a set ending at tb_at+1 games is the
    # tiebreak; below that the set was won outright.
    tb_games = None if advantage else tb_at

    scores: dict[tuple[int, int, bool], float] = {}
    # states[(ga, gb)] = probability of reaching that game score
    states = {(0, 0): 1.0}
    truncated = 0.0

    while states:
        nxt: dict[tuple[int, int], float] = {}
        for (ga, gb), prob in states.items():
            server_is_a = ((ga + gb) % 2 == 0) == a_serves_first
            p_a_game = hold_a if server_is_a else 1.0 - hold_b
            for won_by_a, p_game in ((True, p_a_game), (False, 1.0 - p_a_game)):
                na, nb = (ga + 1, gb) if won_by_a else (ga, gb + 1)
                mass = prob * p_game
                if mass < _PRUNE:
                    truncated += mass
                    continue
                hi, lo = max(na, nb), min(na, nb)
                if tb_games is not None and na == nb == tb_games:
                    # tiebreak decides the set
                    p_tb_a = p_tiebreak(
                        pa, pb, tb_to,
                        a_serves_first=((na + nb) % 2 == 0) == a_serves_first,
                    )
                    scores[(na + 1, nb, True)] = scores.get((na + 1, nb, True), 0.0) + mass * p_tb_a
                    scores[(na, nb + 1, True)] = scores.get((na, nb + 1, True), 0.0) + mass * (1 - p_tb_a)
                elif hi >= target and hi - lo >= 2:
                    scores[(na, nb, False)] = scores.get((na, nb, False), 0.0) + mass
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + mass
        states = nxt

    return SetOutcome(scores, truncated)


def p_set(pa: float, pb: float, spec: FormatSpec, a_serves_first: bool = True,
          deciding: bool = False) -> float:
    """P(A wins a set)."""
    return set_distribution(pa, pb, spec, a_serves_first, deciding).p_a


@dataclass(frozen=True)
class MatchDistribution:
    """Everything the closed form knows about one match.

    ``games`` maps (games won by A, games won by B) to probability;
    ``sets`` maps (sets won by A, sets won by B) to probability;
    ``tiebreak_any`` is P(at least one tiebreak is played).
    """

    p_a: float
    sets: dict[tuple[int, int], float]
    games: dict[tuple[int, int], float]
    tiebreak_any: float
    truncated_mass: float
    spec: FormatSpec = field(repr=False, default=None)

    @property
    def max_total_games(self) -> int:
        """The ladder's upper bound: no line beyond this is priced."""
        return max(ga + gb for ga, gb in self.games)

    def total_games_pmf(self) -> dict[int, float]:
        pmf: dict[int, float] = {}
        for (ga, gb), p in self.games.items():
            pmf[ga + gb] = pmf.get(ga + gb, 0.0) + p
        return dict(sorted(pmf.items()))

    def player_games_pmf(self, player_a: bool = True) -> dict[int, float]:
        pmf: dict[int, float] = {}
        for (ga, gb), p in self.games.items():
            g = ga if player_a else gb
            pmf[g] = pmf.get(g, 0.0) + p
        return dict(sorted(pmf.items()))


def match_distribution(pa: float, pb: float, spec: FormatSpec,
                       a_serves_first: bool = True) -> MatchDistribution:
    """Exact joint distribution over sets, games and tiebreak occurrence.

    Who serves first in a set depends on the parity of the previous set's
    total games, so that is carried through the DP rather than assumed.
    """
    need = spec.best_of // 2 + 1
    # per-set distributions, cached by (serves first, is deciding)
    cache: dict[tuple[bool, bool], SetOutcome] = {}

    def sets_out(first: bool, deciding: bool) -> SetOutcome:
        if (first, deciding) not in cache:
            cache[(first, deciding)] = set_distribution(pa, pb, spec, first, deciding)
        return cache[(first, deciding)]

    # state: (sets_a, sets_b, games_a, games_b, a_serves_first, any_tb)
    states = {(0, 0, 0, 0, a_serves_first, False): 1.0}
    sets_dist: dict[tuple[int, int], float] = {}
    games: dict[tuple[int, int], float] = {}
    tiebreak_any = 0.0
    truncated = 0.0

    while states:
        nxt: dict[tuple[int, int, int, int, bool, bool], float] = {}
        for (sa, sb, ga, gb, first, tb_seen), prob in states.items():
            deciding = (sa + sb) == spec.best_of - 1
            out = sets_out(first, deciding)
            truncated += prob * out.truncated_mass
            for (sga, sgb, tb), p_set_score in out.scores.items():
                mass = prob * p_set_score
                if mass < _PRUNE:
                    truncated += mass
                    continue
                nsa, nsb = (sa + 1, sb) if sga > sgb else (sa, sb + 1)
                nga, ngb = ga + sga, gb + sgb
                ntb = tb_seen or tb
                # server alternates every game: after an even number of games
                # the player who opened the set opens the next one
                nfirst = first if (sga + sgb) % 2 == 0 else not first
                if nsa == need or nsb == need:
                    sets_dist[(nsa, nsb)] = sets_dist.get((nsa, nsb), 0.0) + mass
                    games[(nga, ngb)] = games.get((nga, ngb), 0.0) + mass
                    if ntb:
                        tiebreak_any += mass
                else:
                    key = (nsa, nsb, nga, ngb, nfirst, ntb)
                    nxt[key] = nxt.get(key, 0.0) + mass
        states = nxt

    p_a = sum(p for (sa, sb), p in sets_dist.items() if sa > sb)
    return MatchDistribution(p_a, sets_dist, games, tiebreak_any, truncated, spec)


def p_match(pa: float, pb: float, spec: FormatSpec,
            a_serves_first: bool = True) -> float:
    """P(A wins the match)."""
    return match_distribution(pa, pb, spec, a_serves_first).p_a


def total_games_distribution(pa: float, pb: float, spec: FormatSpec,
                             a_serves_first: bool = True) -> dict[int, float]:
    """Exact pmf over total games played, truncated per the tail threshold."""
    return match_distribution(pa, pb, spec, a_serves_first).total_games_pmf()


def fair_price(p: float, push: float = 0.0) -> float:
    """Fair decimal price for a probability, with push handling.

    A whole-number line can tie, returning the stake. Breaking even then
    requires ``p * price + push = 1``, i.e. price = (1 - push) / p.
    """
    if p <= 0.0:
        return float("inf")
    return (1.0 - push) / p


def ladder(pa: float, pb: float, spec: FormatSpec,
           a_serves_first: bool = True) -> list[dict]:
    """Every over/under total-games line, from the minimum to the truncation point.

    Whole-number lines carry their push probability; half lines cannot push.
    This is the debug view — ``price.py`` consumes the same numbers.
    """
    dist = match_distribution(pa, pb, spec, a_serves_first)
    pmf = dist.total_games_pmf()
    lo, hi = min(pmf), max(pmf)
    rows = []
    for twice in range(2 * lo - 1, 2 * hi + 2):
        line = twice / 2
        over = sum(p for g, p in pmf.items() if g > line)
        under = sum(p for g, p in pmf.items() if g < line)
        push = pmf.get(int(line), 0.0) if float(line).is_integer() else 0.0
        rows.append({
            "line": line,
            "p_over": over,
            "p_under": under,
            "p_push": push,
            "price_over": fair_price(over, push),
            "price_under": fair_price(under, push),
        })
    return rows


def demo() -> None:
    """Worked example (ground rule 10)."""
    from model.rules import rules_for

    print(f"p_hold(0.50) = {p_hold(0.50):.6f}")
    print(f"p_hold(0.62) = {p_hold(0.62):.6f}")
    print(f"p_tiebreak(0.62, 0.60, 7)  = {p_tiebreak(0.62, 0.60, 7):.6f}")
    print(f"p_tiebreak(0.62, 0.60, 10) = {p_tiebreak(0.62, 0.60, 10):.6f}")

    for label, code, year in [("Wimbledon 2010 (advantage)", "540", 2010),
                              ("Wimbledon 2023 (10-pt TB)", "540", 2023),
                              ("Brisbane 2019 (bo3)", "339", 2019)]:
        spec = rules_for(code, year)
        d = match_distribution(0.64, 0.60, spec)
        pmf = d.total_games_pmf()
        print(f"\n{label}: bo{spec.best_of}, {spec.final_set} [{spec.provenance}]")
        print(f"  P(A wins) = {d.p_a:.4f}   fair price {fair_price(d.p_a):.3f}")
        print(f"  P(any tiebreak) = {d.tiebreak_any:.4f}")
        print(f"  total games: min {min(pmf)}, max {d.max_total_games}, "
              f"mean {sum(g * p for g, p in pmf.items()):.2f}, "
              f"truncated mass {d.truncated_mass:.2e}")
        print(f"  set betting: "
              + ", ".join(f"{sa}-{sb}: {p:.3f}" for (sa, sb), p in sorted(d.sets.items())))
        mid = [r for r in ladder(0.64, 0.60, spec) if abs(r["line"] - 22.5) < 3]
        for r in mid[:4]:
            print(f"  line {r['line']:5.1f}  over {r['p_over']:.4f} @ "
                  f"{r['price_over']:.3f}   under {r['p_under']:.4f} @ "
                  f"{r['price_under']:.3f}")


if __name__ == "__main__":
    demo()
