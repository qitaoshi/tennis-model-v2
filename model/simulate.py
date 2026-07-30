"""Point-level Monte Carlo, matching the engine's assumptions.

Two uses, per MODEL_PROMPT.md:
  a) verifying ``engine.py``'s closed form across every format variant,
  b) quantities with no closed form (ace counts and similar exotic props),
     which need a per-serve ace rate as an extra input.

This is a supporting module, not a stage. It consumes the same
:class:`~model.rules.FormatSpec` as the engine and is seeded, so results are
reproducible (ground rule 9).

Sample-size guidance: standard error on a probability estimate is
sqrt(p(1-p)/n). 50,000 matches — roughly 10^7 simulated points for a
best-of-three — gives ~0.002 on a near-even probability, which is the
resolution the Stage 1 tests compare against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import constants as C
from model.rules import FormatSpec


@dataclass(frozen=True)
class SimResult:
    """Outcome of a batch of simulated matches."""

    p_a: float
    games_a: np.ndarray
    games_b: np.ndarray
    tiebreak_any: np.ndarray
    sets_a: np.ndarray
    sets_b: np.ndarray
    n: int

    @property
    def total_games(self) -> np.ndarray:
        return self.games_a + self.games_b

    @property
    def se(self) -> float:
        """Standard error on ``p_a``."""
        return float(np.sqrt(self.p_a * (1 - self.p_a) / self.n))


def simulate_match(pa: float, pb: float, spec: FormatSpec, n: int = 50_000,
                   seed: int = C.MC_SEED, a_serves_first: bool = True) -> SimResult:
    """Simulate ``n`` matches point by point under ``spec``.

    Vectorized across matches: every simulation advances one point per
    iteration, so the loop runs for as many iterations as the longest match
    needs, not once per point per match.
    """
    rng = np.random.default_rng(seed)
    need = spec.best_of // 2 + 1
    target = spec.games_to_win_set

    sets_a = np.zeros(n, dtype=np.int32)
    sets_b = np.zeros(n, dtype=np.int32)
    games_a = np.zeros(n, dtype=np.int32)   # match totals
    games_b = np.zeros(n, dtype=np.int32)
    sg_a = np.zeros(n, dtype=np.int32)      # games in the current set
    sg_b = np.zeros(n, dtype=np.int32)
    pts_a = np.zeros(n, dtype=np.int32)
    pts_b = np.zeros(n, dtype=np.int32)
    server_a = np.full(n, a_serves_first)   # who serves the current game
    set_first_a = np.full(n, a_serves_first)
    in_tb = np.zeros(n, dtype=bool)
    tb_first_a = np.full(n, a_serves_first)
    tb_seen = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    while active.any():
        idx = np.flatnonzero(active)

        # --- who serves this point -------------------------------------
        srv = server_a[idx].copy()
        tb = in_tb[idx]
        if tb.any():
            played = pts_a[idx] + pts_b[idx]
            a_turn = ((played + 1) // 2) % 2 == 0
            srv = np.where(tb, a_turn == tb_first_a[idx], srv)

        p_srv = np.where(srv, pa, pb)
        srv_won = rng.random(idx.size) < p_srv
        a_won = srv == srv_won  # A won the point iff (A served and won) or (B served and lost)

        pts_a[idx] += a_won
        pts_b[idx] += ~a_won

        # --- tiebreak resolution ---------------------------------------
        if tb.any():
            final_to = spec.final_tb_to if spec.final_tb_to is not None else spec.tb_to
            tb_to = np.where(_is_deciding(spec, sets_a[idx], sets_b[idx]),
                             final_to, spec.tb_to)
            hi = np.maximum(pts_a[idx], pts_b[idx])
            lo = np.minimum(pts_a[idx], pts_b[idx])
            done = tb & (hi >= tb_to) & (hi - lo >= 2)
            if done.any():
                d = idx[done]
                won_a = pts_a[d] > pts_b[d]
                sg_a[d] += won_a
                sg_b[d] += ~won_a
                pts_a[d] = 0
                pts_b[d] = 0
                in_tb[d] = False
                _close_set(d, sets_a, sets_b, sg_a, sg_b, games_a, games_b,
                           server_a, set_first_a, need, active)

        # --- game resolution -------------------------------------------
        reg = ~in_tb[idx]
        if reg.any():
            hi = np.maximum(pts_a[idx], pts_b[idx])
            lo = np.minimum(pts_a[idx], pts_b[idx])
            done = reg & (hi >= 4) & (hi - lo >= 2) & active[idx]
            if done.any():
                d = idx[done]
                won_a = pts_a[d] > pts_b[d]
                pts_a[d] = 0
                pts_b[d] = 0
                sg_a[d] += won_a
                sg_b[d] += ~won_a
                server_a[d] = ~server_a[d]

                deciding = _is_deciding(spec, sets_a[d], sets_b[d])
                tb_at = np.where(deciding, spec.final_tb_at if spec.final_tb_at
                                 is not None else -1, spec.tb_at)
                advantage = deciding & (spec.final_set == "advantage")

                hi_g = np.maximum(sg_a[d], sg_b[d])
                lo_g = np.minimum(sg_a[d], sg_b[d])
                set_over = (hi_g >= target) & (hi_g - lo_g >= 2)
                start_tb = (~advantage) & (sg_a[d] == tb_at) & (sg_b[d] == tb_at)

                if start_tb.any():
                    t = d[start_tb]
                    in_tb[t] = True
                    tb_seen[t] = True
                    tb_first_a[t] = server_a[t]
                if set_over.any():
                    _close_set(d[set_over], sets_a, sets_b, sg_a, sg_b,
                               games_a, games_b, server_a, set_first_a, need,
                               active)

    return SimResult(
        p_a=float((sets_a > sets_b).mean()),
        games_a=games_a, games_b=games_b, tiebreak_any=tb_seen,
        sets_a=sets_a, sets_b=sets_b, n=n,
    )


def _is_deciding(spec: FormatSpec, sets_a: np.ndarray, sets_b: np.ndarray) -> np.ndarray:
    return (sets_a + sets_b) == spec.best_of - 1


def _close_set(d: np.ndarray, sets_a, sets_b, sg_a, sg_b, games_a, games_b,
               server_a, set_first_a, need: int, active) -> None:
    """Bank a finished set and set up the next one."""
    won_a = sg_a[d] > sg_b[d]
    sets_a[d] += won_a
    sets_b[d] += ~won_a
    games_a[d] += sg_a[d]
    games_b[d] += sg_b[d]
    # Server alternates every game, so after an even number of games the
    # player who opened the set opens the next one.
    even = (sg_a[d] + sg_b[d]) % 2 == 0
    set_first_a[d] = np.where(even, set_first_a[d], ~set_first_a[d])
    server_a[d] = set_first_a[d]
    sg_a[d] = 0
    sg_b[d] = 0
    active[d] = (sets_a[d] < need) & (sets_b[d] < need)


def simulate_aces(pa: float, ace_rate_a: float, pb: float, ace_rate_b: float,
                  spec: FormatSpec, n: int = 20_000,
                  seed: int = C.MC_SEED) -> np.ndarray:
    """Ace-count distribution for player A — no closed form exists.

    ``ace_rate_a`` is aces per point served. Aces are drawn as a thinning of
    A's won service points, which keeps the match outcome distribution
    identical to :func:`simulate_match`.
    """
    rng = np.random.default_rng(seed)
    res = simulate_match(pa, pb, spec, n=n, seed=seed)
    # A serves about half the games; each game is ~ 1 / (1 - p) * 4 points.
    serve_points = np.rint(
        (res.games_a + res.games_b) / 2 * 4 / max(1e-9, (1 - abs(pa - 0.5)))
    ).astype(int)
    return rng.binomial(serve_points, ace_rate_a)


def demo() -> None:
    """Worked example (ground rule 10)."""
    from model.engine import match_distribution
    from model.rules import rules_for

    for label, code, year in [("Wimbledon 2010 (advantage)", "540", 2010),
                              ("Wimbledon 2023 (10-pt TB)", "540", 2023),
                              ("Brisbane 2019 (bo3)", "339", 2019)]:
        spec = rules_for(code, year)
        sim = simulate_match(0.64, 0.60, spec, n=20_000)
        exact = match_distribution(0.64, 0.60, spec)
        print(f"{label}:")
        print(f"  P(A) sim {sim.p_a:.4f} +/- {sim.se:.4f}   exact {exact.p_a:.4f}")
        print(f"  mean games sim {sim.total_games.mean():.2f}   exact "
              f"{sum(g * p for g, p in exact.total_games_pmf().items()):.2f}")
        print(f"  P(tiebreak) sim {sim.tiebreak_any.mean():.4f}   exact "
              f"{exact.tiebreak_any:.4f}")


if __name__ == "__main__":
    demo()
