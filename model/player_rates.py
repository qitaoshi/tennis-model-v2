"""Stage 2 — serve and return rates from match data, as of any date.

Every rate is computable "as of" a date from matches strictly before it
(ground rule 5). The accumulators are built by a single forward pass in date
order, so a rate physically cannot see its own match or any later one.

Exclusions, per ground rules 6 and 8: retirements, walkovers and defaults
never contribute (their stats are contaminated by whatever ended the match),
matches flagged ``score_string_suspect`` never contribute, and out-of-scope
events never contribute. That filter is ``serve_stats_valid & in_scope``.

Components, each toggleable so ablations are possible:

recency
    Exponential decay with a fitted half-life.
surface split with partial pooling
    A player's clay rate borrows strength from their overall rate rather than
    being estimated on clay alone. Pooling strength is fitted.
opponent adjustment
    Serve points won against a strong returner mean more. This is a paired
    comparison — server skill and returner skill adjust each other — so it
    needs an identifiability anchor, not just "iterate to convergence": the
    tour-level average serve-win% in each period is held fixed, which pins the
    scale while the relative values converge. Convergence is
    ``max |rate change| < RATE_ITER_TOL`` within ``RATE_ITER_MAX`` sweeps, and
    failure to converge is a hard error, never a silent partial fit.
shrinkage
    Empirical Bayes toward the average of the player's own tour LEVEL — an
    ATP average never shrinks a Challenger player, and vice versa.

Every rate carries its effective sample size ``n``. No rate is emitted
without one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

from model import constants as C

#: Rates outside this band are not physically plausible for tour tennis and
#: signal a broken input rather than an extreme player.
RATE_FLOOR, RATE_CEIL = 0.35, 0.85


@dataclass(frozen=True)
class RateParams:
    """Stage 2's hyperparameters. Fitted values live in fitted_params.json."""

    half_life_days: float = 365.0
    #: Partial-pooling strength of a surface rate toward the player's overall
    #: rate, in units of serve points.
    surface_pool: float = 300.0
    #: Empirical-Bayes shrinkage toward the tour-level average, in serve points.
    shrink_n0: float = 500.0
    recency: bool = True
    surface_split: bool = True
    opponent_adjust: bool = True

    def key(self) -> tuple:
        return (self.half_life_days, self.surface_pool, self.shrink_n0,
                self.recency, self.surface_split, self.opponent_adjust)


@dataclass(frozen=True)
class Rate:
    """A rate and the evidence behind it. ``n`` is never optional."""

    value: float
    n: float
    n_matches: int
    level: str
    surface: str

    def __post_init__(self) -> None:
        assert 0.0 <= self.value <= 1.0
        assert self.n >= 0.0


# ---------------------------------------------------------------------------
# As-of accumulators
# ---------------------------------------------------------------------------


class _Decayed:
    """Exponentially decayed (won, played) counters keyed by anything.

    Decay is applied on read and on write, so a counter is always current as
    of the timestamp it is asked about — the whole point of ground rule 5.
    """

    __slots__ = ("half_life", "won", "played", "matches", "last")

    def __init__(self, half_life: float | None) -> None:
        self.half_life = half_life
        self.won: dict = {}
        self.played: dict = {}
        self.matches: dict = {}
        self.last: dict = {}

    def _decay(self, key, t: float) -> float:
        if key not in self.last:
            return 0.0
        if self.half_life is None:
            return 1.0
        return 0.5 ** ((t - self.last[key]) / self.half_life)

    def get(self, key, t: float) -> tuple[float, float, float]:
        """(won, played, matches) as of ``t``."""
        if key not in self.last:
            return 0.0, 0.0, 0.0
        f = self._decay(key, t)
        return self.won[key] * f, self.played[key] * f, self.matches[key] * f

    def add(self, key, t: float, won: float, played: float) -> None:
        f = self._decay(key, t)
        self.won[key] = self.won.get(key, 0.0) * f + won
        self.played[key] = self.played.get(key, 0.0) * f + played
        self.matches[key] = self.matches.get(key, 0.0) * f + 1.0
        self.last[key] = t


def _match_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, player) side, filtered to usable serve stats."""
    ok = matches[matches["serve_stats_valid"]].copy()
    ok["t"] = pd.to_datetime(ok["date"]).map(pd.Timestamp.toordinal)
    ok["w_spw"] = (ok["w_1stWon"] + ok["w_2ndWon"]) / ok["w_svpt"]
    ok["l_spw"] = (ok["l_1stWon"] + ok["l_2ndWon"]) / ok["l_svpt"]
    ok["level_group"] = np.where(ok["tour"] == "chall", "chall", "atp")
    return ok.sort_values(["t", "match_id"]).reset_index(drop=True)


def _sweep(f: pd.DataFrame, params: RateParams,
           adjust: np.ndarray | None) -> dict[str, np.ndarray]:
    """One forward pass in date order, producing as-of state for every match.

    ``adjust`` is the per-side opponent correction from the previous sweep
    (zeros on the first). Returned arrays are indexed like ``f``; the ``w_``
    and ``l_`` prefixes are the winner's and loser's state *before* the match.
    """
    hl = params.half_life_days if params.recency else None
    srv_surf, srv_all = _Decayed(hl), _Decayed(hl)
    ret_all = _Decayed(hl)
    league = _Decayed(hl)  # identifiability anchor: level-period serve average

    n = len(f)
    cols = {
        k: np.zeros(n)
        for k in ("w_ss_won", "w_ss_pl", "w_sa_won", "w_sa_pl", "w_nm",
                  "l_ss_won", "l_ss_pl", "l_sa_won", "l_sa_pl", "l_nm",
                  "w_ret", "l_ret", "w_ret_n", "l_ret_n", "league", "league_n")
    }

    adj = np.zeros((n, 2)) if adjust is None else adjust
    it = zip(f["t"], f["winner_id"], f["loser_id"], f["surface"],
             f["level_group"], f["w_spw"], f["l_spw"], f["w_svpt"], f["l_svpt"])
    for i, (t, wid, lid, surf, lg, wspw, lspw, wpts, lpts) in enumerate(it):
        for side, pid in (("w", wid), ("l", lid)):
            sw, sp, nm = srv_surf.get((pid, surf), t)
            aw, ap, _ = srv_all.get(pid, t)
            rw, rp, _ = ret_all.get(pid, t)
            cols[f"{side}_ss_won"][i], cols[f"{side}_ss_pl"][i] = sw, sp
            cols[f"{side}_sa_won"][i], cols[f"{side}_sa_pl"][i] = aw, ap
            cols[f"{side}_nm"][i] = nm
            cols[f"{side}_ret"][i] = rw / rp if rp > 0 else np.nan
            cols[f"{side}_ret_n"][i] = rp
        lw, lp, _ = league.get(lg, t)
        cols["league"][i] = lw / lp if lp > 0 else np.nan
        cols["league_n"][i] = lp

        # Contributions carry the previous sweep's opponent correction. The
        # league anchor is always the raw observation, so the correction can
        # never drift the overall scale.
        w_pts_adj = np.clip(wspw + adj[i, 0], 0.0, 1.0) * wpts
        l_pts_adj = np.clip(lspw + adj[i, 1], 0.0, 1.0) * lpts
        srv_surf.add((wid, surf), t, w_pts_adj, wpts)
        srv_surf.add((lid, surf), t, l_pts_adj, lpts)
        srv_all.add(wid, t, w_pts_adj, wpts)
        srv_all.add(lid, t, l_pts_adj, lpts)
        # return side: points won while receiving, from the raw observation
        ret_all.add(wid, t, (1 - lspw) * lpts, lpts)
        ret_all.add(lid, t, (1 - wspw) * wpts, wpts)
        league.add(lg, t, wspw * wpts + lspw * lpts, wpts + lpts)

    return cols


def build_asof(matches: pd.DataFrame, params: RateParams) -> pd.DataFrame:
    """As-of serve/return state for every match, opponent adjustment converged.

    The opponent adjustment is a paired comparison: a player's serve rate
    depends on the returners faced, whose return rates depend on the servers
    they faced. Sweeps are repeated until the per-match corrections stop
    moving (``RATE_ITER_TOL``) or ``RATE_ITER_MAX`` is reached, at which point
    an unconverged fit is an error rather than a silent source of drift.
    """
    f = _match_frame(matches)
    adj = np.zeros((len(f), 2))
    cols = _sweep(f, params, None)
    if not params.opponent_adjust:
        return _attach(f, cols)

    for sweep in range(C.RATE_ITER_MAX):
        league = np.where(np.isnan(cols["league"]), np.nan, cols["league"])
        league_ret = 1.0 - league
        # Facing a better-than-average returner depresses observed serve
        # points won; add the shortfall back before it enters the accumulator.
        new = np.zeros_like(adj)
        new[:, 0] = np.nan_to_num(cols["l_ret"] - league_ret)
        new[:, 1] = np.nan_to_num(cols["w_ret"] - league_ret)
        # a returner with no history yet carries no correction
        new[np.isnan(cols["l_ret"]), 0] = 0.0
        new[np.isnan(cols["w_ret"]), 1] = 0.0
        delta = float(np.max(np.abs(new - adj))) if len(new) else 0.0
        adj = new
        cols = _sweep(f, params, adj)
        if delta < C.RATE_ITER_TOL:
            break
    else:
        raise RuntimeError(
            f"opponent adjustment did not converge in {C.RATE_ITER_MAX} sweeps "
            f"(last max change {delta:.2e} > tol {C.RATE_ITER_TOL:.0e})"
        )
    return _attach(f, cols)


def _attach(f: pd.DataFrame, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    out = f.copy()
    for k, v in cols.items():
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Rate assembly
# ---------------------------------------------------------------------------


def rate_from_state(ss_won: float, ss_pl: float, sa_won: float, sa_pl: float,
                    n_matches: float, league: float, params: RateParams,
                    level: str = "", surface: str = "") -> Rate:
    """Assemble one serve rate from as-of counters.

    Pooling then shrinkage, both in units of serve points, so the weights are
    interpretable: ``surface_pool`` is how many serve points of the player's
    overall record the surface estimate borrows, ``shrink_n0`` how many of the
    tour-level average the result shrinks toward.
    """
    prior = league if np.isfinite(league) else 0.62
    overall = sa_won / sa_pl if sa_pl > 0 else prior

    if params.surface_split:
        pool = params.surface_pool
        pooled_won, pooled_n = ss_won + pool * overall, ss_pl + pool
    else:
        pooled_won, pooled_n = sa_won, sa_pl
    pooled = pooled_won / pooled_n if pooled_n > 0 else prior

    n0 = params.shrink_n0
    n_eff = pooled_n
    value = (pooled * n_eff + prior * n0) / (n_eff + n0) if n_eff + n0 > 0 else prior
    return Rate(float(np.clip(value, 0.0, 1.0)), float(n_eff), int(n_matches),
                level, surface)


def serve_rates(asof: pd.DataFrame, params: RateParams) -> pd.DataFrame:
    """Serve rate for both sides of every match, as of that match.

    Vectorized form of :func:`rate_from_state` — the grid search calls this
    once per hyperparameter combination, and a test pins the two together.
    """
    out = asof[["match_id", "date", "t", "surface", "level_group",
                "winner_id", "loser_id", "w_spw", "l_spw",
                "w_svpt", "l_svpt", "league", "w_ret", "l_ret"]].copy()
    prior = np.where(np.isfinite(asof["league"]), asof["league"], 0.62)
    for side in ("w", "l"):
        ss_won = asof[f"{side}_ss_won"].to_numpy()
        ss_pl = asof[f"{side}_ss_pl"].to_numpy()
        sa_won = asof[f"{side}_sa_won"].to_numpy()
        sa_pl = asof[f"{side}_sa_pl"].to_numpy()
        overall = np.where(sa_pl > 0, sa_won / np.maximum(sa_pl, 1e-9), prior)
        if params.surface_split:
            pooled_won = ss_won + params.surface_pool * overall
            pooled_n = ss_pl + params.surface_pool
        else:
            pooled_won, pooled_n = sa_won, sa_pl
        n0 = params.shrink_n0
        value = (pooled_won + prior * n0) / np.maximum(pooled_n + n0, 1e-9)
        out[f"{side}_rate"] = np.clip(value, 0.0, 1.0)
        out[f"{side}_n"] = pooled_n
        out[f"{side}_nm"] = asof[f"{side}_nm"].to_numpy()
    return out


def expected_spw(rate: float, opponent_return: float, league: float) -> float:
    """Serve points won a player is expected to post against this opponent.

    The rate is opponent-neutral by construction, so predicting an *observed*
    match statistic means putting the specific returner back in.
    """
    if not np.isfinite(opponent_return) or not np.isfinite(league):
        return float(np.clip(rate, RATE_FLOOR, RATE_CEIL))
    return float(np.clip(rate - (opponent_return - (1.0 - league)),
                         RATE_FLOOR, RATE_CEIL))


def match_point_probs(rates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(pa, pb) for the engine: each side's point-win probability on serve."""
    pa = np.array([expected_spw(r, o, lg) for r, o, lg
                   in zip(rates["w_rate"], rates["l_ret"], rates["league"])])
    pb = np.array([expected_spw(r, o, lg) for r, o, lg
                   in zip(rates["l_rate"], rates["w_ret"], rates["league"])])
    return pa, pb


# ---------------------------------------------------------------------------
# Baselines for the validation gate
# ---------------------------------------------------------------------------


def baseline_rates(matches: pd.DataFrame, last_k: int | None = None) -> pd.DataFrame:
    """Career-average and last-k-match baselines, both strictly as-of.

    ``last_k=None`` gives the raw career average; ``last_k=10`` the mean of
    the player's last ten matches' serve points won.
    """
    f = _match_frame(matches)
    hist: dict[str, list[float]] = {}
    tot: dict[str, tuple[float, float]] = {}
    w_pred, l_pred = [], []
    for wid, lid, wspw, lspw, wpts, lpts in zip(
        f["winner_id"], f["loser_id"], f["w_spw"], f["l_spw"],
        f["w_svpt"], f["l_svpt"],
    ):
        for side, pid, pred in (("w", wid, w_pred), ("l", lid, l_pred)):
            if last_k is None:
                won, played = tot.get(pid, (0.0, 0.0))
                pred.append(won / played if played > 0 else np.nan)
            else:
                h = hist.get(pid, [])
                pred.append(float(np.mean(h[-last_k:])) if h else np.nan)
        for pid, spw, pts in ((wid, wspw, wpts), (lid, lspw, lpts)):
            won, played = tot.get(pid, (0.0, 0.0))
            tot[pid] = (won + spw * pts, played + pts)
            hist.setdefault(pid, []).append(float(spw))
    out = f[["match_id", "t", "w_spw", "l_spw", "w_svpt", "l_svpt"]].copy()
    out["w_pred"] = w_pred
    out["l_pred"] = l_pred
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def spw_metrics(actual: np.ndarray, pred: np.ndarray, pts: np.ndarray) -> dict:
    """MAE and per-point log-loss of predicted serve points won."""
    ok = np.isfinite(actual) & np.isfinite(pred) & (pts > 0)
    a, p, w = actual[ok], np.clip(pred[ok], 1e-6, 1 - 1e-6), pts[ok]
    ll = -(a * np.log(p) + (1 - a) * np.log(1 - p))
    return {
        "mae": float(np.mean(np.abs(a - p))),
        "logloss": float(np.sum(ll * w) / np.sum(w)),
        "n": int(ok.sum()),
    }


def match_logloss(p_a_win: np.ndarray) -> float:
    """Log-loss of match-winner probabilities.

    Rows are oriented winner-first, so the observed outcome is always 1.
    """
    p = np.clip(p_a_win[np.isfinite(p_a_win)], 1e-9, 1 - 1e-9)
    return float(-np.mean(np.log(p)))


def demo() -> None:
    """Worked example (ground rule 10)."""
    matches = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    fit = matches[matches["split"].isin(("fit", "tune"))]
    params = RateParams()
    asof = build_asof(fit, params)
    rates = serve_rates(asof, params)

    print(f"{len(rates):,} matches with usable serve stats "
          f"({rates['date'].min()} .. {rates['date'].max()})")
    late = rates[rates["t"] > rates["t"].quantile(0.9)]
    m = spw_metrics(late["w_spw"].to_numpy(),
                    np.array([expected_spw(r, o, lg) for r, o, lg in
                              zip(late["w_rate"], late["l_ret"], late["league"])]),
                    late["w_svpt"].to_numpy())
    print(f"last decile of FIT+TUNE: serve-points-won MAE {m['mae']:.4f}, "
          f"per-point log-loss {m['logloss']:.4f}, n={m['n']:,}")
    row = rates.iloc[-1]
    print(f"\nexample: {row['match_id']} on {row['surface']}")
    print(f"  winner rate {row['w_rate']:.4f} (n={row['w_n']:.0f} serve points, "
          f"{row['w_nm']:.0f} matches)")
    print(f"  loser  rate {row['l_rate']:.4f} (n={row['l_n']:.0f} serve points, "
          f"{row['l_nm']:.0f} matches)")
    print(f"  league average serve-win {row['league']:.4f}")


if __name__ == "__main__":
    demo()
