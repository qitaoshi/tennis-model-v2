"""Match-context adjustment: fatigue, workload, and entry route.

Every stage before this one describes a player's *ability*: how well they
serve, how good they are on clay, how strong the field is. None of them
describe the *state the player walks on court in*. Two matches with identical
ratings are not the same match if one player is fresh and the other finished a
three-hour quarter-final eighteen hours ago.

The features, all computed strictly as-of the match being predicted:

* ``rest_days`` — days since the player's previous match, capped, because the
  difference between three weeks off and four is nothing while the difference
  between one day and three is real.
* ``load7`` — minutes played in the previous seven days, the accumulated cost
  of getting this far in the draw.
* ``is_qualifier`` — entered through qualifying or as a lucky loser, so has
  played two or three extra matches this week before the main draw started.

They are applied as an additive shift in logit space to the serve-point
probability the earlier stages produce, with one coefficient per feature
fitted by maximum likelihood against observed serve points won. Logit space
rather than raw, so the adjustment cannot push a probability outside (0, 1)
and so its effect is proportionally larger where there is more room to move.

Nothing here is fitted in a module body: coefficients live in
``fitted_params.json`` under ``fatigue``, written by
``scripts/fit_fatigue.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Rest beyond this many days is indistinguishable from "fully recovered".
REST_CAP_DAYS: float = 21.0

#: Entry codes that mean the player played extra matches to get into the draw.
QUALIFIER_ENTRIES = ("Q", "LL")

FEATURES = ("rest_days", "load7", "is_qualifier")

#: Canonical draw order. TML records only ``tourney_date`` — the tournament
#: START date — so every match in an event shares one date. Taken literally
#: that makes "days since last match" read 0 for every within-tournament match
#: (49% of the data) and piles a whole week of minutes onto a single day,
#: which is exactly the information this module exists to use. Rounds are
#: therefore spread one day apart from the tournament's own first round.
#: Approximate: a slam plays early rounds over two days, and rain moves
#: things. It is far closer to the truth than "the entire draw happened on
#: Monday".
ROUND_ORDER = ("RR", "R128", "R64", "R32", "R16", "QF", "SF", "F")

#: Standardisation constants. These are scale choices, not fitted values —
#: they exist so the coefficients come out in comparable units and so a
#: player's feature vector does not depend on which slice of data is loaded.
SCALE = {"rest_days": REST_CAP_DAYS, "load7": 300.0, "is_qualifier": 1.0}


@dataclass(frozen=True)
class FatigueParams:
    """Fitted logit-space coefficients, one per feature."""

    rest_days: float = 0.0
    load7: float = 0.0
    is_qualifier: float = 0.0
    enabled: bool = True

    def vector(self) -> np.ndarray:
        return np.array([self.rest_days, self.load7, self.is_qualifier])


def match_day(matches: pd.DataFrame) -> np.ndarray:
    """Approximate calendar day of each match, as an ordinal.

    ``tourney_date`` plus one day per round completed, counted from the
    earliest round that tournament actually played — so a 32-draw's first
    round is day 0 of that event, not day 2 because a 128-draw exists
    somewhere. See ``ROUND_ORDER`` for why this beats using tourney_date
    directly.

    An unrecognised round contributes no offset rather than raising: the
    fallback costs this one match its within-event resolution, where a raise
    would cost the whole run.
    """
    rank = {r: i for i, r in enumerate(ROUND_ORDER)}
    base = np.array([d.toordinal() for d in matches["date"]], dtype=float)
    r = matches["round"].astype(str).map(rank)
    key = (matches["tourney_id"].astype(str) + "|"
           + matches["season_file"].astype(str) + "|" + matches["tour"].astype(str))
    first = r.groupby(key.to_numpy()).transform("min")
    off = (r - first).fillna(0.0).to_numpy(dtype=float)
    return base + off


def build_context(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-match, per-side context features, computed only from the past.

    Returns one row per match with ``w_`` and ``l_`` prefixed columns.

    The as-of discipline matters more here than anywhere else in the model,
    because the obvious implementation leaks: a player's rest before match N
    is a function of match N-1, and a naive groupby-shift over a frame sorted
    by anything other than time silently uses match N+1. The frame is sorted
    by date and match order once, up front, and every quantity is a backward
    scan from that ordering.
    """
    m = matches.copy()
    m["_day"] = match_day(m)
    m = m.sort_values(["_day", "match_num"], kind="mergesort").reset_index(drop=True)
    t = m["_day"].to_numpy(dtype=float)
    minutes = pd.to_numeric(m["minutes"], errors="coerce").to_numpy(dtype=float)

    n = len(m)
    out = {f"{s}_{f}": np.zeros(n) for s in ("w", "l") for f in FEATURES}

    # last match date and a rolling 7-day minutes window, per player
    last_seen: dict[str, float] = {}
    history: dict[str, list[tuple[float, float]]] = {}

    wid = m["winner_id"].to_numpy()
    lid = m["loser_id"].to_numpy()
    for side, ids, entry_col in (("w", wid, "winner_entry"), ("l", lid, "loser_entry")):
        out[f"{side}_is_qualifier"] = (
            m[entry_col].astype(str).isin(QUALIFIER_ENTRIES).to_numpy().astype(float)
        )

    for i in range(n):
        for side, pid in (("w", wid[i]), ("l", lid[i])):
            prev = last_seen.get(pid)
            # No prior match means no evidence of fatigue, not infinite rest;
            # the cap makes those two the same value, which is what we want.
            rest = REST_CAP_DAYS if prev is None else min(t[i] - prev, REST_CAP_DAYS)
            out[f"{side}_rest_days"][i] = rest

            hist = history.get(pid)
            if hist:
                cutoff = t[i] - 7.0
                out[f"{side}_load7"][i] = sum(mins for tt, mins in hist if tt >= cutoff)

        # Update state only AFTER both sides of this match are recorded, so a
        # match never contributes to its own features.
        for side, pid in (("w", wid[i]), ("l", lid[i])):
            last_seen[pid] = t[i]
            mins = minutes[i]
            if np.isfinite(mins):
                h = history.setdefault(pid, [])
                h.append((t[i], float(mins)))
                if len(h) > 40:  # ponytail: bounded scan, 7 days never needs more
                    del h[:-40]

    res = pd.DataFrame(out)
    res["match_id"] = m["match_id"].to_numpy()
    return res


def design(ctx: pd.DataFrame, side: str) -> np.ndarray:
    """Standardised feature matrix for one side, shape (n, len(FEATURES)).

    ``rest_days`` enters as a *deficit* (how far below full rest the player
    is), so a positive coefficient means tiredness costs serve points. Without
    that flip the sign of the fitted coefficient reads backwards, which is the
    kind of thing that gets misreported six months later.
    """
    cols = []
    for f in FEATURES:
        v = ctx[f"{side}_{f}"].to_numpy(dtype=float)
        if f == "rest_days":
            v = (REST_CAP_DAYS - v) / SCALE[f]
        else:
            v = v / SCALE[f]
        cols.append(v)
    return np.column_stack(cols)


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def adjust(p: np.ndarray, x: np.ndarray, params: FatigueParams) -> np.ndarray:
    """Serve probability after the context shift."""
    if not params.enabled:
        return np.asarray(p, dtype=float)
    z = _logit(np.asarray(p, dtype=float)) - x @ params.vector()
    return 1.0 / (1.0 + np.exp(-z))


def fit(p: np.ndarray, x: np.ndarray, won: np.ndarray, played: np.ndarray
        ) -> FatigueParams:
    """Maximum-likelihood coefficients against observed serve points won.

    The baseline probability enters as a fixed offset rather than a fitted
    intercept: the earlier stages already decided what this player's serve
    rate is, and this stage is only allowed to say how much the match context
    moves it. Fitting an intercept here would let it quietly relevel every
    prediction and undo the calibration Stages 2-4 earned.
    """
    from scipy.optimize import minimize

    off = _logit(np.asarray(p, dtype=float))
    ok = (np.isfinite(off) & np.isfinite(won) & np.isfinite(played) & (played > 0)
          & np.isfinite(x).all(axis=1))
    off, xx, w, nn = off[ok], x[ok], won[ok], played[ok]

    def nll(beta: np.ndarray) -> float:
        z = off - xx @ beta
        # binomial log-likelihood, written via log1p/logaddexp for stability
        return float(np.sum(nn * np.logaddexp(0.0, z) - w * z))

    res = minimize(nll, np.zeros(xx.shape[1]), method="L-BFGS-B")
    b = res.x
    return FatigueParams(rest_days=float(b[0]), load7=float(b[1]),
                         is_qualifier=float(b[2]))


def demo() -> None:
    """Worked example (ground rule 10)."""
    rng = np.random.default_rng(20260808)
    n = 40_000
    x = np.column_stack([rng.random(n), rng.random(n) * 1.5, (rng.random(n) < 0.1)])
    true = np.array([0.30, 0.20, 0.10])
    base = np.clip(0.62 + rng.normal(0, 0.03, n), 0.4, 0.8)
    true_p = 1 / (1 + np.exp(-(_logit(base) - x @ true)))
    played = rng.integers(50, 120, n).astype(float)
    won = rng.binomial(played.astype(int), true_p).astype(float)

    got = fit(base, x, won, played)
    print(f"true  {true}")
    print(f"fitted [{got.rest_days:.3f} {got.load7:.3f} {got.is_qualifier:.3f}]")
    assert np.allclose(got.vector(), true, atol=0.03), got

    # Direction: a tired player must be predicted to win FEWER serve points.
    tired = np.array([[1.0, 1.0, 1.0]])
    fresh = np.zeros((1, 3))
    p0 = np.array([0.62])
    assert adjust(p0, tired, got)[0] < adjust(p0, fresh, got)[0]
    assert adjust(p0, fresh, got)[0] == 0.62 or abs(adjust(p0, fresh, got)[0] - 0.62) < 1e-9
    assert adjust(p0, tired, FatigueParams(enabled=False))[0] == 0.62

    # No leakage: a match must never contribute to its own context, so
    # truncating the future cannot change an earlier row.
    m = pd.DataFrame({
        "date": pd.to_datetime(
            ["2020-01-01", "2020-01-03", "2020-01-06", "2020-01-20"]).date,
        "match_num": [1, 2, 3, 4],
        "round": ["R32", "R32", "R32", "R32"],
        "tourney_id": ["t1", "t2", "t3", "t4"],
        "season_file": [2020, 2020, 2020, 2020],
        "tour": ["atp"] * 4,
        "winner_id": ["a", "a", "b", "a"],
        "loser_id": ["b", "c", "a", "d"],
        "winner_entry": ["Q", None, None, None],
        "loser_entry": [None, None, None, "LL"],
        "minutes": [120.0, 90.0, 150.0, 60.0],
        "match_id": ["m1", "m2", "m3", "m4"],
    })
    full = build_context(m)
    trunc = build_context(m.iloc[:3])
    cols = [c for c in full.columns if c != "match_id"]
    assert np.allclose(full.iloc[:3][cols].to_numpy(), trunc[cols].to_numpy()), \
        "context leaks the future"
    # a's first match: no history, so full rest and no load
    assert full.loc[0, "w_rest_days"] == REST_CAP_DAYS
    assert full.loc[0, "w_load7"] == 0.0
    assert full.loc[0, "w_is_qualifier"] == 1.0
    # a's second match is 2 days later, carrying the first match's 120 minutes
    assert full.loc[1, "w_rest_days"] == 2.0
    assert full.loc[1, "w_load7"] == 120.0
    # a's fourth match is 14 days after the third: rest 14, and the 7-day
    # window has rolled past everything
    assert full.loc[3, "w_rest_days"] == 14.0
    assert full.loc[3, "w_load7"] == 0.0

    # Rounds inside one event must land on different days, or rest_days reads
    # 0 for every within-tournament match — the defect that made the first
    # fit come back null.
    ev = pd.DataFrame({
        "date": pd.to_datetime(["2020-05-04"] * 4).date,
        "match_num": [1, 2, 3, 4],
        "round": ["R32", "R16", "QF", "SF"],
        "tourney_id": ["t9"] * 4,
        "season_file": [2020] * 4,
        "tour": ["atp"] * 4,
        "winner_id": ["a", "a", "a", "a"],
        "loser_id": ["b", "c", "d", "e"],
        "winner_entry": [None] * 4,
        "loser_entry": [None] * 4,
        "minutes": [100.0, 100.0, 100.0, 100.0],
        "match_id": ["e1", "e2", "e3", "e4"],
    })
    days = match_day(ev)
    assert list(days - days[0]) == [0, 1, 2, 3], days
    ec = build_context(ev)
    assert list(ec["w_rest_days"]) == [REST_CAP_DAYS, 1.0, 1.0, 1.0], \
        ec["w_rest_days"].tolist()
    # and the workload accumulates across those days
    assert list(ec["w_load7"]) == [0.0, 100.0, 200.0, 300.0], ec["w_load7"].tolist()
    print("self-check passed: coefficients recovered, rounds separated, "
          "no future leakage")


if __name__ == "__main__":
    demo()
