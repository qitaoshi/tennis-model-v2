"""Stage 5 — cohort prior for thin-data players.

A player with 200 serve points of history is currently shrunk toward the flat
tour-level average. That average is a poor description of, say, a 2-metre
server with a 15% ace rate. This stage replaces the flat prior — for thin
players only — with the average profile of the k most stylistically similar
well-sampled players.

AS-OF-DATE DISCIPLINE, INCLUDING THE STANDARDIZATION STATISTICS (ground rule
5). Two separate paths could leak the future here and both are closed:

* every per-player style feature is accumulated from matches strictly before
  the snapshot date;
* the means and standard deviations used to z-score those features are
  computed from the same as-of population, never once over the whole dataset.
  A thin player's 2015 ace-rate z-score must not know what the tour's ace-rate
  distribution looked like in 2023.

Snapshots are taken at fixed period boundaries and a match uses the most
recent snapshot strictly at or before its date. That is a deliberate
approximation of a continuous as-of computation — it can only ever use older
data, never newer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Style features, in vector order. Missing height and hand are imputed to the
#: as-of population mean rather than dropped.
FEATURES = ("ace_rate", "df_rate", "serve_pw", "return_pw", "clay_lean",
            "hand_left", "height")


@dataclass(frozen=True)
class CohortParams:
    """Stage 5's hyperparameters. Fitted values live in fitted_params.json."""

    k: int = 20
    #: Effective serve points below which a player gets the cohort prior
    #: instead of the flat tour-level prior.
    thin_n: float = 1000.0
    #: A player needs this many serve points to be eligible as a neighbour.
    well_sampled_n: float = 3000.0
    #: Shrinkage, in serve points, of the opponent's record against the cohort
    #: toward their overall record. 0 disables the opponent adjustment.
    opponent_vs_cohort_n0: float = 200.0
    #: Days between style snapshots.
    snapshot_days: int = 91
    enabled: bool = True


@dataclass(frozen=True)
class Snapshot:
    """Everything the kNN needs at one point in time."""

    date_ord: int
    player_ids: np.ndarray
    z: np.ndarray            # standardized features, well-sampled players only
    serve_pw: np.ndarray     # raw serve point-win rate per neighbour
    return_pw: np.ndarray
    mean: np.ndarray         # standardization statistics, as of this date
    std: np.ndarray
    all_ids: dict[str, np.ndarray]  # every player's raw feature vector


def _accumulate(matches: pd.DataFrame) -> pd.DataFrame:
    """Per (match, player) style contributions, in date order."""
    f = matches[matches["serve_stats_valid"]].copy()
    f["t"] = pd.to_datetime(f["date"]).map(pd.Timestamp.toordinal)
    return f.sort_values(["t", "match_id"]).reset_index(drop=True)


class _StyleState:
    """Running per-player style totals. Read-only from the caller's side."""

    def __init__(self) -> None:
        self.tot: dict[str, np.ndarray] = {}
        self.bio: dict[str, tuple[float, float]] = {}

    #  totals: [svpt, spw, ace, df, ret_pts, ret_won, clay_svpt, clay_spw]
    def add(self, pid: str, svpt: float, spw: float, ace: float, df: float,
            ret_pts: float, ret_won: float, is_clay: bool) -> None:
        cur = self.tot.setdefault(pid, np.zeros(8))
        cur += np.array([svpt, spw, ace, df, ret_pts, ret_won,
                         svpt if is_clay else 0.0, spw if is_clay else 0.0])

    def set_bio(self, pid: str, hand: object, height: object) -> None:
        if pid in self.bio:
            return
        h = 1.0 if isinstance(hand, str) and hand.upper().startswith("L") else 0.0
        ht = float(height) if height == height and height else np.nan
        self.bio[pid] = (h, ht)

    def features(self, pid: str) -> tuple[np.ndarray, float]:
        """(raw feature vector, serve points of evidence)."""
        t = self.tot.get(pid)
        if t is None or t[0] <= 0:
            return np.full(len(FEATURES), np.nan), 0.0
        svpt, spw, ace, df, rp, rw, clay_svpt, clay_spw = t
        serve_pw = spw / svpt
        clay = clay_spw / clay_svpt if clay_svpt > 0 else np.nan
        non_clay = ((spw - clay_spw) / (svpt - clay_svpt)
                    if svpt - clay_svpt > 0 else np.nan)
        hand, height = self.bio.get(pid, (np.nan, np.nan))
        return np.array([
            ace / svpt,
            df / svpt,
            serve_pw,
            rw / rp if rp > 0 else np.nan,
            (clay - non_clay) if np.isfinite(clay) and np.isfinite(non_clay) else np.nan,
            hand,
            height,
        ]), float(svpt)


def build_snapshots(matches: pd.DataFrame, params: CohortParams
                    ) -> tuple[list[Snapshot], dict]:
    """Style snapshots at fixed period boundaries, each strictly as-of.

    Also returns per (player, snapshot) raw feature vectors so a thin player
    can be placed in the same space as the neighbours.
    """
    f = _accumulate(matches)
    state = _StyleState()
    snaps: list[Snapshot] = []
    history: dict[tuple[str, int], np.ndarray] = {}

    start = int(f["t"].min())
    boundaries = list(range(start + params.snapshot_days,
                            int(f["t"].max()) + params.snapshot_days,
                            params.snapshot_days))
    bi = 0
    for row in f.itertuples(index=False):
        while bi < len(boundaries) and row.t >= boundaries[bi]:
            snaps.append(_snapshot(state, boundaries[bi], params, history))
            bi += 1
        state.set_bio(row.winner_id, row.winner_hand, row.winner_ht)
        state.set_bio(row.loser_id, row.loser_hand, row.loser_ht)
        state.add(row.winner_id, row.w_svpt, row.w_1stWon + row.w_2ndWon,
                  row.w_ace, row.w_df, row.l_svpt,
                  row.l_svpt - (row.l_1stWon + row.l_2ndWon),
                  row.surface == "Clay")
        state.add(row.loser_id, row.l_svpt, row.l_1stWon + row.l_2ndWon,
                  row.l_ace, row.l_df, row.w_svpt,
                  row.w_svpt - (row.w_1stWon + row.w_2ndWon),
                  row.surface == "Clay")
    while bi < len(boundaries):
        snaps.append(_snapshot(state, boundaries[bi], params, history))
        bi += 1
    return snaps, history


def _snapshot(state: _StyleState, date_ord: int, params: CohortParams,
              history: dict) -> Snapshot:
    ids, raw, ns = [], [], []
    for pid in state.tot:
        vec, n = state.features(pid)
        ids.append(pid)
        raw.append(vec)
        ns.append(n)
        history[(pid, date_ord)] = vec
    ids = np.array(ids)
    raw = np.array(raw) if len(raw) else np.zeros((0, len(FEATURES)))
    ns = np.array(ns)

    well = ns >= params.well_sampled_n
    pool = raw[well] if well.any() else raw
    # standardization statistics: as-of, from this population only
    mean = np.nanmean(pool, axis=0) if len(pool) else np.zeros(len(FEATURES))
    std = np.nanstd(pool, axis=0) if len(pool) else np.ones(len(FEATURES))
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)

    z = (np.where(np.isfinite(pool), pool, mean) - mean) / std
    serve_pw = pool[:, FEATURES.index("serve_pw")] if len(pool) else np.zeros(0)
    return_pw = pool[:, FEATURES.index("return_pw")] if len(pool) else np.zeros(0)
    return Snapshot(date_ord, ids[well] if well.any() else ids, z,
                    np.nan_to_num(serve_pw, nan=float(mean[2])),
                    np.nan_to_num(return_pw, nan=float(mean[3])),
                    mean, std, {})


def snapshot_for(snaps: list[Snapshot], date_ord: int) -> Snapshot | None:
    """The most recent snapshot strictly at or before ``date_ord``."""
    lo, hi = 0, len(snaps) - 1
    found = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if snaps[mid].date_ord <= date_ord:
            found = snaps[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return found


def cohort_prior(vec: np.ndarray, snap: Snapshot, params: CohortParams
                 ) -> tuple[float, float, np.ndarray]:
    """(serve prior, return prior, neighbour indices) for one style vector.

    Neighbours are the k nearest well-sampled players in standardized style
    space, using the snapshot's own means and standard deviations.
    """
    if snap is None or len(snap.z) == 0:
        return float("nan"), float("nan"), np.zeros(0, dtype=int)
    zv = (np.where(np.isfinite(vec), vec, snap.mean) - snap.mean) / snap.std
    d = np.linalg.norm(snap.z - zv, axis=1)
    k = min(params.k, len(d))
    idx = np.argpartition(d, k - 1)[:k]
    return float(np.mean(snap.serve_pw[idx])), float(np.mean(snap.return_pw[idx])), idx


def return_terciles(snap: Snapshot) -> tuple[float, float]:
    """Return-strength tercile boundaries among well-sampled players, as-of."""
    if snap is None or len(snap.return_pw) < 3:
        return (0.36, 0.39)
    return tuple(np.quantile(snap.return_pw, [1 / 3, 2 / 3]))


def tercile_of(return_pw: float, bounds: tuple[float, float]) -> int:
    return int(return_pw >= bounds[0]) + int(return_pw >= bounds[1])


def vs_cohort_totals(matches: pd.DataFrame, snaps: list[Snapshot]
                     ) -> dict[str, np.ndarray]:
    """As-of serve record of each player against each return-strength tercile.

    This is what lets a well-sampled opponent's history against players *like*
    the thin one adjust the matchup: a server who historically holds up well
    against strong returners should not be priced the same as one who does not
    when the thin player's cohort is a strong-returning one.

    Terciles are cut on the as-of well-sampled return-rate distribution, so no
    boundary is informed by later data.
    """
    f = _accumulate(matches)
    tot: dict[tuple[str, int], np.ndarray] = {}
    ret: dict[str, np.ndarray] = {}
    n = len(f)
    out = {k: np.zeros(n) for k in ("w_vs_svpt", "w_vs_spw", "l_vs_svpt",
                                    "l_vs_spw", "w_opp_tercile", "l_opp_tercile")}

    def ret_rate(pid: str) -> float:
        r = ret.get(pid)
        return float(r[1] / r[0]) if r is not None and r[0] > 0 else np.nan

    for i, row in enumerate(f.itertuples(index=False)):
        snap = snapshot_for(snaps, int(row.t))
        bounds = return_terciles(snap)
        mean_ret = float(snap.mean[FEATURES.index("return_pw")]) if snap else 0.38

        w_ret, l_ret = ret_rate(row.winner_id), ret_rate(row.loser_id)
        w_t = tercile_of(l_ret if np.isfinite(l_ret) else mean_ret, bounds)
        l_t = tercile_of(w_ret if np.isfinite(w_ret) else mean_ret, bounds)
        out["w_opp_tercile"][i], out["l_opp_tercile"][i] = w_t, l_t
        for side, pid, t in (("w", row.winner_id, w_t), ("l", row.loser_id, l_t)):
            cur = tot.get((pid, t))
            if cur is not None:
                out[f"{side}_vs_svpt"][i], out[f"{side}_vs_spw"][i] = cur

        for pid, t, svpt, won in ((row.winner_id, w_t, row.w_svpt,
                                   row.w_1stWon + row.w_2ndWon),
                                  (row.loser_id, l_t, row.l_svpt,
                                   row.l_1stWon + row.l_2ndWon)):
            cur = tot.setdefault((pid, t), np.zeros(2))
            cur += np.array([svpt, won])
        for pid, faced, lost in ((row.winner_id, row.l_svpt,
                                  row.l_svpt - (row.l_1stWon + row.l_2ndWon)),
                                 (row.loser_id, row.w_svpt,
                                  row.w_svpt - (row.w_1stWon + row.w_2ndWon))):
            cur = ret.setdefault(pid, np.zeros(2))
            cur += np.array([faced, lost])

    out["match_id"] = f["match_id"].to_numpy()
    return out


def opponent_vs_cohort_rate(vs_svpt: float, vs_spw: float, overall: float,
                            n0: float) -> float:
    """Blend a player's record against the cohort's tier with their overall rate.

    Empirical Bayes in serve points, the same form Stage 2 uses, so a handful
    of points against the tier barely moves the estimate.
    """
    if n0 <= 0 or vs_svpt <= 0:
        return overall
    return float((vs_spw + n0 * overall) / (vs_svpt + n0))


def demo() -> None:
    """Worked example (ground rule 10)."""
    from model import constants as C

    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    m = m[m["in_scope"] & m["split"].isin(("fit", "tune"))]
    params = CohortParams()
    snaps, history = build_snapshots(m, params)
    print(f"{len(snaps)} snapshots, {snaps[0].date_ord} .. {snaps[-1].date_ord}")

    late = snaps[-1]
    print(f"latest snapshot: {len(late.player_ids)} well-sampled players")
    print("standardization means (as of that date):")
    for name, mu, sd in zip(FEATURES, late.mean, late.std):
        print(f"  {name:12s} mean {mu:8.4f}  std {sd:.4f}")

    # a big server profile: high ace rate, high serve pw, average elsewhere
    vec = late.mean.copy()
    vec[FEATURES.index("ace_rate")] = late.mean[0] + 2 * late.std[0]
    vec[FEATURES.index("serve_pw")] = late.mean[2] + 1.5 * late.std[2]
    s, r, idx = cohort_prior(vec, late, params)
    print(f"\nbig-server cohort ({len(idx)} neighbours): serve prior {s:.4f}, "
          f"return prior {r:.4f}")
    print(f"flat population prior: serve {late.mean[2]:.4f}, "
          f"return {late.mean[3]:.4f}")


if __name__ == "__main__":
    demo()
