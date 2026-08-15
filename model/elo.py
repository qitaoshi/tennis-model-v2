"""Stage 3 — surface Elo.

Ratings are updated in date order, so the rating used to price a match is
always built from matches strictly before it (ground rule 5).

Each player carries an overall rating and one rating per surface; the rating
that plays a match is a blend of the two, with the blend weight fitted rather
than assumed. Absence regresses a rating toward its level reference at a
fitted rate — a player returning after a year is not the player who left.

Ground rule 6, as decided in Stage 0:

* retirements DO update ratings — TML records a winner and the completed part
  of the score, so the win/loss outcome is not in doubt;
* walkovers, defaults and unknown outcomes never update anything;
* matches flagged ``score_string_suspect`` still update ratings, because the
  outcome is rarely in doubt even when the detailed score is garbled. Nothing
  in this stage reads game-level detail.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Outcomes that move a rating (ground rule 6).
RATING_OUTCOMES = ("completed", "retired")

BASE_RATING = 1500.0


@dataclass(frozen=True)
class EloParams:
    """Stage 3's hyperparameters. Fitted values live in fitted_params.json."""

    k: float = 24.0
    #: Challenger results move ratings by ``k * k_chall_mult``.
    k_chall_mult: float = 1.0
    #: Weight on the surface rating when blending with the overall rating.
    surface_weight: float = 0.5
    #: Half-life in days of regression toward the level reference during an
    #: absence. None disables inactivity regression.
    inactivity_half_life: float | None = 1095.0
    #: How far below the tour reference a new Challenger player starts.
    level_gap: float = 100.0
    #: Cross-level correction added to a Challenger-built rating when it meets
    #: a tour-built one. Fitted only if the cross-level check demands it.
    level_offset: float = 0.0
    #: Rating points per natural-log unit of ATP rank used to seed a player the
    #: ladder has never seen. 0.0 reproduces the original behaviour exactly: a
    #: debutant starts at the flat level reference regardless of whether they
    #: are ranked 40 or 900.
    #:
    #: Why this exists: matches involving a debutant have Elo-layer ECE 0.0800
    #: against 0.0025 when both players are known (reports/
    #: match_winner_diagnosis.md). The ranking is in the data, is as-of by
    #: construction, and nothing was using it here.
    rank_seed_scale: float = 0.0
    #: Rank treated as "average" by the seed, i.e. the rank that gets exactly
    #: the flat reference. A scale choice, not a fitted value.
    rank_seed_pivot: float = 250.0


#: Every field of EloParams that Stage 3 selects and writes to
#: fitted_params.json. Anything added here is picked up by every consumer at
#: once — see params_from_fitted.
SELECTED_FIELDS = ("k", "k_chall_mult", "surface_weight",
                   "inactivity_half_life", "level_gap", "level_offset",
                   "rank_seed_scale")


def params_from_fitted(s3: dict) -> "EloParams":
    """Build EloParams from a fitted_params.json ``stage_3`` block.

    Six call sites used to spell this construction out by hand, each listing
    the fields it knew about. A field added to Stage 3 was therefore silently
    dropped by all of them — a selected parameter that changes nothing
    downstream, with nothing raising. ``rank_seed_scale`` was exactly that
    shape of change, so the construction lives in one place now.

    Missing keys fall back to the EloParams default, which is the pre-Stage-3
    behaviour, so a params file written before a field existed still loads.
    """
    return EloParams(**{f: s3[f] for f in SELECTED_FIELDS if f in s3})


def expected_score(ra: float, rb: float) -> float:
    """Standard Elo expectation."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


class _Ratings:
    """Overall and per-surface ratings with inactivity regression."""

    __slots__ = ("params", "overall", "surface", "last", "n", "chall_n")

    def __init__(self, params: EloParams) -> None:
        self.params = params
        self.overall: dict[str, float] = {}
        self.surface: dict[tuple[str, str], float] = {}
        self.last: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.chall_n: dict[str, int] = {}

    def reference(self, level_group: str) -> float:
        return BASE_RATING - (self.params.level_gap if level_group == "chall" else 0.0)

    def seed(self, pid: str, level_group: str, rank: float) -> None:
        """Give a never-seen player a starting rating from their ATP rank.

        No-op if the player is already rated, if the feature is off, or if the
        rank is missing — an unranked debutant genuinely carries no
        information, so it keeps the flat reference.

        Uses log rank because the gap in strength between rank 5 and 25 is
        much larger than between 505 and 525, and clamps the result so a
        freak rank cannot hand out a rating the ladder would take a season to
        work off.
        """
        if pid in self.overall or not self.params.rank_seed_scale:
            return
        if rank is None or not np.isfinite(rank) or rank <= 0:
            return
        ref = self.reference(level_group)
        offset = self.params.rank_seed_scale * np.log(
            self.params.rank_seed_pivot / float(rank))
        self.overall[pid] = ref + float(np.clip(offset, -300.0, 300.0))

    def _regress(self, value: float, ref: float, days: float) -> float:
        hl = self.params.inactivity_half_life
        if hl is None or days <= 0:
            return value
        return ref + (value - ref) * 0.5 ** (days / hl)

    def get(self, pid: str, surf: str, t: float, level_group: str) -> tuple[float, float]:
        """(overall, surface) rating as of ``t``, after inactivity regression."""
        ref = self.reference(level_group)
        days = t - self.last[pid] if pid in self.last else 0.0
        o = self._regress(self.overall.get(pid, ref), ref, days)
        s = self._regress(self.surface.get((pid, surf), o), ref, days)
        return o, s

    def blended(self, pid: str, surf: str, t: float, level_group: str) -> float:
        o, s = self.get(pid, surf, t, level_group)
        w = self.params.surface_weight
        return w * s + (1 - w) * o

    def chall_share(self, pid: str) -> float:
        n = self.n.get(pid, 0)
        return self.chall_n.get(pid, 0) / n if n else 0.0

    def update(self, pid: str, surf: str, t: float, level_group: str,
               score: float, expected: float) -> None:
        o, s = self.get(pid, surf, t, level_group)
        k = self.params.k * (self.params.k_chall_mult if level_group == "chall" else 1.0)
        delta = k * (score - expected)
        self.overall[pid] = o + delta
        self.surface[(pid, surf)] = s + delta
        self.last[pid] = t
        self.n[pid] = self.n.get(pid, 0) + 1
        if level_group == "chall":
            self.chall_n[pid] = self.chall_n.get(pid, 0) + 1


def ratings_at(matches: pd.DataFrame, params: EloParams) -> "_Ratings":
    """Final rating state after replaying ``matches``.

    price.py needs a rating on a date the player is not playing on. Same
    forward pass, same as-of guarantee: pass only earlier matches.
    """
    return run_elo(matches, params, return_state=True)


def run_elo(matches: pd.DataFrame, params: EloParams,
            return_state: bool = False):
    """Pre-match ratings and win probabilities for every match, in date order.

    Returns one row per match, oriented winner-first: ``p_winner`` is the
    model's pre-match probability for the player who actually won, so
    log-loss is ``-mean(log(p_winner))``.
    """
    f = matches[matches["in_scope"] & matches["outcome"].isin(RATING_OUTCOMES)].copy()
    f["t"] = pd.to_datetime(f["date"]).map(pd.Timestamp.toordinal)
    f["level_group"] = np.where(f["tour"] == "chall", "chall", "atp")
    f = f.sort_values(["t", "match_id"]).reset_index(drop=True)

    r = _Ratings(params)
    n = len(f)
    out = {k: np.zeros(n) for k in ("w_rating", "l_rating", "p_winner",
                                    "w_chall_share", "l_chall_share",
                                    "w_n", "l_n")}
    w_rank = pd.to_numeric(f["winner_rank"], errors="coerce").to_numpy(dtype=float)
    l_rank = pd.to_numeric(f["loser_rank"], errors="coerce").to_numpy(dtype=float)
    for i, (t, wid, lid, surf, lg) in enumerate(zip(
        f["t"], f["winner_id"], f["loser_id"], f["surface"], f["level_group"]
    )):
        surf = surf if isinstance(surf, str) else "Unknown"
        # Seed before reading, so a debutant's first match is priced with the
        # seed rather than being seeded by its own result afterwards.
        r.seed(wid, lg, w_rank[i])
        r.seed(lid, lg, l_rank[i])
        rw = r.blended(wid, surf, t, lg)
        rl = r.blended(lid, surf, t, lg)
        # cross-level correction, applied only where the two ratings come from
        # different tiers (see the cross-level consistency check)
        if params.level_offset:
            rw += params.level_offset * _chall_built(r, wid)
            rl += params.level_offset * _chall_built(r, lid)
        p = expected_score(rw, rl)

        out["w_rating"][i], out["l_rating"][i] = rw, rl
        out["p_winner"][i] = p
        out["w_chall_share"][i] = r.chall_share(wid)
        out["l_chall_share"][i] = r.chall_share(lid)
        out["w_n"][i], out["l_n"][i] = r.n.get(wid, 0), r.n.get(lid, 0)

        r.update(wid, surf, t, lg, 1.0, p)
        r.update(lid, surf, t, lg, 0.0, 1.0 - p)

    if return_state:
        return r
    res = f[["match_id", "date", "t", "surface", "level_group", "tour",
             "winner_id", "loser_id", "winner_rank", "loser_rank",
             "tourney_code", "season_file", "split"]].copy()
    for k, v in out.items():
        res[k] = v
    return res


def _chall_built(r: _Ratings, pid: str) -> float:
    """1.0 if this rating is essentially Challenger-built, else 0.0."""
    return 1.0 if r.chall_share(pid) >= 0.8 else 0.0


# ---------------------------------------------------------------------------
# Baseline and metrics
# ---------------------------------------------------------------------------


def ranking_baseline(res: pd.DataFrame, fit_mask: np.ndarray) -> np.ndarray:
    """Rankings-based win probability, logistic in log rank difference.

    The coefficient is fitted on FIT only, so the baseline is as honest a
    comparison as the model it is judged against.
    """
    lw = np.log(np.clip(res["winner_rank"].to_numpy(dtype=float), 1, None))
    ll = np.log(np.clip(res["loser_rank"].to_numpy(dtype=float), 1, None))
    diff = ll - lw  # positive when the winner was the higher-ranked player
    ok = np.isfinite(diff) & fit_mask
    # single-parameter logistic fit by maximum likelihood on FIT
    betas = np.linspace(0.05, 2.0, 196)
    lls = [np.mean(np.log(np.clip(1 / (1 + np.exp(-b * diff[ok])), 1e-9, 1)))
           for b in betas]
    beta = float(betas[int(np.argmax(lls))])
    p = 1 / (1 + np.exp(-beta * np.nan_to_num(diff)))
    return np.clip(p, 1e-6, 1 - 1e-6)


def log_loss(p_winner: np.ndarray) -> float:
    p = np.clip(p_winner[np.isfinite(p_winner)], 1e-9, 1 - 1e-9)
    return float(-np.mean(np.log(p)))


def brier(p_winner: np.ndarray) -> float:
    """Brier score with every row oriented winner-first (outcome always 1)."""
    p = p_winner[np.isfinite(p_winner)]
    return float(np.mean((1.0 - p) ** 2))


def decile_calibration(p_winner: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed win rate by predicted-probability decile.

    Rows are winner-first, so each match is symmetrized into two observations
    — (p, 1) and (1-p, 0) — before binning; otherwise every outcome is a win
    by construction and calibration is meaningless.
    """
    p = np.concatenate([p_winner, 1.0 - p_winner])
    y = np.concatenate([np.ones_like(p_winner), np.zeros_like(p_winner)])
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "n": int(m.sum()), "predicted": float(p[m].mean()),
                     "observed": float(y[m].mean())})
    df = pd.DataFrame(rows)
    df["gap"] = df["observed"] - df["predicted"]
    return df


def cross_level_check(res: pd.DataFrame) -> dict:
    """Calibration on Challenger-built vs tour-rated matchups.

    A known failure mode of single-scale Elo, distinct from K tuning: ratings
    built almost entirely inside the Challenger pool can sit on a different
    scale from tour-built ones, so matches between the two are systematically
    mispriced even when overall calibration looks fine.
    """
    w_chall = res["w_chall_share"] >= 0.8
    l_chall = res["l_chall_share"] >= 0.8
    cross = (w_chall & ~l_chall) | (~w_chall & l_chall)
    seen = (res["w_n"] >= 10) & (res["l_n"] >= 10)
    sub = res[cross & seen]
    if sub.empty:
        return {"n": 0}
    # orient by the Challenger-built player, so the gap has a readable sign
    chall_is_winner = (sub["w_chall_share"] >= 0.8).to_numpy()
    p_chall = np.where(chall_is_winner, sub["p_winner"], 1 - sub["p_winner"])
    observed = chall_is_winner.astype(float)
    return {
        "n": int(len(sub)),
        "predicted_chall_winrate": float(np.mean(p_chall)),
        "observed_chall_winrate": float(np.mean(observed)),
        "gap": float(np.mean(observed) - np.mean(p_chall)),
        "logloss": log_loss(sub["p_winner"].to_numpy()),
    }


def demo() -> None:
    """Worked example (ground rule 10)."""
    from model import constants as C

    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    m = m[m["split"].isin(("fit", "tune"))]
    res = run_elo(m, EloParams())
    tune = res[res["split"] == "tune"]
    print(f"{len(res):,} rated matches; TUNE {len(tune):,}")
    print(f"TUNE log-loss {log_loss(tune['p_winner'].to_numpy()):.5f}, "
          f"Brier {brier(tune['p_winner'].to_numpy()):.5f}")
    print("\ndecile calibration (TUNE):")
    print(decile_calibration(tune["p_winner"].to_numpy()).to_string(index=False))
    print(f"\ncross-level check: {cross_level_check(tune)}")


if __name__ == "__main__":
    demo()
