"""Stage 6 — venue court-speed index.

A serve-dominance multiplier per (VENUE, SURFACE) relative to the same-surface
tour average, estimated from historical hold rates and applied to the LEVEL of
(pa + pb) before the engine — never to the split. A fast court makes both
players hold more; it does not make either of them better.

Keyed on (venue, surface), never venue alone. Stage 0's audit found 105
tournament codes whose surface changes across seasons, so pooling a venue's
clay years with its hard years under one multiplier would blend two different
courts. City and sponsor renames are handled by keying on the stable
``tourney_code`` rather than the tournament name.

As-of-date discipline (ground rule 5): a venue's multiplier for pricing a
match uses only that venue's history strictly before the match date. Matches
flagged ``score_string_suspect`` are excluded from the history, as are
retirements and walkovers, via the same ``serve_stats_valid`` filter Stage 2
uses.

Provenance (ground rule 7): a venue with no prior history gets exactly the
surface average and is flagged ``measured=False``, the same pattern rules.py
uses for format specs. An unmeasured venue must not look like a measured one
downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VenueParams:
    """Stage 6's hyperparameters. Fitted values live in fitted_params.json."""

    #: Empirical-Bayes shrinkage toward the surface mean, in serve points. A
    #: venue with 15 matches of history should barely move.
    shrink_n0: float = 20000.0
    #: Use the indoor flag as a separate key component where TML records it.
    use_indoor: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class VenueIndex:
    """One venue-surface multiplier and how much evidence is behind it."""

    key: tuple
    multiplier: float
    n_serve_points: float
    n_matches: int
    measured: bool

    def apply_to_level(self, level: float) -> float:
        """Scale the level, keeping both point probabilities in range."""
        return float(np.clip(level * self.multiplier, 0.60, 1.70))


def _frame(matches: pd.DataFrame, params: VenueParams) -> pd.DataFrame:
    f = matches[matches["serve_stats_valid"] & matches["in_scope"]].copy()
    f["t"] = pd.to_datetime(f["date"]).map(pd.Timestamp.toordinal)
    f["spw"] = ((f["w_1stWon"] + f["w_2ndWon"] + f["l_1stWon"] + f["l_2ndWon"])
                / (f["w_svpt"] + f["l_svpt"]))
    f["svpt"] = f["w_svpt"] + f["l_svpt"]
    f["surface_key"] = f["surface"].fillna("Unknown")
    if params.use_indoor:
        f["venue_key"] = list(zip(f["tourney_code"], f["surface_key"],
                                  f["indoor"].fillna("?")))
    else:
        f["venue_key"] = list(zip(f["tourney_code"], f["surface_key"]))
    return f.sort_values(["t", "match_id"]).reset_index(drop=True)


def build_asof_index(matches: pd.DataFrame, params: VenueParams) -> pd.DataFrame:
    """Per-match venue multiplier, computed from that venue's earlier matches.

    Returns one row per match with the multiplier that would have been used to
    price it, the evidence behind it, and whether the venue was measured at
    all.
    """
    f = _frame(matches, params)
    venue_won: dict[tuple, float] = {}
    venue_pts: dict[tuple, float] = {}
    venue_matches: dict[tuple, int] = {}
    surf_won: dict[str, float] = {}
    surf_pts: dict[str, float] = {}

    n = len(f)
    mult = np.ones(n)
    n_pts = np.zeros(n)
    n_matches = np.zeros(n, dtype=int)
    measured = np.zeros(n, dtype=bool)
    surf_rate = np.full(n, np.nan)

    for i, (vk, sk, spw, svpt) in enumerate(zip(
        f["venue_key"], f["surface_key"], f["spw"], f["svpt"]
    )):
        s_won, s_pts = surf_won.get(sk, 0.0), surf_pts.get(sk, 0.0)
        v_won, v_pts = venue_won.get(vk, 0.0), venue_pts.get(vk, 0.0)
        s_rate = s_won / s_pts if s_pts > 0 else np.nan
        surf_rate[i] = s_rate

        if v_pts > 0 and np.isfinite(s_rate) and s_rate > 0:
            # empirical Bayes toward the surface mean, in serve points
            v_rate = (v_won + params.shrink_n0 * s_rate) / (v_pts + params.shrink_n0)
            mult[i] = v_rate / s_rate
            measured[i] = True
        n_pts[i] = v_pts
        n_matches[i] = venue_matches.get(vk, 0)

        venue_won[vk] = v_won + spw * svpt
        venue_pts[vk] = v_pts + svpt
        venue_matches[vk] = venue_matches.get(vk, 0) + 1
        surf_won[sk] = s_won + spw * svpt
        surf_pts[sk] = s_pts + svpt

    out = f[["match_id", "date", "t", "surface_key", "tourney_code", "split",
             "spw", "svpt"]].copy()
    out["venue_key"] = f["venue_key"]
    out["multiplier"] = mult
    out["venue_n_points"] = n_pts
    out["venue_n_matches"] = n_matches
    out["measured"] = measured
    out["surface_rate"] = surf_rate
    return out


def index_for(asof: pd.DataFrame, match_id: str) -> VenueIndex:
    """The venue index used to price one match, as a structured record."""
    row = asof.loc[asof["match_id"] == match_id].iloc[0]
    return VenueIndex(key=row["venue_key"], multiplier=float(row["multiplier"]),
                      n_serve_points=float(row["venue_n_points"]),
                      n_matches=int(row["venue_n_matches"]),
                      measured=bool(row["measured"]))


def apply_multiplier(pa: float, pb: float, multiplier: float) -> tuple[float, float]:
    """Scale the LEVEL by the multiplier, leaving the split untouched.

    The split is who is better; the venue has no opinion on that.
    """
    level, split = pa + pb, pa - pb
    new_level = float(np.clip(level * multiplier, 0.60, 1.70))
    return (float(np.clip((new_level + split) / 2, 0.01, 0.99)),
            float(np.clip((new_level - split) / 2, 0.01, 0.99)))


def demo() -> None:
    """Worked example (ground rule 10)."""
    from model import constants as C

    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    m = m[m["split"].isin(("fit", "tune"))]
    asof = build_asof_index(m, VenueParams())
    print(f"{len(asof):,} matches; {asof['measured'].mean():.1%} at a venue with "
          "prior history")

    late = asof[asof["venue_n_matches"] > 100].drop_duplicates("venue_key", keep="last")
    late = late.sort_values("multiplier")
    print("\nslowest venues (lowest serve dominance vs their surface):")
    for _, r in late.head(5).iterrows():
        print(f"  {str(r['venue_key']):32s} x{r['multiplier']:.4f}  "
              f"({r['venue_n_matches']} matches)")
    print("fastest venues:")
    for _, r in late.tail(5).iloc[::-1].iterrows():
        print(f"  {str(r['venue_key']):32s} x{r['multiplier']:.4f}  "
              f"({r['venue_n_matches']} matches)")

    print("\napplying a multiplier moves the level, never the split:")
    for mult in (0.97, 1.0, 1.03):
        pa, pb = apply_multiplier(0.64, 0.60, mult)
        print(f"  x{mult}: pa {pa:.4f} pb {pb:.4f}  level {pa + pb:.4f} "
              f"split {pa - pb:+.4f}")


if __name__ == "__main__":
    demo()
