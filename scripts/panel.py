"""Shared evaluation panel: run the frozen pipeline over a set of matches.

Stages 6, 7 and 8 all need "the model as it stands, applied to every match in
a window". This builds exactly that from ``fitted_params.json``, so a stage
under test is the only thing that changes between runs.

Nothing here reads past the TEST boundary unless asked for explicitly; the
caller passes the splits it is entitled to see.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from model import cohort as CH
from model import combine as CB
from model import constants as C
from model import elo as E
from model import player_rates as PR
from model import venue as V
from model.engine import match_distribution
from model.rules import FormatSpec, rules_for


@lru_cache(maxsize=200_000)
def dist_for(pa: float, pb: float, key: tuple):
    """Cached exact match distribution for a rounded (pa, pb) and format."""
    spec = FormatSpec(best_of=key[0], games_to_win_set=key[1], tb_at=key[2],
                      tb_to=key[3], final_set=key[4], final_tb_at=key[5],
                      final_tb_to=key[6], provenance="documented", source="panel")
    return match_distribution(pa, pb, spec)


def crps_games(pmf: dict[int, float], actual: int) -> float:
    xs = np.array(sorted(pmf))
    cdf = np.cumsum([pmf[x] for x in xs])
    return float(np.sum((cdf - (xs >= actual).astype(float)) ** 2))


def load_fitted() -> dict:
    return json.loads(C.FITTED_PARAMS_PATH.read_text())


def build(splits: tuple[str, ...] = ("fit", "tune"),
          venue_params: V.VenueParams | None = None,
          cohort: bool = True) -> pd.DataFrame:
    """Blended (pa, pb) per match with outcomes attached.

    ``venue_params`` applies the Stage 6 multiplier to the level. Passing None
    leaves the level untouched, which is the pre-Stage-6 baseline.

    ``cohort=False`` switches off the Stage 5 cohort prior for this build only,
    the way ``venue_params`` already switches off Stage 6. It exists so an
    ablation can turn the component off by argument. The backtest used to do it
    by writing ``enabled: false`` into fitted_params.json, running, and writing
    it back — which made a shared file briefly wrong for every other reader and
    left it wrong outright if the run died in between.
    """
    fitted = load_fitted()
    from model.data_audit import HOLDOUT_PARQUET

    want_holdout = "holdout" in splits
    m = pd.read_parquet(C.PROCESSED_DIR
                        / (HOLDOUT_PARQUET if want_holdout else "matches.parquet"))
    # Ratings and rates must be replayed over ALL prior history, not just the
    # requested window, or a 2025 match would be priced with no 2024 form.
    keep = set(splits) | ({"fit", "tune", "burned_test", "test"} if want_holdout else set())
    m = m[m["in_scope"] & m["split"].isin(keep)]
    if not want_holdout:
        assert m["date"].max() < C.HOLDOUT_CUTOFF

    s2 = fitted["stage_2"]
    rp = PR.RateParams(half_life_days=s2["half_life_days"],
                       surface_pool=s2["surface_pool"], shrink_n0=s2["shrink_n0"])
    asof = PR.build_asof(m, rp)

    priors = None
    s5 = fitted.get("stage_5")
    if s5 and s5.get("enabled") and cohort:
        cp = CH.CohortParams(k=s5["k"], thin_n=s5["thin_n"],
                             well_sampled_n=s5["well_sampled_n"],
                             opponent_vs_cohort_n0=s5["opponent_vs_cohort_n0"],
                             snapshot_days=s5["snapshot_days"])
        snaps, history = CH.build_snapshots(m, cp)
        priors = {"w": np.full(len(asof), np.nan), "l": np.full(len(asof), np.nan)}
        cache: dict[tuple[str, int], float] = {}
        for i, (t, wid, lid, wn, ln) in enumerate(zip(
            asof["t"], asof["winner_id"], asof["loser_id"],
            asof["w_ss_pl"] + asof["w_sa_pl"], asof["l_ss_pl"] + asof["l_sa_pl"],
        )):
            snap = CH.snapshot_for(snaps, int(t))
            if snap is None:
                continue
            for side, pid, n in (("w", wid, wn), ("l", lid, ln)):
                if n >= cp.thin_n:
                    continue
                key = (pid, snap.date_ord)
                if key not in cache:
                    vec = history.get(key, np.full(len(CH.FEATURES), np.nan))
                    cache[key] = CH.cohort_prior(vec, snap, cp)[0]
                priors[side][i] = cache[key]

    rates = PR.serve_rates(asof, rp, prior_override=priors)
    rates["serve_pa"] = [PR.expected_spw(r, o, lg) for r, o, lg
                         in zip(rates["w_rate"], rates["l_ret"], rates["league"])]
    rates["serve_pb"] = [PR.expected_spw(r, o, lg) for r, o, lg
                         in zip(rates["l_rate"], rates["w_ret"], rates["league"])]

    s3 = fitted["stage_3"]
    ep = E.EloParams(k=s3["k"], k_chall_mult=s3["k_chall_mult"],
                     surface_weight=s3["surface_weight"],
                     inactivity_half_life=s3["inactivity_half_life"],
                     level_gap=s3["level_gap"], level_offset=s3["level_offset"])
    elo = E.run_elo(m, ep)[["match_id", "p_winner"]]

    p = rates.merge(elo, on="match_id", how="inner")
    meta = m.set_index("match_id")
    for col in ("tourney_code", "season_file", "split", "tour", "level_label",
                "winner_games", "loser_games", "tiebreaks", "n_sets", "best_of"):
        p[col] = meta.loc[p["match_id"], col].to_numpy()
    p["total_games"] = p["winner_games"] + p["loser_games"]

    specs = {(c, int(y)): rules_for(c, int(y))
             for c, y in p[["tourney_code", "season_file"]].drop_duplicates().values}
    p = p[[specs[(c, int(y))].supported for c, y
           in zip(p["tourney_code"], p["season_file"])]].copy()
    p["spec_key"] = [CB._spec_key(specs[(c, int(y))]) for c, y
                     in zip(p["tourney_code"], p["season_file"])]
    p["format_provenance"] = [specs[(c, int(y))].provenance for c, y
                              in zip(p["tourney_code"], p["season_file"])]

    s4 = fitted["stage_4"]
    cbp = CB.CombineParams(w=s4["w"], w_both_well=s4["w_both_well"],
                           w_one_thin=s4["w_one_thin"],
                           w_both_thin=s4["w_both_thin"], thin_n=s4["thin_n"])
    blended = CB.combine_many(p["serve_pa"].to_numpy(), p["serve_pb"].to_numpy(),
                              p["p_winner"].to_numpy(), p["spec_key"].tolist(),
                              cbp, p["w_n"].to_numpy(), p["l_n"].to_numpy())
    p["pa"], p["pb"] = blended["pa"], blended["pb"]
    p["bucket"] = blended["bucket"]

    # --- Stage 6: venue multiplier on the level ---------------------------
    p["venue_multiplier"] = 1.0
    p["venue_measured"] = False
    p["venue_n_matches"] = 0
    if venue_params is not None and venue_params.enabled:
        # The venue effect is measured as a residual against the model's own
        # expectation, so it is court speed rather than field strength.
        expected = pd.Series(
            ((p["serve_pa"] * p["w_svpt"] + p["serve_pb"] * p["l_svpt"])
             / (p["w_svpt"] + p["l_svpt"])).to_numpy(),
            index=p["match_id"].to_numpy(),
        )
        vi = V.build_asof_index(m, venue_params, expected=expected).set_index("match_id")
        common = p["match_id"].isin(vi.index)
        sub = vi.loc[p.loc[common, "match_id"]]
        p.loc[common, "venue_multiplier"] = sub["multiplier"].to_numpy()
        p.loc[common, "venue_measured"] = sub["measured"].to_numpy()
        p.loc[common, "venue_n_matches"] = sub["venue_n_matches"].to_numpy()
        pa, pb = [], []
        for a, b, mult in zip(p["pa"], p["pb"], p["venue_multiplier"]):
            na, nb = V.apply_multiplier(a, b, mult)
            pa.append(na)
            pb.append(nb)
        p["pa"], p["pb"] = pa, pb
    return p[p["split"].isin(splits)].copy()


def evaluate(panel: pd.DataFrame) -> dict:
    """Match-winner log-loss and total-games CRPS over a panel."""
    lls, crps = [], []
    for pa, pb, key, actual in zip(panel["pa"], panel["pb"], panel["spec_key"],
                                   panel["total_games"]):
        d = dist_for(round(float(pa), 3), round(float(pb), 3), key)
        lls.append(-np.log(max(d.p_a, 1e-9)))
        if np.isfinite(actual) and actual > 0:
            crps.append(crps_games(d.total_games_pmf(), int(actual)))
    return {"logloss": float(np.mean(lls)) if lls else float("nan"),
            "crps": float(np.mean(crps)) if crps else float("nan"),
            "n": len(lls)}


def demo() -> None:
    """Worked example (ground rule 10)."""
    p = build(splits=("tune",))
    print(f"{len(p):,} TUNE matches through the frozen pipeline")
    print(evaluate(p.head(500)))


if __name__ == "__main__":
    demo()
