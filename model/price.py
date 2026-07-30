"""The product: two players in, fair probabilities and prices for every market.

``price_match`` runs the whole frozen pipeline for one matchup as of one date:

    Stage 2 serve rates  ->  Stage 5 cohort prior where thin
    Stage 3 Elo          ->  Stage 4 blend                    ->  (pa, pb)
    Stage 6 venue index  ->  scales the level
    Stage 7 corrections  ->  distribution shape
    Stage 1 engine       ->  every market
    Stage 8 isotonic maps ->  final probabilities

The format comes from rules.py via (tournament, year) — never a caller flag —
and its provenance rides out in the metadata alongside every other thing a
downstream stake decision would want: each side's effective sample size, the
elo-vs-serve disagreement, whether the venue was measured, which corrections
and calibration maps were applied, thin-data flags, and — for advantage-set
formats — which requested lines fall beyond the truncation point and are
therefore not priced.

No bookmaker odds appear anywhere in this path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from model import cohort as CH
from model import combine as CB
from model import constants as C
from model import corrections as CR
from model import elo as E
from model import player_rates as PR
from model import recalibrate as RC
from model import venue as V
from model.engine import MatchDistribution, fair_price
from model.rules import FormatSpec, rules_for


@dataclass
class Selection:
    """One priced selection."""

    market: str
    selection: str
    probability: float
    fair_decimal: float
    push_probability: float = 0.0
    calibrated: bool = False
    note: str = ""


@dataclass
class PricedMatch:
    """Everything the pricer knows about one matchup."""

    player_a: str
    player_b: str
    as_of_date: date
    selections: list[Selection]
    metadata: dict[str, Any] = field(default_factory=dict)

    def market(self, name: str) -> list[Selection]:
        return [s for s in self.selections if s.market == name]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(s) for s in self.selections])


class Pricer:
    """Holds the replayed history so repeated pricing is cheap.

    Everything is rebuilt from matches strictly before ``as_of_date``, so the
    as-of guarantee is structural rather than a convention: a pricer built for
    2019-06-01 has never seen a 2019-06-02 match.
    """

    def __init__(self, as_of_date: date, fitted: dict | None = None,
                 allow_holdout: bool = False) -> None:
        import json

        self.as_of_date = as_of_date
        self.fitted = fitted or json.loads(C.FITTED_PARAMS_PATH.read_text())
        if not allow_holdout and as_of_date >= C.HOLDOUT_CUTOFF:
            raise ValueError(
                f"pricing on/after the holdout cutoff ({C.HOLDOUT_CUTOFF}) "
                "requires allow_holdout=True — that data is for the final "
                "backtest only"
            )

        m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
        self.history = m[m["in_scope"] & (m["date"] < as_of_date)]

        s2 = self.fitted["stage_2"]
        self.rate_params = PR.RateParams(half_life_days=s2["half_life_days"],
                                         surface_pool=s2["surface_pool"],
                                         shrink_n0=s2["shrink_n0"])
        self.rate_state = PR.state_at(self.history, self.rate_params)

        s3 = self.fitted["stage_3"]
        self.elo_params = E.EloParams(
            k=s3["k"], k_chall_mult=s3["k_chall_mult"],
            surface_weight=s3["surface_weight"],
            inactivity_half_life=s3["inactivity_half_life"],
            level_gap=s3["level_gap"], level_offset=s3["level_offset"])
        self.elo_state = E.ratings_at(self.history, self.elo_params)

        s4 = self.fitted["stage_4"]
        self.combine_params = CB.CombineParams(
            w=s4["w"], w_both_well=s4["w_both_well"],
            w_one_thin=s4["w_one_thin"], w_both_thin=s4["w_both_thin"],
            thin_n=s4["thin_n"])

        s5 = self.fitted.get("stage_5", {})
        self.cohort_params = CH.CohortParams(
            k=s5.get("k", 5), thin_n=s5.get("thin_n", 1000.0),
            well_sampled_n=s5.get("well_sampled_n", 3000.0),
            opponent_vs_cohort_n0=s5.get("opponent_vs_cohort_n0", 0.0),
            snapshot_days=s5.get("snapshot_days", 91),
            enabled=bool(s5.get("enabled", False)))
        if self.cohort_params.enabled:
            self.snaps, self.style_history = CH.build_snapshots(
                self.history, self.cohort_params)
        else:
            self.snaps, self.style_history = [], {}

        s6 = self.fitted.get("stage_6", {})
        self.venue_params = V.VenueParams(
            shrink_n0=s6.get("shrink_n0", 20000.0),
            use_indoor=s6.get("use_indoor", False),
            enabled=bool(s6.get("enabled", False)))
        self.venue_index = (V.build_asof_index(self.history, self.venue_params)
                            if self.venue_params.enabled else None)

        s7 = self.fitted.get("stage_7", {})
        self.correction_params = CR.CorrectionParams(
            tiebreak_inflation=s7.get("tiebreak_inflation", 0.0),
            level_sigma=s7.get("level_sigma", 0.0),
            split_sigma=s7.get("split_sigma", 0.0),
            recenter=s7.get("recenter", True),
            scheme=s7.get("provenance_scheme", "pooled"),
            inferred_weight=s7.get("inferred_weight", 1.0))
        self.corrections_applied = bool(s7)

        try:
            self.maps = RC.CalibrationMaps.load()
        except (FileNotFoundError, OSError):
            self.maps = None

    # -- components ------------------------------------------------------

    def _level_group(self, player_id: str) -> str:
        played = self.history[(self.history["winner_id"] == player_id)
                              | (self.history["loser_id"] == player_id)]
        if played.empty:
            return "atp"
        return "chall" if (played["tour"] == "chall").mean() > 0.5 else "atp"

    def _cohort_prior(self, player_id: str, t: float, n: float) -> float | None:
        if not self.cohort_params.enabled or n >= self.cohort_params.thin_n:
            return None
        snap = CH.snapshot_for(self.snaps, int(t))
        if snap is None:
            return None
        vec = self.style_history.get((player_id, snap.date_ord),
                                     np.full(len(CH.FEATURES), np.nan))
        return CH.cohort_prior(vec, snap, self.cohort_params)[0]

    def _venue_multiplier(self, tourney_code: str, surface: str) -> tuple[float, bool, int]:
        if self.venue_index is None:
            return 1.0, False, 0
        key = (tourney_code, surface)
        rows = self.venue_index[self.venue_index["venue_key"] == key]
        if rows.empty:
            return 1.0, False, 0
        last = rows.iloc[-1]
        return float(last["multiplier"]), True, int(last["venue_n_matches"]) + 1

    # -- pricing ---------------------------------------------------------

    def price(self, player_a: str, player_b: str, tournament_id: str, year: int,
              surface: str, venue: str | None = None) -> PricedMatch:
        spec = rules_for(tournament_id, year)
        t = float(pd.Timestamp(self.as_of_date).toordinal())
        lg_a, lg_b = self._level_group(player_a), self._level_group(player_b)

        rate_a, ret_a = PR.rate_at(self.rate_state, player_a, surface, t, lg_a,
                                   self.rate_params)
        rate_b, ret_b = PR.rate_at(self.rate_state, player_b, surface, t, lg_b,
                                   self.rate_params)
        prior_a = self._cohort_prior(player_a, t, rate_a.n)
        prior_b = self._cohort_prior(player_b, t, rate_b.n)
        if prior_a is not None:
            rate_a, ret_a = PR.rate_at(self.rate_state, player_a, surface, t, lg_a,
                                       self.rate_params, prior_override=prior_a)
        if prior_b is not None:
            rate_b, ret_b = PR.rate_at(self.rate_state, player_b, surface, t, lg_b,
                                       self.rate_params, prior_override=prior_b)

        lg_won, lg_pl, _ = self.rate_state["league"].get(lg_a, t)
        league = lg_won / lg_pl if lg_pl > 0 else 0.62
        serve_pa = PR.expected_spw(rate_a.value, ret_b, league)
        serve_pb = PR.expected_spw(rate_b.value, ret_a, league)

        elo_a = self.elo_state.blended(player_a, surface, t, lg_a)
        elo_b = self.elo_state.blended(player_b, surface, t, lg_b)
        elo_p = E.expected_score(elo_a, elo_b)

        blended = CB.combine(serve_pa, serve_pb, elo_p, spec,
                             self.combine_params, rate_a.n, rate_b.n)
        pa, pb = blended["pa"], blended["pb"]

        mult, measured, venue_n = self._venue_multiplier(
            tournament_id, surface if isinstance(surface, str) else "Unknown")
        if self.venue_params.enabled:
            pa, pb = V.apply_multiplier(pa, pb, mult)

        dist = CR.corrected_distribution(pa, pb, spec, self.correction_params)

        meta = {
            "pa": pa, "pb": pb,
            "level": pa + pb, "split": pa - pb,
            "serve_pa": serve_pa, "serve_pb": serve_pb,
            "player_a_effective_n": rate_a.n, "player_b_effective_n": rate_b.n,
            "player_a_matches": rate_a.n_matches, "player_b_matches": rate_b.n_matches,
            "elo_a": elo_a, "elo_b": elo_b, "elo_win_prob": elo_p,
            "serve_win_prob": blended["serve_p"],
            "elo_serve_disagreement_pp": blended["disagreement_pp"],
            "blend_weight_used": blended["w_used"],
            "sample_bucket": blended["bucket"],
            "thin_data_a": prior_a is not None, "thin_data_b": prior_b is not None,
            "cohort_prior_a": prior_a, "cohort_prior_b": prior_b,
            "cohort_enabled": self.cohort_params.enabled,
            "venue_multiplier": mult, "venue_measured": measured,
            "venue_history_matches": venue_n,
            "venue_enabled": self.venue_params.enabled,
            "format": {
                "best_of": spec.best_of, "final_set": spec.final_set,
                "final_tb_at": spec.final_tb_at, "final_tb_to": spec.final_tb_to,
                "games_to_win_set": spec.games_to_win_set,
                "provenance": spec.provenance, "source": spec.source,
                "supported": spec.supported,
            },
            "corrections": {
                "applied": self.corrections_applied,
                "tiebreak_inflation": self.correction_params.tiebreak_inflation,
                "split_sigma": self.correction_params.split_sigma,
                "level_sigma": self.correction_params.level_sigma,
                "provenance_scheme": self.correction_params.scheme,
            },
            "calibration_maps": sorted(self.maps.maps) if self.maps else [],
            "truncation_point_games": dist.max_total_games,
            "truncated_mass": dist.truncated_mass,
            "lines_beyond_truncation_not_priced": spec.final_set == "advantage",
            "as_of_date": str(self.as_of_date),
        }
        if not spec.supported:
            meta["unsupported_format"] = (
                "this format is not priced by the engine; no selections returned")
            return PricedMatch(player_a, player_b, self.as_of_date, [], meta)

        sels = self._selections(dist, spec)
        return PricedMatch(player_a, player_b, self.as_of_date, sels, meta)

    def _cal(self, family: str, p: float) -> tuple[float, bool]:
        if self.maps is None or family not in self.maps.maps:
            return p, False
        return float(self.maps.apply(family, p)), True

    def _selections(self, dist: MatchDistribution, spec: FormatSpec) -> list[Selection]:
        out: list[Selection] = []

        # --- match winner -------------------------------------------------
        for who, p in (("A", dist.p_a), ("B", 1 - dist.p_a)):
            q, cal = self._cal("match_winner", p)
            out.append(Selection("match_winner", who, q, fair_price(q), 0.0, cal))

        # --- set betting (exact sets) -------------------------------------
        for (sa, sb), p in sorted(dist.sets.items()):
            q, cal = self._cal("set_score", p)
            out.append(Selection("set_betting", f"{sa}-{sb}", q, fair_price(q),
                                 0.0, cal))

        # --- totals ladder -------------------------------------------------
        pmf = dist.total_games_pmf()
        median = _median_of(pmf)
        lo, hi = min(pmf), max(pmf)
        for twice in range(2 * lo - 1, 2 * hi + 2):
            line = twice / 2
            over = sum(p for g, p in pmf.items() if g > line)
            under = sum(p for g, p in pmf.items() if g < line)
            push = pmf.get(int(line), 0.0) if float(line).is_integer() else 0.0
            fam = f"totals_under_{RC.totals_region(line, median)}"
            u, cal = self._cal(fam, under)
            o = 1.0 - u - push
            out.append(Selection("total_games", f"over {line}", max(o, 1e-9),
                                 fair_price(max(o, 1e-9), push), push, cal))
            out.append(Selection("total_games", f"under {line}", max(u, 1e-9),
                                 fair_price(max(u, 1e-9), push), push, cal))

        # --- games handicap -------------------------------------------------
        diffs: dict[int, float] = {}
        for (ga, gb), p in dist.games.items():
            diffs[ga - gb] = diffs.get(ga - gb, 0.0) + p
        for twice in range(2 * min(diffs) - 1, 2 * max(diffs) + 2):
            h = twice / 2
            a_cov = sum(p for d, p in diffs.items() if d > h)
            b_cov = sum(p for d, p in diffs.items() if d < h)
            push = diffs.get(int(h), 0.0) if float(h).is_integer() else 0.0
            out.append(Selection("games_handicap", f"A {h:+g}", max(a_cov, 1e-9),
                                 fair_price(max(a_cov, 1e-9), push), push))
            out.append(Selection("games_handicap", f"B {-h:+g}", max(b_cov, 1e-9),
                                 fair_price(max(b_cov, 1e-9), push), push))

        # --- tiebreak yes/no -------------------------------------------------
        out.append(Selection("tiebreak", "yes", dist.tiebreak_any,
                             fair_price(dist.tiebreak_any)))
        out.append(Selection("tiebreak", "no", 1 - dist.tiebreak_any,
                             fair_price(1 - dist.tiebreak_any)))

        # --- per-player games -------------------------------------------------
        for who, player_a in (("A", True), ("B", False)):
            ppmf = dist.player_games_pmf(player_a)
            pmed = _median_of(ppmf)
            for twice in range(2 * min(ppmf) - 1, 2 * max(ppmf) + 2):
                line = twice / 2
                over = sum(p for g, p in ppmf.items() if g > line)
                under = sum(p for g, p in ppmf.items() if g < line)
                push = ppmf.get(int(line), 0.0) if float(line).is_integer() else 0.0
                out.append(Selection(f"player_games_{who}", f"over {line}",
                                     max(over, 1e-9),
                                     fair_price(max(over, 1e-9), push), push))
                out.append(Selection(f"player_games_{who}", f"under {line}",
                                     max(under, 1e-9),
                                     fair_price(max(under, 1e-9), push), push))
        return out


def _median_of(pmf: dict[int, float]) -> float:
    cum = 0.0
    for g in sorted(pmf):
        cum += pmf[g]
        if cum >= 0.5:
            return float(g)
    return float(max(pmf))


@lru_cache(maxsize=8)
def _pricer(as_of: date, allow_holdout: bool) -> Pricer:
    return Pricer(as_of, allow_holdout=allow_holdout)


def price_match(player_a: str, player_b: str, tournament_id: str, year: int,
                surface: str, venue: str | None, as_of_date: date,
                allow_holdout: bool = False) -> PricedMatch:
    """Price every market for one matchup, as of ``as_of_date``.

    The format is looked up from (tournament_id, year); there is no caller
    flag for it. ``venue`` is accepted for call-site readability — the venue
    index is keyed on ``tournament_id`` and surface, per Stage 0's finding
    that tournament names are not stable identifiers.
    """
    return _pricer(as_of_date, allow_holdout).price(
        player_a, player_b, tournament_id, year, surface, venue)


def demo() -> None:
    """Worked example (ground rule 10)."""
    m = pd.read_parquet(C.PROCESSED_DIR / "matches.parquet")
    sample = m[(m["split"] == "tune") & m["in_scope"]
               & m["outcome"].eq("completed")].iloc[500]
    priced = price_match(sample["winner_id"], sample["loser_id"],
                         sample["tourney_code"], int(sample["season_file"]),
                         sample["surface"], sample["tourney_name"],
                         sample["date"])

    print(f"{sample['winner_name']} vs {sample['loser_name']} "
          f"({sample['tourney_name']} {sample['season_file']}, {sample['surface']})")
    print(f"actual result: {sample['score']}\n")
    md = priced.metadata
    print(f"pa {md['pa']:.4f}  pb {md['pb']:.4f}  "
          f"format bo{md['format']['best_of']} {md['format']['final_set']} "
          f"[{md['format']['provenance']}]")
    print(f"effective n: A {md['player_a_effective_n']:.0f}, "
          f"B {md['player_b_effective_n']:.0f}   "
          f"elo/serve disagreement {md['elo_serve_disagreement_pp']:+.1f}pp")
    print(f"venue x{md['venue_multiplier']:.4f} "
          f"(measured={md['venue_measured']})   "
          f"corrections {md['corrections']['applied']}   "
          f"maps {md['calibration_maps'] or 'none'}")
    print(f"ladder upper bound {md['truncation_point_games']} games\n")

    for market in ("match_winner", "set_betting", "tiebreak"):
        for s in priced.market(market)[:6]:
            print(f"  {s.market:15s} {s.selection:10s} p {s.probability:.4f}  "
                  f"fair {s.fair_decimal:8.3f}"
                  + ("  [calibrated]" if s.calibrated else ""))
    mid = [s for s in priced.market("total_games")
           if abs(float(s.selection.split()[-1]) - _median_of_priced(priced)) < 1.6]
    for s in mid[:8]:
        print(f"  {s.market:15s} {s.selection:10s} p {s.probability:.4f}  "
              f"fair {s.fair_decimal:8.3f}  push {s.push_probability:.4f}")
    print(f"\n{len(priced.selections)} selections priced across "
          f"{len({s.market for s in priced.selections})} markets")


def _median_of_priced(priced: PricedMatch) -> float:
    unders = [(float(s.selection.split()[-1]), s.probability)
              for s in priced.market("total_games") if s.selection.startswith("under")]
    for line, p in sorted(unders):
        if p >= 0.5:
            return line
    return unders[-1][0] if unders else 0.0


if __name__ == "__main__":
    demo()
