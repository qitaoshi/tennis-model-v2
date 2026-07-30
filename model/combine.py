"""Stage 4 — combine serve rates and Elo ratings into (pa, pb).

The engine needs two point-win probabilities. Their SUM (the level, i.e. how
serve-dominated the match is) comes from the Stage 2 rate model, which is what
that model actually measures. Their DIFFERENCE (the split, i.e. who is better)
is a blend of two opinions:

    split = (1 - w) * serve_gap + w * elo_gap

``elo_gap`` is the serve gap that reproduces Elo's win probability at this
match's level, found by numerically inverting the Stage 1 engine. p_match is
monotonic in the split at fixed level, so the inversion is well posed; it is
also called once per match during fitting, so the engine is evaluated over a
(level, gap) grid once per format and the inverse read off by interpolation.

The disagreement between the two opinions is carried out as metadata in
percentage points, never silently absorbed into the blend: a match where Elo
and the serve rates disagree by 15pp is a different object from one where they
agree, even if the blend lands in the same place.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from model.engine import p_match
from model.rules import FormatSpec

#: Grid the engine is evaluated on for inversion.
LEVEL_GRID = np.round(np.arange(1.00, 1.51, 0.02), 4)
GAP_GRID = np.round(np.arange(-0.40, 0.4001, 0.01), 4)


@dataclass(frozen=True)
class CombineParams:
    """Stage 4's hyperparameters. Fitted values live in fitted_params.json."""

    #: Blend weight on Elo, overall and per sample-size bucket.
    w: float = 0.5
    w_both_well: float | None = None
    w_one_thin: float | None = None
    w_both_thin: float | None = None
    #: Effective serve points below which a player counts as thin.
    thin_n: float = 1000.0

    def weight_for(self, n_a: float, n_b: float) -> float:
        thin = int(n_a < self.thin_n) + int(n_b < self.thin_n)
        by_bucket = (self.w_both_well, self.w_one_thin, self.w_both_thin)[thin]
        return self.w if by_bucket is None else by_bucket


def bucket_of(n_a: np.ndarray, n_b: np.ndarray, thin_n: float) -> np.ndarray:
    """0 = both well sampled, 1 = one thin, 2 = both thin."""
    return (n_a < thin_n).astype(int) + (n_b < thin_n).astype(int)


def _spec_key(spec: FormatSpec) -> tuple:
    return (spec.best_of, spec.games_to_win_set, spec.tb_at, spec.tb_to,
            spec.final_set, spec.final_tb_at, spec.final_tb_to)


@lru_cache(maxsize=32)
def _inversion_table(key: tuple) -> np.ndarray:
    """P(A wins) on the (level, gap) grid for one format.

    Cached per format: the grid is ~1,300 exact engine evaluations, and every
    match of that format then costs two interpolations instead of a solve.
    """
    best_of, gws, tb_at, tb_to, final_set, ftb_at, ftb_to = key
    spec = FormatSpec(best_of=best_of, games_to_win_set=gws, tb_at=tb_at,
                      tb_to=tb_to, final_set=final_set, final_tb_at=ftb_at,
                      final_tb_to=ftb_to, provenance="documented",
                      source="inversion grid")
    table = np.zeros((len(LEVEL_GRID), len(GAP_GRID)))
    for i, level in enumerate(LEVEL_GRID):
        for j, gap in enumerate(GAP_GRID):
            pa, pb = (level + gap) / 2, (level - gap) / 2
            if not (0.01 < pa < 0.99 and 0.01 < pb < 0.99):
                table[i, j] = np.nan
                continue
            table[i, j] = p_match(round(pa, 6), round(pb, 6), spec)
    return table


def elo_gap_for(elo_p: float, level: float, spec: FormatSpec) -> float:
    """Serve gap reproducing ``elo_p`` at this level, by grid interpolation."""
    table = _inversion_table(_spec_key(spec))
    lvl = float(np.clip(level, LEVEL_GRID[0], LEVEL_GRID[-1]))
    i = int(np.clip(np.searchsorted(LEVEL_GRID, lvl) - 1, 0, len(LEVEL_GRID) - 2))
    lo, hi = LEVEL_GRID[i], LEVEL_GRID[i + 1]
    frac = 0.0 if hi == lo else (lvl - lo) / (hi - lo)

    gaps = []
    for row in (table[i], table[i + 1]):
        ok = np.isfinite(row)
        gaps.append(np.interp(np.clip(elo_p, 1e-4, 1 - 1e-4),
                              row[ok], GAP_GRID[ok]))
    return float(gaps[0] * (1 - frac) + gaps[1] * frac)


def p_from_grid(level: np.ndarray, gap: np.ndarray, key: tuple) -> np.ndarray:
    """P(A wins) read off the inversion grid by bilinear interpolation.

    Used only where the engine would otherwise be called once per candidate
    hyperparameter per match — w selection, not pricing. price.py always calls
    the exact engine.
    """
    table = _inversion_table(key)
    lv = np.clip(level, LEVEL_GRID[0], LEVEL_GRID[-1])
    gp = np.clip(gap, GAP_GRID[0], GAP_GRID[-1])
    i = np.clip(np.searchsorted(LEVEL_GRID, lv) - 1, 0, len(LEVEL_GRID) - 2)
    j = np.clip(np.searchsorted(GAP_GRID, gp) - 1, 0, len(GAP_GRID) - 2)
    fl = (lv - LEVEL_GRID[i]) / (LEVEL_GRID[i + 1] - LEVEL_GRID[i])
    fg = (gp - GAP_GRID[j]) / (GAP_GRID[j + 1] - GAP_GRID[j])
    out = ((1 - fl) * (1 - fg) * table[i, j]
           + (1 - fl) * fg * table[i, j + 1]
           + fl * (1 - fg) * table[i + 1, j]
           + fl * fg * table[i + 1, j + 1])
    return np.clip(np.nan_to_num(out, nan=0.5), 1e-6, 1 - 1e-6)


def combine(serve_pa: float, serve_pb: float, elo_p: float, spec: FormatSpec,
            params: CombineParams, n_a: float, n_b: float) -> dict:
    """Blend one match's opinions into (pa, pb), keeping the disagreement.

    Returns the blended point probabilities, the weight actually used, both
    sides' effective sample sizes, and the elo-vs-serve disagreement in
    percentage points of match win probability.
    """
    level = serve_pa + serve_pb
    serve_gap = serve_pa - serve_pb
    elo_gap = elo_gap_for(elo_p, level, spec)
    w = params.weight_for(n_a, n_b)
    split = (1 - w) * serve_gap + w * elo_gap

    pa, pb = (level + split) / 2, (level - split) / 2
    serve_p = p_match(round(serve_pa, 3), round(serve_pb, 3), spec)
    return {
        "pa": float(np.clip(pa, 0.01, 0.99)),
        "pb": float(np.clip(pb, 0.01, 0.99)),
        "level": float(level),
        "serve_gap": float(serve_gap),
        "elo_gap": float(elo_gap),
        "w_used": float(w),
        "n_a": float(n_a),
        "n_b": float(n_b),
        "bucket": int((n_a < params.thin_n) + (n_b < params.thin_n)),
        "elo_p": float(elo_p),
        "serve_p": float(serve_p),
        "disagreement_pp": float(100.0 * (elo_p - serve_p)),
    }


def combine_many(serve_pa: np.ndarray, serve_pb: np.ndarray, elo_p: np.ndarray,
                 spec_keys: list[tuple], params: CombineParams,
                 n_a: np.ndarray, n_b: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized blend for a whole evaluation window.

    Only the blend is vectorized; the engine calls behind ``elo_gap_for`` come
    from the cached grid, so no match triggers a fresh DP solve.
    """
    level = serve_pa + serve_pb
    serve_gap = serve_pa - serve_pb
    elo_gap = np.array([
        elo_gap_for(p, lv, FormatSpec(best_of=k[0], games_to_win_set=k[1],
                                      tb_at=k[2], tb_to=k[3], final_set=k[4],
                                      final_tb_at=k[5], final_tb_to=k[6],
                                      provenance="documented", source="grid"))
        for p, lv, k in zip(elo_p, level, spec_keys)
    ])
    bucket = bucket_of(n_a, n_b, params.thin_n)
    weights = np.array([params.weight_for(a, b) for a, b in zip(n_a, n_b)])
    split = (1 - weights) * serve_gap + weights * elo_gap
    return {
        "pa": np.clip((level + split) / 2, 0.01, 0.99),
        "pb": np.clip((level - split) / 2, 0.01, 0.99),
        "level": level, "serve_gap": serve_gap, "elo_gap": elo_gap,
        "w_used": weights, "bucket": bucket,
    }


def demo() -> None:
    """Worked example (ground rule 10)."""
    from model.rules import rules_for

    spec = rules_for("339", 2019)
    print("serve model says 0.64 / 0.60; Elo says the underdog wins 45%\n")
    for w in (0.0, 0.5, 1.0):
        out = combine(0.64, 0.60, 0.45, spec, CombineParams(w=w), 5000, 5000)
        print(f"  w={w}: pa {out['pa']:.4f} pb {out['pb']:.4f}  "
              f"serve_gap {out['serve_gap']:+.4f} elo_gap {out['elo_gap']:+.4f}  "
              f"serve_p {out['serve_p']:.4f} elo_p {out['elo_p']:.4f}  "
              f"disagreement {out['disagreement_pp']:+.1f}pp")

    print("\nthin-data bucketing (w rises when the serve rates are thin):")
    params = CombineParams(w_both_well=0.4, w_one_thin=0.6, w_both_thin=0.8)
    for na, nb, label in ((5000, 5000, "both well sampled"),
                          (5000, 200, "one thin"), (200, 150, "both thin")):
        out = combine(0.64, 0.60, 0.45, spec, params, na, nb)
        print(f"  {label:20s} bucket {out['bucket']} w {out['w_used']:.2f} "
              f"pa {out['pa']:.4f} pb {out['pb']:.4f}")


if __name__ == "__main__":
    demo()
