"""Stage 7 — corrections for the two ways the iid-points assumption fails.

The engine assumes points are iid. They are not, and the failures are
systematic in two separate directions:

(a) **Tiebreak frequency.** Real sets reach 6-6 more often than iid points
    predict, because both players raise their level on serve when the set is
    close. Corrected by inflating hold probability near the 5-5 / 6-6 states.
(b) **Total-games variance.** The engine's games distribution is too narrow,
    because a match's true (pa, pb) is not a fixed number — form, conditions
    and tactics wobble it. Corrected by mixing over a normal wobble in the
    level.

These are separate failures with separate corrections, validated separately.
One parameter does not fix both.

Everything here is measured CONDITIONING ON THE FORMAT RULE IN FORCE at match
time (via rules.py), and excluding retirements, walkovers and matches flagged
``score_string_suspect``. Without the format conditioning, the measured
"engine understates tiebreaks" gap would be partly a rule-mismatch artefact:
an advantage-set match cannot produce a deciding-set tiebreak at all.

PROVENANCE WEIGHTING is resolved by selection, not left open (ground rule 4).
Both corrections are measured three ways — documented-provenance matches only,
documented and inferred pooled, and inferred matches downweighted by a fitted
factor — and the scheme with the better TUNE calibration is written to
fitted_params.json.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.engine import match_distribution
from model.rules import FormatSpec

#: Provenance weighting schemes considered (ground rule 4: pick one, log it).
SCHEMES = ("documented_only", "pooled", "downweight_inferred")


@dataclass(frozen=True)
class CorrectionParams:
    """Stage 7's fitted corrections. Values live in fitted_params.json."""

    #: Additive inflation of hold probability at the 5-5 and 6-6 states.
    tiebreak_inflation: float = 0.0
    #: Standard deviation of the normal wobble applied to the LEVEL.
    level_sigma: float = 0.0
    #: Standard deviation of the normal wobble applied to the SPLIT. This is
    #: the lever that matters: iid points at a fixed split produce far more
    #: close sets than real tennis, because a real match's skill gap varies
    #: within the match. Wobbling the split produces the lopsided sets that
    #: are actually observed.
    split_sigma: float = 0.0
    #: Re-solve the centre split so the mixture reproduces the uncorrected
    #: match-win probability. Without this the wobble drags every favourite
    #: toward even money and destroys match-winner calibration, which Stages
    #: 3 and 4 already got right.
    recenter: bool = True
    #: Number of quadrature points used to mix over each wobble.
    n_mix: int = 5
    #: Weight on inferred-provenance matches when measuring (1.0 = pooled).
    inferred_weight: float = 1.0
    scheme: str = "pooled"


def _gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for mixing over a normal wobble."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


def _mix(level: float, split: float, spec: FormatSpec, params: CorrectionParams):
    """Mixture distribution over the level and split wobbles."""
    from model.engine import MatchDistribution

    nodes, weights = _gauss_hermite(params.n_mix)
    l_nodes = nodes if params.level_sigma > 0 else np.zeros(1)
    l_w = weights if params.level_sigma > 0 else np.ones(1)
    s_nodes = nodes if params.split_sigma > 0 else np.zeros(1)
    s_w = weights if params.split_sigma > 0 else np.ones(1)

    games: dict[tuple[int, int], float] = {}
    sets: dict[tuple[int, int], float] = {}
    p_a = tb_any = truncated = 0.0
    for zl, wl in zip(l_nodes, l_w):
        for zs, ws in zip(s_nodes, s_w):
            L = level + params.level_sigma * zl
            S = split + params.split_sigma * zs
            a = float(np.clip((L + S) / 2, 0.05, 0.95))
            b = float(np.clip((L - S) / 2, 0.05, 0.95))
            d = match_distribution(round(a, 3), round(b, 3), spec,
                                   close_inflation=params.tiebreak_inflation)
            w = wl * ws
            p_a += w * d.p_a
            tb_any += w * d.tiebreak_any
            truncated += w * d.truncated_mass
            for k, v in d.games.items():
                games[k] = games.get(k, 0.0) + w * v
            for k, v in d.sets.items():
                sets[k] = sets.get(k, 0.0) + w * v
    return MatchDistribution(p_a, sets, games, tb_any, truncated, spec)


def corrected_distribution(pa: float, pb: float, spec: FormatSpec,
                           params: CorrectionParams):
    """Match distribution with both corrections applied.

    The tiebreak correction moves hold probability at the close-set states
    only, symmetrically, so it changes tiebreak frequency without changing who
    wins. The variance correction mixes over a normal wobble in the split (and
    optionally the level), which produces the lopsided sets that iid points at
    a fixed split cannot.

    With ``recenter`` the centre split is re-solved so the mixture reproduces
    the uncorrected match-win probability: the mixture is there to fix the
    shape of the set and games distributions, not to relitigate who wins.
    """
    infl = params.tiebreak_inflation
    if params.split_sigma <= 0 and params.level_sigma <= 0:
        return match_distribution(round(pa, 4), round(pb, 4), spec,
                                  close_inflation=infl)

    level, split = pa + pb, pa - pb
    if not params.recenter or params.split_sigma <= 0:
        return _mix(level, split, spec, params)

    target = match_distribution(round(pa, 4), round(pb, 4), spec,
                                close_inflation=infl).p_a
    lo, hi = -0.45, 0.45
    if _mix(level, lo, spec, params).p_a > target:
        return _mix(level, lo, spec, params)
    if _mix(level, hi, spec, params).p_a < target:
        return _mix(level, hi, spec, params)
    for _ in range(12):  # p_a is monotone in the centre split
        mid = (lo + hi) / 2
        if _mix(level, mid, spec, params).p_a < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return _mix(level, (lo + hi) / 2, spec, params)


def measure_tiebreak_gap(observed_tb: np.ndarray, predicted_tb: np.ndarray,
                         weights: np.ndarray | None = None) -> float:
    """Observed minus predicted tiebreak occurrence rate."""
    w = np.ones_like(observed_tb, dtype=float) if weights is None else weights
    return float(np.average(observed_tb, weights=w) - np.average(predicted_tb, weights=w))


def pit_values(cdf_at_actual: np.ndarray, cdf_before: np.ndarray) -> np.ndarray:
    """Randomized PIT for a discrete forecast, so uniformity is the target."""
    rng = np.random.default_rng(0)
    u = rng.random(len(cdf_at_actual))
    return cdf_before + u * (cdf_at_actual - cdf_before)


def pit_deviation(pit: np.ndarray, n_bins: int = 10,
                  weights: np.ndarray | None = None) -> float:
    """Mean absolute deviation of the PIT histogram from uniform.

    ``weights`` carries provenance weighting. It must be applied here rather
    than upstream: a caller that instead drops rows with weight zero turns a
    *downweighting* scheme into a pass-through, because a downweighted row
    still has weight > 0. That is exactly the bug this signature exists to
    prevent -- with weights folded in, ``downweight_inferred`` and ``pooled``
    can differ on the variance correction, which the Stage 7 provenance
    comparison in MODEL_PROMPT.md requires them to be able to do.
    """
    counts, _ = np.histogram(pit, bins=n_bins, range=(0, 1), weights=weights)
    total = float(counts.sum())
    return float(np.mean(np.abs(counts / max(total, 1e-12) - 1 / n_bins)))


def coverage(pit: np.ndarray, level: float = 0.8,
             weights: np.ndarray | None = None) -> float:
    """Empirical coverage of the central interval at ``level``.

    Provenance-weighted for the same reason as :func:`pit_deviation`.
    """
    inside = ((pit >= (1 - level) / 2) & (pit <= 1 - (1 - level) / 2))
    if weights is None:
        return float(np.mean(inside))
    return float(np.average(inside.astype(float), weights=weights))


def provenance_weights(provenance: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Per-match measurement weight under the selected provenance scheme."""
    documented = provenance == "documented"
    if params.scheme == "documented_only":
        return documented.astype(float)
    if params.scheme == "downweight_inferred":
        return np.where(documented, 1.0, params.inferred_weight)
    return np.ones(len(provenance), dtype=float)


def demo() -> None:
    """Worked example (ground rule 10)."""
    from model.rules import rules_for

    spec = rules_for("339", 2019)
    base = match_distribution(0.64, 0.60, spec)
    print("uncorrected: P(A) %.4f  P(tiebreak) %.4f  games sd %.3f"
          % (base.p_a, base.tiebreak_any, _sd(base.total_games_pmf())))

    for infl in (0.0, 0.005, 0.01):
        for sigma in (0.0, 0.01, 0.02):
            d = corrected_distribution(0.64, 0.60, spec,
                                       CorrectionParams(tiebreak_inflation=infl,
                                                        level_sigma=sigma))
            print(f"  inflation {infl:.3f} sigma {sigma:.3f}: "
                  f"P(A) {d.p_a:.4f}  P(tiebreak) {d.tiebreak_any:.4f}  "
                  f"games sd {_sd(d.total_games_pmf()):.3f}")


def _sd(pmf: dict[int, float]) -> float:
    xs = np.array(list(pmf), dtype=float)
    ps = np.array(list(pmf.values()))
    mean = float(np.sum(xs * ps))
    return float(np.sqrt(np.sum(ps * (xs - mean) ** 2)))


if __name__ == "__main__":
    demo()
