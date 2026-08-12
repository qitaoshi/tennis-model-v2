"""Stage 8 — recalibration of the priced probability families.

Maps are fitted on FIT+TUNE predictions against outcomes, frozen to disk, and
applied as the final step in price.py.

WHAT IS ACTUALLY SHIPPING, as of the 2026-08-12 refit. Read this before
assuming, because an earlier version of this docstring described maps that
were not the ones on disk, and that gap is what forced the whole audit.

Six families are priced: ``match_winner``, ``totals_under_low`` / ``_mid`` /
``_high`` (the over/under ladder split by line region relative to the
distribution's own median, since lines deep in a tail calibrate differently
from lines near the middle), ``set_score``, and ``games_handicap``.

Only THREE of them carry a map. ``scripts/refit_calibration.py`` applies a
map to a family only where it beats identity on TUNE, and on the 2026-08-12
refit it beat identity on the three totals families alone:

* mapped:   totals_under_low, totals_under_mid, totals_under_high
* identity: match_winner, set_score, games_handicap

``CalibrationMaps.apply`` passes unknown families through unchanged, so an
absent family IS identity for it — there is no second code path.

The method is Platt in logit space over a 5-year window, NOT isotonic.
Isotonic was the original choice, for the good reason that it can only
reorder probabilities monotonically. In practice it saturates: a top bin
holding few same-resolving samples returns exactly 1.0, making a fair price
1.00. Every isotonic and blended candidate was rejected on that shape test
in the 2026-08-12 refit, so Platt won by being the only non-degenerate
family of candidates, not by scoring best.

KNOWN DEFECT, shipped deliberately and not yet fixed: totals_under_low
regressed on the TEST read (ECE 0.0065 -> 0.0121) after improving on TUNE.
Do not "fix" this by dropping its map — that would be a selection made on a
TEST result. See fitted_params.json["calibration_refit"] for the full
constraint, which also covers why any rule devised now is contaminated.

The pre-cutoff TEST set was the ONE AND ONLY gate for these maps and was
spent on 2026-08-12. Nothing was fitted on it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from model import constants as C

MAPS_PATH: Path = C.PROCESSED_DIR / "calibration_maps.pkl"

#: Total-games line regions, as offsets from the predicted median total.
TOTALS_REGIONS = ("low", "mid", "high")


def totals_region(line: float, median_total: float) -> str:
    """Which region of the ladder a line sits in for this match."""
    d = line - median_total
    if d < -2.5:
        return "low"
    if d > 2.5:
        return "high"
    return "mid"


#: Probabilities are clipped this far from 0/1 before taking a logit, so a
#: p of exactly 0 or 1 cannot produce an infinite feature.
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(q / (1 - q))


@dataclass
class PlattMap:
    """Two-parameter logistic recalibration in logit space.

    ``p_cal = sigmoid(a * logit(p) + b)``.

    Isotonic is the right default in the body of the distribution, where there
    are enough samples per step. It is the wrong tool in a thin tail: it fits a
    step function, and the match_winner map carries only seven steps below
    p=0.2 — so a longshot's calibrated probability is decided by a handful of
    matches and jumps discontinuously.

    This has two parameters and is smooth everywhere, which is what a tail
    wants. ``a`` is the interesting one: a < 1 shrinks every probability toward
    even money, which is the correction a model that overrates longshots needs,
    and it applies that correction as a continuous function of how extreme the
    prediction is rather than as isolated steps.

    Duck-types ``IsotonicRegression.predict`` so ``CalibrationMaps`` does not
    need to know which kind of map it holds.
    """

    a: float
    b: float

    def predict(self, p: np.ndarray) -> np.ndarray:
        z = self.a * _logit(p) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def fit(p: np.ndarray, y: np.ndarray) -> "PlattMap":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(_logit(p).reshape(-1, 1), (np.asarray(y) > 0.5).astype(int))
        return PlattMap(a=float(lr.coef_[0][0]), b=float(lr.intercept_[0]))


@dataclass
class BlendedMap:
    """Isotonic in the body, Platt in the tails, as a monotone lookup table.

    Neither map is best everywhere: isotonic wins where the data is dense
    because it can follow an arbitrary monotone shape, and Platt wins in the
    tails because it does not run out of samples. This uses each where it is
    strong and crossfades over a band so the result stays continuous — a hard
    switch would put a jump in the middle of the price ladder.

    The crossfade is baked onto a fixed grid at fit time rather than evaluated
    per call. That is not an optimisation: a weighted average of two monotone
    curves is NOT itself monotone, and a calibration map that is not monotone
    can reorder two selections and invent discrimination the model never had.
    Precomputing lets the running maximum enforce monotonicity once, on the
    grid, instead of hoping the blend happens to behave.
    """

    xs: np.ndarray
    ys: np.ndarray

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(np.asarray(p, dtype=float), 0.0, 1.0),
                         self.xs, self.ys)

    @staticmethod
    def fit(iso: IsotonicRegression, platt: PlattMap, lo: float, hi: float,
            band: float, n_grid: int = 2001) -> "BlendedMap":
        xs = np.linspace(0.0, 1.0, n_grid)
        w = np.ones_like(xs)
        below, above = xs < lo, xs > hi
        w[below] = np.clip((xs[below] - (lo - band)) / band, 0, 1)
        w[above] = np.clip(((hi + band) - xs[above]) / band, 0, 1)
        ys = w * iso.predict(xs) + (1 - w) * platt.predict(xs)
        # The crossfade can dip where the two curves cross; the running max
        # removes exactly those dips and leaves the rest of the curve alone.
        ys = np.maximum.accumulate(np.clip(ys, 0.0, 1.0))
        return BlendedMap(xs=xs, ys=ys)


#: Candidate map families, tried head to head on TUNE. "isotonic" is the
#: incumbent that shipped with the original build.
METHODS = ("isotonic", "platt", "blended")


def _fit_one(method: str, p: np.ndarray, y: np.ndarray):
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        return iso
    if method == "platt":
        return PlattMap.fit(p, y)
    if method == "blended":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        return BlendedMap.fit(iso, PlattMap.fit(p, y), lo=0.20, hi=0.80, band=0.10)
    raise ValueError(f"unknown calibration method: {method!r}")


@dataclass
class CalibrationMaps:
    """Frozen isotonic maps, one per priced family."""

    maps: dict[str, IsotonicRegression | PlattMap | BlendedMap]
    n_fitted: dict[str, int]
    #: Which candidate family each map came from. Defaults to isotonic so a
    #: pickle written by the original build still loads.
    method: dict[str, str] = field(default_factory=dict)

    def apply(self, family: str, p: float | np.ndarray) -> float | np.ndarray:
        """Calibrated probability. Unknown families pass through unchanged."""
        iso = self.maps.get(family)
        arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = arr if iso is None else np.clip(iso.predict(np.clip(arr, 0, 1)), 1e-6, 1 - 1e-6)
        return float(out[0]) if np.isscalar(p) or np.ndim(p) == 0 else out

    def save(self, path: Path = MAPS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))
        return path

    @staticmethod
    def load(path: Path = MAPS_PATH) -> "CalibrationMaps":
        return pickle.loads(path.read_bytes())


def fit_maps(samples: dict[str, tuple[np.ndarray, np.ndarray]],
             method: str | dict[str, str] = "isotonic") -> CalibrationMaps:
    """Fit one map per family from (predicted, outcome) pairs.

    ``method`` is either one name applied to every family, or a per-family
    dict — families calibrate differently, so the winner need not be the same
    for match_winner as for the totals ladder.
    """
    maps, n, used = {}, {}, {}
    for family, (p, y) in samples.items():
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 200:
            continue
        how = method if isinstance(method, str) else method.get(family, "isotonic")
        maps[family] = _fit_one(how, p[ok], y[ok])
        n[family] = int(ok.sum())
        used[family] = how
    return CalibrationMaps(maps, n, used)


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability diagram as a table: predicted vs observed by bin."""
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return []
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "n": int(m.sum()), "predicted": float(p[m].mean()),
                     "observed": float(y[m].mean()),
                     "gap": float(y[m].mean() - p[m].mean())})
    return rows


def calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: n-weighted mean |observed - predicted|."""
    rows = reliability(p, y, n_bins)
    if not rows:
        return float("nan")
    total = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["gap"]) for r in rows) / total)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[ok] - y[ok]) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(p) & np.isfinite(y)
    q = np.clip(p[ok], 1e-9, 1 - 1e-9)
    return float(-np.mean(y[ok] * np.log(q) + (1 - y[ok]) * np.log(1 - q)))


def demo() -> None:
    """Worked example (ground rule 10)."""
    rng = np.random.default_rng(C.MC_SEED)
    # a deliberately overconfident forecaster
    truth = rng.random(20_000)
    pred = np.clip(0.5 + 1.6 * (truth - 0.5), 0.01, 0.99)
    y = (rng.random(20_000) < truth).astype(float)

    print(f"before: ECE {calibration_error(pred, y):.4f}, "
          f"Brier {brier(pred, y):.4f}, log-loss {log_loss(pred, y):.4f}")
    for how in METHODS:
        maps = fit_maps({"match_winner": (pred, y)}, method=how)
        cal = np.asarray(maps.apply("match_winner", pred))
        print(f"{how:9s} ECE {calibration_error(cal, y):.4f}, "
              f"Brier {brier(cal, y):.4f}, log-loss {log_loss(cal, y):.4f}")

    maps = fit_maps({"match_winner": (pred, y)}, method="isotonic")
    cal = np.asarray(maps.apply("match_winner", pred))
    print("\nreliability after isotonic:")
    for r in reliability(cal, y):
        print(f"  bin {r['bin']} n={r['n']:5d} predicted {r['predicted']:.3f} "
              f"observed {r['observed']:.3f} gap {r['gap']:+.3f}")
    print(f"\nunknown families pass through: "
          f"{maps.apply('not_a_family', 0.37):.4f}")

    # Self-check: the properties every map family must have.
    grid = np.linspace(0.001, 0.999, 400)
    for how in METHODS:
        m = fit_maps({"f": (pred, y)}, method=how)
        q = np.asarray(m.apply("f", grid))
        assert np.all(np.diff(q) >= -1e-9), f"{how} is not monotone"
        assert np.all((q >= 0) & (q <= 1)), f"{how} left [0, 1]"
        assert m.method["f"] == how
    # An overconfident forecaster needs its tail pulled in, so Platt's slope
    # must come out below 1. This is the whole reason the family exists.
    pl = PlattMap.fit(pred, y)
    assert pl.a < 1.0, f"expected shrinkage, got a={pl.a:.3f}"
    # Blended must be continuous across both crossfade bands.
    b = _fit_one("blended", pred, y)
    fine = np.linspace(0.0, 1.0, 5001)
    assert np.max(np.abs(np.diff(b.predict(fine)))) < 0.02, "blend is not smooth"
    # and it must agree with each parent where that parent has full weight
    pure_iso = _fit_one("isotonic", pred, y)
    assert abs(b.predict(np.array([0.5]))[0] - pure_iso.predict([0.5])[0]) < 0.02
    # A map pickled without the method field (the original build's artifact)
    # must still load and apply.
    old = CalibrationMaps(maps={"f": pl}, n_fitted={"f": 10})
    assert 0 < old.apply("f", 0.3) < 1 and old.method == {}
    print(f"\nself-check passed: platt slope a={pl.a:.3f}, b={pl.b:+.3f}")


if __name__ == "__main__":
    demo()
