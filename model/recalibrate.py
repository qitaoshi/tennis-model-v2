"""Stage 8 — isotonic recalibration of the priced probability families.

Isotonic regression maps, fitted on FIT+TUNE predictions against outcomes,
frozen to disk, and applied as the final step in price.py. Isotonic is chosen
because it can only reorder probabilities monotonically: it fixes calibration
without inventing discrimination the model does not have.

One map per family that is actually priced, because their miscalibration is
not the same shape:

* ``match_winner``
* ``totals_under_<region>`` — the over/under ladder, split by line region
  relative to the distribution's own median, since lines deep in a tail
  calibrate differently from lines near the middle
* ``set_score`` — exact set-score selections

The maps are frozen artifacts. The gate that judges them is the ONE AND ONLY
use of the pre-cutoff TEST set, and nothing is fitted on it.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
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


@dataclass
class CalibrationMaps:
    """Frozen isotonic maps, one per priced family."""

    maps: dict[str, IsotonicRegression]
    n_fitted: dict[str, int]

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


def fit_maps(samples: dict[str, tuple[np.ndarray, np.ndarray]]) -> CalibrationMaps:
    """Fit one isotonic map per family from (predicted, outcome) pairs."""
    maps, n = {}, {}
    for family, (p, y) in samples.items():
        ok = np.isfinite(p) & np.isfinite(y)
        if ok.sum() < 200:
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p[ok], y[ok])
        maps[family] = iso
        n[family] = int(ok.sum())
    return CalibrationMaps(maps, n)


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
    maps = fit_maps({"match_winner": (pred, y)})
    cal = maps.apply("match_winner", pred)
    print(f"after:  ECE {calibration_error(cal, y):.4f}, "
          f"Brier {brier(cal, y):.4f}, log-loss {log_loss(cal, y):.4f}")
    print("\nreliability after:")
    for r in reliability(cal, y):
        print(f"  bin {r['bin']} n={r['n']:5d} predicted {r['predicted']:.3f} "
              f"observed {r['observed']:.3f} gap {r['gap']:+.3f}")
    print(f"\nunknown families pass through: "
          f"{maps.apply('not_a_family', 0.37):.4f}")


if __name__ == "__main__":
    demo()
