from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.polynomial import Polynomial

Point = Tuple[float, float]


@dataclass(frozen=True)
class PolyFitXY:
    """Polynomial fit for x as a function of y (x = f(y))."""

    degree: int
    poly: Polynomial  # NumPy Polynomial object
    y_min: float
    y_max: float
    n_points: int
    rmse: float


def _to_xy(points: Iterable[Point]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        if p is None or len(p) != 2:
            continue
        x, y = p
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_poly_x_of_y(
    points: Iterable[Point],
    degree: int,
    *,
    min_points: Optional[int] = None,
) -> Optional[PolyFitXY]:
    """
    Fit a polynomial x = f(y) to lane points.

    Args:
        points: Iterable of (x, y) points.
        degree: Polynomial degree (e.g., 2, 3, 4).
        min_points: Minimum required points (defaults to degree + 1).

    Returns:
        PolyFitXY if successful, otherwise None.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    x, y = _to_xy(points)
    req = min_points if min_points is not None else (degree + 1)
    if x.size < req:
        return None

    # Degenerate input guard: y needs variation
    if np.isclose(np.std(y), 0.0):
        return None

    poly = Polynomial.fit(y, x, deg=degree)  # x = f(y)
    x_hat = poly(y)
    rmse = float(np.sqrt(np.mean((x_hat - x) ** 2)))

    return PolyFitXY(
        degree=degree,
        poly=poly,
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
        n_points=int(x.size),
        rmse=rmse,
    )


def sample_poly_x_of_y(
    fit: PolyFitXY,
    *,
    num: int = 120,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a fitted curve as (x(y), y) points.

    Returns:
        (x_samples, y_samples)
    """
    ymin = fit.y_min if y_min is None else float(y_min)
    ymax = fit.y_max if y_max is None else float(y_max)
    ys = np.linspace(ymin, ymax, num=num, dtype=float)
    xs = np.polyval(fit.coeffs, ys)
    return xs, ys
