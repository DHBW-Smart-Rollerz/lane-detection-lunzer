from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.polynomial import Polynomial

from .common_helpers import densify_polyline, to_xy_arrays

Point = Tuple[float, float]


@dataclass(frozen=True)
class PolyFitXY:
    """Polynomial fit for x as a function of y (x = f(y))."""

    degree: int
    poly: Polynomial  # NumPy Polynomial object
    poly_std_coeffs: np.ndarray
    y_min: float
    y_max: float
    n_points: int


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
        PolyFitXY: Data class object - if successful, otherwise None.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    x, y = to_xy_arrays(points)
    req = min_points if min_points is not None else (degree + 1)
    if x.size < req:
        return None

    # Degenerate input guard: y needs variation
    if np.isclose(np.std(y), 0.0):
        return None

    poly_fit = Polynomial.fit(y, x, deg=degree)  # x = f(y)

    return PolyFitXY(
        degree=degree,
        poly=poly_fit,
        poly_std_coeffs=poly_fit.convert().coef,
        y_min=float(np.min(y)),
        y_max=float(np.max(y)),
        n_points=int(x.size),
    )
