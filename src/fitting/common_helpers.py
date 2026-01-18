import re
from typing import Iterable, List, Tuple

import numpy as np
from numpy.polynomial import Polynomial

Point = Tuple[float, float]


def to_xy_arrays(points: Iterable[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert iterable of (x, y) points to float numpy arrays, skipping invalid entries."""
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        if p is None or len(p) != 2:
            continue
        x, y = p
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def x_errors_x_of_y(points: Iterable[Point], poly) -> np.ndarray:
    """
    Compute per-point x-errors for a curve x=f(y) evaluated at the given points.

    Args:
        points: Iterable of (x, y) ground-truth points.
        poly: Callable object that supports poly(y) -> x_pred (e.g., numpy Polynomial).

    Returns:
        1D numpy array of errors (x_pred - x_true). Empty array if no valid points.
    """
    x, y = to_xy_arrays(points)
    if x.size == 0:
        return np.asarray([], dtype=float)
    x_hat = poly(y)
    return np.asarray(x_hat - x, dtype=float)


def rmse_from_errors(err: np.ndarray) -> float:
    """Root mean squared error from an error vector."""
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err**2)))


def mae_from_errors(err: np.ndarray) -> float:
    """Mean absolute error from an error vector."""
    if err.size == 0:
        return float("nan")
    return float(np.mean(np.abs(err)))


def medae_from_errors(err: np.ndarray) -> float:
    """Median absolute error from an error vector."""
    if err.size == 0:
        return float("nan")
    return float(np.median(np.abs(err)))


def maxae_from_errors(err: np.ndarray) -> float:
    """Maximum absolute error from an error vector."""
    if err.size == 0:
        return float("nan")
    return float(np.max(np.abs(err)))


def poly_from_result_row(row):
    """
    Reconstruct a polynomial x = f(y) from stored CSV coefficients.

    The function reads coefficient columns (coef_c0 .. coef_c4) from a
    result row and builds a NumPy Polynomial in standard power basis.
    NaN coefficients are ignored, allowing reconstruction of lower-degree
    polynomials from a fixed-width schema.

    Args:
        row:
            A pandas Series or dict-like object containing polynomial
            coefficients under the keys 'coef_c0' .. 'coef_c4'.

    Returns:
        numpy.polynomial.Polynomial:
            Reconstructed polynomial x = f(y) in pixel coordinates.
    """
    coef = [
        row["coef_c0"],
        row["coef_c1"],
        row["coef_c2"],
        row["coef_c3"],
        row["coef_c4"],
    ]
    coef = [
        c for c in coef if not (c is None or (isinstance(c, float) and np.isnan(c)))
    ]
    return Polynomial(coef)


def sample_poly_x_of_y(poly, y_min, y_max, n=200):
    """
    Sample a polynomial x = f(y) over a given y-interval.

    The polynomial is evaluated at evenly spaced y-values between
    y_min and y_max. The resulting (x, y) pairs can be used for
    visualization or further analysis.

    Args:
        poly:
            A callable polynomial object implementing x = f(y)
            (e.g., numpy.polynomial.Polynomial).
        y_min:
            Lower bound of the y-range (inclusive).
        y_max:
            Upper bound of the y-range (inclusive).
        n:
            Number of sample points along the y-axis.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Arrays (xs, ys) containing sampled x- and y-coordinates.
    """
    ys = np.linspace(float(y_min), float(y_max), n)
    xs = poly(ys)
    return xs, ys


def split_point_string_to_points(string):
    """
    Parse a CVAT polyline point string into a list of (x, y) coordinates.

    The expected format is a sequence of comma-separated coordinate pairs,
    e.g. "x1,y1 x2,y2 ...". Floating-point and negative values are supported.

    Args:
        string (str):
            Polyline point string as stored in the annotation file.

    Returns:
        list[tuple[float, float]]:
            List of (x, y) coordinate pairs. Returns an empty list if the
            input is invalid or empty.
    """
    _point_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")

    if not string or not isinstance(string, str):
        return []

    pts = _point_re.findall(
        string
    )  # Liste von Strings: [('x1','y1'), ('x2','y2'), ...]
    return [(float(x), float(y)) for x, y in pts]


def get_points_of_all_3_lanes(data_row):
    """
    Extract polyline point lists for left, center, and right lane from one dataset row.

    The function reads the lane columns ("left lane", "center lane", "right lane") and
    converts each polyline point string into a list of (x, y) tuples using
    `split_point_string_to_points`.

    Args:
        data_row:
            A pandas Series or dict-like row containing lane label fields.

    Returns:
        tuple[list[tuple[float, float]] | None, list[tuple[float, float]] | None, list[tuple[float, float]] | None]:
            (left_lane_points, center_lane_points, right_lane_points). Each entry may be
            an empty list or None if the lane is missing/unparsable, depending on the
            behavior of `split_point_string_to_points`.
    """
    left_lane_data = split_point_string_to_points(data_row["left lane"])
    center_lane_data = split_point_string_to_points(data_row["center lane"])
    right_lane_data = split_point_string_to_points(data_row["right lane"])
    return left_lane_data, center_lane_data, right_lane_data
