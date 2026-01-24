import math
import re
from typing import Iterable, List, Tuple

import cv2
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

    The function reads coefficient columns (coef_c0 .. coef_c6) from a
    result row and builds a NumPy Polynomial in standard power basis.
    NaN coefficients are ignored, allowing reconstruction of lower-degree
    polynomials from a fixed-width schema.

    Args:
        row:
            A pandas Series or dict-like object containing polynomial
            coefficients under the keys 'coef_c0' .. 'coef_c6'.

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
        row["coef_c5"],
        row["coef_c6"],
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


def densify_polyline(
    points: Iterable[Point],
    *,
    step_px: float = 10.0,
    dedup_eps: float = 1e-6,
    include_last: bool = True,
) -> List[Point]:
    """
    Linearly densify a polyline by inserting points along each segment.

    The function walks through consecutive point pairs and inserts additional
    points so that the distance between neighboring samples is roughly `step_px`
    (measured in Euclidean pixel distance). This is a pure geometric densification
    (no smoothing).

    Invalid points (None, wrong length, None coords, non-finite) are skipped.

    Args:
        points:
            Iterable of (x, y) points describing the polyline in order.
        step_px:
            Desired spacing between consecutive output points in pixels.
            Must be > 0. Typical values: 5..20 depending on your label density.
        dedup_eps:
            Distance threshold for de-duplicating consecutive points.
            If the next candidate is within `dedup_eps` of the last output point,
            it is skipped.
        include_last:
            If True, ensures the last valid input point is included in the output.

    Returns:
        List[Point]:
            Densified list of (x, y) points in the original order.

    Raises:
        ValueError:
            If `step_px` is not > 0.
    """
    if step_px <= 0:
        raise ValueError("step_px must be > 0")

    # --- sanitize input (keep order)
    clean: List[Point] = []
    for p in points:
        if p is None or len(p) != 2:
            continue
        x, y = p
        if x is None or y is None:
            continue
        x_f = float(x)
        y_f = float(y)
        if not (math.isfinite(x_f) and math.isfinite(y_f)):
            continue
        clean.append((x_f, y_f))

    if not clean:
        return []

    out: List[Point] = [clean[0]]

    def _is_dup(a: Point, b: Point) -> bool:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) <= (dedup_eps * dedup_eps)

    for (x0, y0), (x1, y1) in zip(clean, clean[1:]):
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)

        # Skip zero-length segments
        if seg_len <= dedup_eps:
            if not _is_dup(out[-1], (x1, y1)):
                out.append((x1, y1))
            continue

        # Number of sub-steps to keep spacing <= step_px
        n_steps = int(math.floor(seg_len / step_px))

        # Insert intermediate points (exclude start, exclude end)
        # t in (0, 1)
        for k in range(1, n_steps + 1):
            t = (k * step_px) / seg_len
            if t >= 1.0:
                break
            xi = x0 + t * dx
            yi = y0 + t * dy
            cand = (xi, yi)
            if not _is_dup(out[-1], cand):
                out.append(cand)

        # Append end point (or skip if not desired)
        if include_last:
            if not _is_dup(out[-1], (x1, y1)):
                out.append((x1, y1))
        else:
            # If we don't include last, still keep continuity by de-duping
            if not _is_dup(out[-1], (x1, y1)):
                out.append((x1, y1))

    # If include_last, ensure final valid input point is present
    if include_last and not _is_dup(out[-1], clean[-1]):
        out.append(clean[-1])

    return out


def stitch_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gap: int = 16,
    pad_color: tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """
    Stitch two images horizontally with a small gap.

    If heights differ, images are padded (bottom) to the max height.

    Args:
        left: Left BGR image.
        right: Right BGR image.
        gap: Gap between images in pixels.
        pad_color: BGR color for padding area.

    Returns:
        Stitched BGR image.
    """
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    h = max(h1, h2)

    def _pad_to_h(img: np.ndarray, target_h: int) -> np.ndarray:
        ih, iw = img.shape[:2]
        if ih == target_h:
            return img
        pad = target_h - ih
        return cv2.copyMakeBorder(
            img, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=pad_color
        )

    left_p = _pad_to_h(left, h)
    right_p = _pad_to_h(right, h)

    spacer = np.full((h, gap, 3), pad_color, dtype=np.uint8)
    return cv2.hconcat([left_p, spacer, right_p])
