"""
Fit cubic clamped B-splines to lane annotations and export results to CSV.

This script mirrors fit_beziers.py, but fits a B-spline curve:

    C(t) = sum_{i=0..m-1} P_i * N_{i,p}(t)

Where:
- m = number of control points
- p = degree (fixed to 3 by default)
- N_{i,p}(t) are B-spline basis functions defined by a clamped (open uniform) knot vector.

Parameterization:
- chord-length parameterization on the (optionally densified) polyline -> t_k in [0,1]

Solve:
- Linear least squares separately for x and y:
    A(t) * ctrl_x ≈ x
    A(t) * ctrl_y ≈ y
  where A_{k,i} = N_{i,p}(t_k)

Outputs:
- Per image + lane + (n_control_points) record with ctrl_p{i}_{x|y} columns.
- Geometric errors (RMSE/MAE/MedAE/MaxAE) as Euclidean point-to-curve distance
  using a densely sampled spline polyline.

Notes:
- For degree=3 (cubic), at least 4 control points are required.
  If n_control_points < degree+1, fit_success will be False.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fitting.common_helpers import (
    densify_polyline,
    get_points_of_all_3_lanes,
    mae_from_errors,
    maxae_from_errors,
    medae_from_errors,
    rmse_from_errors,
)
from fitting.load_data import get_current_data

Point = Tuple[float, float]


# ---------------------------
# Data extraction / preprocessing
# ---------------------------


def lane_points_from_row(row) -> Dict[str, List[Point]]:
    """Extract left/center/right lane point lists from a dataset row."""
    left, center, right = get_points_of_all_3_lanes(row)
    return {
        "left": left or [],
        "center": center or [],
        "right": right or [],
    }


def maybe_densify_points(
    points: List[Point], densify_step_px: Optional[float]
) -> List[Point]:
    """Optionally densify a polyline by linear interpolation."""
    if densify_step_px is None:
        return points
    if not points:
        return []
    return densify_polyline(points, step_px=float(densify_step_px))


# ---------------------------
# Parameterization
# ---------------------------


def chord_length_parameterize(points: List[Point]) -> np.ndarray:
    """
    Assign parameter t in [0,1] to each point using chord-length parameterization.
    Returns array shape (N,).
    """
    if len(points) == 0:
        return np.asarray([], dtype=float)
    if len(points) == 1:
        return np.asarray([0.0], dtype=float)

    pts = np.asarray(points, dtype=float)  # (N,2)
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)  # (N-1,)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, len(points), dtype=float)
    return (cum / total).astype(float)


# ---------------------------
# B-spline math (clamped / open-uniform)
# ---------------------------


def clamped_uniform_knots(n_control_points: int, degree: int) -> np.ndarray:
    """
    Build a clamped (open-uniform) knot vector on [0,1].

    Knot vector length: m + p + 1
    - first p+1 knots = 0
    - last  p+1 knots = 1
    - interior knots uniformly spaced (if any)

    Args:
        n_control_points: m
        degree: p

    Returns:
        knots: (m + p + 1,)
    """
    m = int(n_control_points)
    p = int(degree)
    if m <= 0 or p < 0:
        return np.asarray([], dtype=float)

    # number of interior knots (excluding the clamped ends)
    n_interior = m - p - 1
    if n_interior < 0:
        n_interior = 0

    if n_interior == 0:
        interior = np.asarray([], dtype=float)
    else:
        # interior knots are uniform in (0,1)
        interior = np.linspace(0.0, 1.0, n_interior + 2, dtype=float)[1:-1]

    knots = np.concatenate(
        [
            np.zeros(p + 1, dtype=float),
            interior,
            np.ones(p + 1, dtype=float),
        ]
    )
    return knots


def bspline_basis_matrix(
    t: np.ndarray, n_control_points: int, degree: int, knots: np.ndarray
) -> np.ndarray:
    """
    Compute the B-spline basis matrix A for parameters t.

    A[k,i] = N_{i,degree}(t_k), i=0..m-1

    Vectorized Cox–de Boor recursion in matrix form.

    Args:
        t: (N,) in [0,1]
        n_control_points: m
        degree: p
        knots: (m+p+1,)

    Returns:
        A: (N, m)
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    m = int(n_control_points)
    p = int(degree)

    if t.size == 0 or m <= 0:
        return np.zeros((t.size, m), dtype=float)

    # Degree 0 basis: N_{i,0}(t) = 1 if u_i <= t < u_{i+1}
    # Special-case t==1 -> last basis function = 1
    A0 = np.zeros((t.size, m), dtype=float)
    for i in range(m):
        u0 = knots[i]
        u1 = knots[i + 1]
        mask = (t >= u0) & (t < u1)
        A0[mask, i] = 1.0

    # Ensure t==1 maps to the last basis function (clamped end)
    A0[np.isclose(t, 1.0), :] = 0.0
    A0[np.isclose(t, 1.0), m - 1] = 1.0

    A = A0
    # Cox-de Boor recursion up to degree p
    for d in range(1, p + 1):
        A_next = np.zeros_like(A)
        for i in range(m):
            # left term
            denom1 = knots[i + d] - knots[i]
            if denom1 > 1e-12:
                w1 = (t - knots[i]) / denom1
                A_next[:, i] += w1 * A[:, i]

            # right term uses A[:, i+1]
            if i + 1 < m:
                denom2 = knots[i + d + 1] - knots[i + 1]
                if denom2 > 1e-12:
                    w2 = (knots[i + d + 1] - t) / denom2
                    A_next[:, i] += w2 * A[:, i + 1]

        A = A_next

    return A


def bspline_eval(
    ctrl: np.ndarray, t: np.ndarray, knots: np.ndarray, degree: int
) -> np.ndarray:
    """
    Evaluate a B-spline curve at parameters t using basis matrix * control points.

    Args:
        ctrl: (m,2)
        t: (N,)
        knots: knot vector
        degree: p

    Returns:
        (N,2)
    """
    ctrl = np.asarray(ctrl, dtype=float)
    m = ctrl.shape[0]
    A = bspline_basis_matrix(
        np.asarray(t, dtype=float), n_control_points=m, degree=int(degree), knots=knots
    )
    return A @ ctrl


def fit_bspline_ctrl_points(
    points: List[Point],
    n_control_points: int,
    degree: int = 3,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a clamped B-spline (fixed degree) with `n_control_points` to the given points.

    Returns:
        (ctrl, knots) where:
          ctrl: (m,2)
          knots: (m+p+1,)
        or None if fit is not possible.
    """
    m = int(n_control_points)
    p = int(degree)

    # For degree p, need at least p+1 control points
    if m < p + 1:
        return None
    if len(points) < m:
        return None

    pts = np.asarray(points, dtype=float)  # (N,2)
    t = chord_length_parameterize(points)  # (N,)

    knots = clamped_uniform_knots(m, p)
    A = bspline_basis_matrix(t, n_control_points=m, degree=p, knots=knots)  # (N,m)

    x = pts[:, 0]
    y = pts[:, 1]
    try:
        ctrl_x, *_ = np.linalg.lstsq(A, x, rcond=None)
        ctrl_y, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    ctrl = np.stack([ctrl_x, ctrl_y], axis=1).astype(float)  # (m,2)
    return ctrl, knots


# ---------------------------
# Geometry error: point-to-polyline distance
# ---------------------------


def point_to_polyline_distances(
    points: List[Point], polyline: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean distance from each point to the closest segment of a polyline.

    Args:
        points: list of (x,y)
        polyline: (M,2) sampled curve points in order

    Returns:
        distances: (N,) float distances in px
    """
    if len(points) == 0 or polyline is None or len(polyline) < 2:
        return np.asarray([], dtype=float)

    P = np.asarray(points, dtype=float)  # (N,2)
    A = polyline[:-1]  # (M-1,2)
    B = polyline[1:]  # (M-1,2)
    AB = B - A  # (M-1,2)
    AB2 = np.sum(AB * AB, axis=1)  # (M-1,)

    out = np.empty((P.shape[0],), dtype=float)
    for k, p in enumerate(P):
        AP = p - A
        tt = np.zeros((A.shape[0],), dtype=float)
        mask = AB2 > 1e-12
        tt[mask] = np.sum(AP[mask] * AB[mask], axis=1) / AB2[mask]
        tt = np.clip(tt, 0.0, 1.0)
        proj = A + (AB * tt[:, None])
        d2 = np.sum((proj - p) ** 2, axis=1)
        out[k] = math.sqrt(float(np.min(d2)))
    return out


def build_output_path(densify_step_px: Optional[float]) -> str:
    """Build output path (similar naming to bezier results)."""
    if densify_step_px is None:
        return "artifacts/results_bspline_raw.csv"
    step = float(densify_step_px)
    step_tag = str(step).replace(".", "p")
    return f"artifacts/results_bspline_densified_step{step_tag}px.csv"


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    """Execute with arguments."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--densify-step-px",
        type=float,
        default=None,
        help="If set, densify lane polylines to ~this point spacing (pixels) before fitting.",
    )
    p.add_argument(
        "--n-control-points",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="Number of B-spline control points to fit (default: 3 4 5 6).",
    )
    p.add_argument(
        "--degree",
        type=int,
        default=3,
        help="B-spline degree (default: 3). For perfect comparability keep this fixed.",
    )
    p.add_argument(
        "--curve-samples",
        type=int,
        default=400,
        help="Number of samples along the fitted spline for distance-based errors (default: 400).",
    )
    args = p.parse_args()

    densify_step_px: Optional[float] = args.densify_step_px
    n_control_points_list: List[int] = list(args.n_control_points)
    degree: int = int(args.degree)
    curve_samples: int = int(args.curve_samples)

    data = get_current_data()
    results: List[Dict[str, Any]] = []

    variant = (
        "raw" if densify_step_px is None else f"densified_step{densify_step_px:g}px"
    )

    max_ctrl = max(max(n_control_points_list), 6)
    max_ctrl = min(max_ctrl, 12)

    for _, row in data.iterrows():
        meta = {
            "task_name": row.get("task_name", ""),
            "image_id": row.get("image_id", ""),
            "image_path": row.get("image_path", ""),
        }

        lanes = lane_points_from_row(row)

        for lane_name, pts_raw in lanes.items():
            n_raw = len(pts_raw)
            pts_used = maybe_densify_points(pts_raw, densify_step_px)
            n_used = len(pts_used)

            for m in n_control_points_list:
                m = int(m)
                min_required_points = int(m)  # keep consistent with other scripts
                # NOTE: additional spline requirement: m >= degree+1 (otherwise fit impossible)
                ctrl_and_knots = fit_bspline_ctrl_points(
                    pts_used, n_control_points=m, degree=degree
                )

                rec: Dict[str, Any] = {
                    **meta,
                    "lane": lane_name,
                    "method": "bspline",
                    "variant": variant,
                    "n_control_points": int(m),
                    "degree": int(degree),
                    "clamped": True,
                    "densify_step_px": float("nan")
                    if densify_step_px is None
                    else float(densify_step_px),
                    "min_required_points": int(min_required_points),
                    "n_points_in_raw": int(n_raw),
                    "n_points_in_used": int(n_used),
                    "lane_present": n_raw > 0,
                    "enough_points": n_used >= min_required_points,
                    "fit_success": ctrl_and_knots is not None,
                    "rmse_dist_px_used": float("nan"),
                    "mae_dist_px_used": float("nan"),
                    "medae_dist_px_used": float("nan"),
                    "maxae_dist_px_used": float("nan"),
                    "rmse_dist_px_gt": float("nan"),
                    "mae_dist_px_gt": float("nan"),
                    "medae_dist_px_gt": float("nan"),
                    "maxae_dist_px_gt": float("nan"),
                }

                for i in range(max_ctrl):
                    rec[f"ctrl_p{i}_x"] = float("nan")
                    rec[f"ctrl_p{i}_y"] = float("nan")

                if ctrl_and_knots is not None:
                    ctrl, knots = ctrl_and_knots

                    for i in range(min(ctrl.shape[0], max_ctrl)):
                        rec[f"ctrl_p{i}_x"] = float(ctrl[i, 0])
                        rec[f"ctrl_p{i}_y"] = float(ctrl[i, 1])

                    ts = np.linspace(0.0, 1.0, num=curve_samples, dtype=float)
                    curve = bspline_eval(ctrl, ts, knots=knots, degree=degree)

                    dist_used = point_to_polyline_distances(pts_used, curve)
                    rec["rmse_dist_px_used"] = rmse_from_errors(dist_used)
                    rec["mae_dist_px_used"] = mae_from_errors(dist_used)
                    rec["medae_dist_px_used"] = medae_from_errors(dist_used)
                    rec["maxae_dist_px_used"] = maxae_from_errors(dist_used)

                    dist_gt = point_to_polyline_distances(pts_raw, curve)
                    rec["rmse_dist_px_gt"] = rmse_from_errors(dist_gt)
                    rec["mae_dist_px_gt"] = mae_from_errors(dist_gt)
                    rec["medae_dist_px_gt"] = medae_from_errors(dist_gt)
                    rec["maxae_dist_px_gt"] = maxae_from_errors(dist_gt)

                results.append(rec)

    df_results = pd.DataFrame(results)

    print(df_results.head(10))
    print("\nFit success rate:")
    print(df_results.groupby(["lane", "n_control_points"])["fit_success"].mean())

    out_csv = build_output_path(densify_step_px)
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

# Usage:
#
# RAW
# python src/fit_bsplines.py
#
# Densified
# python src/fit_bsplines.py --densify-step-px 10
#
# Control points sweep (still cubic degree=3):
# python src/fit_bsplines.py --densify-step-px 10 --n-control-points 4 5 6 7 8 9 10 --degree 3
