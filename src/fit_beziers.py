"""
Fit Bezier curves to lane annotations and export results to CSV.

This is the Bezier analogue of `fit_polynomials.py`, but the fitted representation
is a set of Bezier control points (parametric Bezier curve in x/y).

We fit a degree-(m-1) Bezier curve with `m` control points:

    B(t) = sum_{i=0..n} P_i * B_i^n(t),   n=m-1,  P_i=(x_i,y_i)

Parameterization:
- We assign t_k to each sample point using chord-length parameterization along
  the (possibly densified) polyline.

Solve:
- Linear least squares separately for x and y:
    A(t) * ctrl_x ≈ x
    A(t) * ctrl_y ≈ y
  where A_{k,i} = Bernstein(n,i)(t_k)

Outputs:
- Per image + lane + (n_control_points) record with ctrl_p{i}_{x|y} columns.
- Also computes simple geometric errors (RMSE/MAE/MedAE/MaxAE) as Euclidean
  point-to-curve distance using a densely sampled Bezier polyline.
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
# Bezier math
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
        # all points identical (or numerically too close)
        return np.linspace(0.0, 1.0, len(points), dtype=float)
    return (cum / total).astype(float)


def bernstein_matrix(t: np.ndarray, degree: int) -> np.ndarray:
    """
    Build Bernstein basis matrix A for degree n.
      A[k,i] = C(n,i) * t_k^i * (1-t_k)^(n-i)
    Shape: (N, n+1).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    n = int(degree)
    N = t.shape[0]
    A = np.empty((N, n + 1), dtype=float)

    one_minus = 1.0 - t
    for i in range(n + 1):
        c = math.comb(n, i)
        A[:, i] = c * (t**i) * (one_minus ** (n - i))
    return A


def bezier_eval(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Evaluate parametric Bezier curve at parameters t.

    Args:
        ctrl: (m,2) control points, m = degree+1
        t: (N,) parameters in [0,1]

    Returns:
        (N,2) points on curve
    """
    ctrl = np.asarray(ctrl, dtype=float)
    m = ctrl.shape[0]
    degree = m - 1
    A = bernstein_matrix(np.asarray(t, dtype=float), degree)  # (N,m)
    return A @ ctrl  # (N,2)


def fit_bezier_ctrl_points(
    points: List[Point], n_control_points: int
) -> Optional[np.ndarray]:
    """
    Fit a Bezier curve with `n_control_points` to the given points.

    Returns:
        ctrl: (n_control_points, 2) or None if fit is not possible.
    """
    m = int(n_control_points)
    if m < 2:
        return None
    if len(points) < m:
        return None

    pts = np.asarray(points, dtype=float)  # (N,2)
    t = chord_length_parameterize(points)  # (N,)
    A = bernstein_matrix(t, degree=m - 1)  # (N,m)

    x = pts[:, 0]
    y = pts[:, 1]
    try:
        ctrl_x, *_ = np.linalg.lstsq(A, x, rcond=None)
        ctrl_y, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    ctrl = np.stack([ctrl_x, ctrl_y], axis=1)  # (m,2)
    return ctrl.astype(float)


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
        t = np.zeros((A.shape[0],), dtype=float)
        mask = AB2 > 1e-12
        t[mask] = np.sum(AP[mask] * AB[mask], axis=1) / AB2[mask]
        t = np.clip(t, 0.0, 1.0)
        proj = A + (AB * t[:, None])
        d2 = np.sum((proj - p) ** 2, axis=1)
        out[k] = math.sqrt(float(np.min(d2)))
    return out


def build_output_path(densify_step_px: Optional[float]) -> str:
    """Build output path (similar naming to polynomial results)."""
    if densify_step_px is None:
        return "artifacts/results_bezier_raw.csv"
    step = float(densify_step_px)
    step_tag = str(step).replace(".", "p")
    return f"artifacts/results_bezier_densified_step{step_tag}px.csv"


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
        help="Number of Bezier control points to fit (default: 3 4 5 6). Degree = n_control_points - 1.",
    )
    p.add_argument(
        "--curve-samples",
        type=int,
        default=400,
        help="Number of samples along the fitted Bezier curve for distance-based errors (default: 400).",
    )
    args = p.parse_args()

    densify_step_px: Optional[float] = args.densify_step_px
    n_control_points_list: List[int] = list(args.n_control_points)
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
                degree = int(m) - 1
                min_required_points = int(m)

                ctrl = fit_bezier_ctrl_points(pts_used, n_control_points=m)

                rec: Dict[str, Any] = {
                    **meta,
                    "lane": lane_name,
                    "method": "bezier",
                    "variant": variant,
                    "n_control_points": int(m),
                    "degree": int(degree),
                    "densify_step_px": float("nan")
                    if densify_step_px is None
                    else float(densify_step_px),
                    "min_required_points": int(min_required_points),
                    "n_points_in_raw": int(n_raw),
                    "n_points_in_used": int(n_used),
                    "lane_present": n_raw > 0,
                    "enough_points": n_used >= min_required_points,
                    "fit_success": ctrl is not None,
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

                if ctrl is not None:
                    for i in range(min(ctrl.shape[0], max_ctrl)):
                        rec[f"ctrl_p{i}_x"] = float(ctrl[i, 0])
                        rec[f"ctrl_p{i}_y"] = float(ctrl[i, 1])

                    ts = np.linspace(0.0, 1.0, num=curve_samples, dtype=float)
                    curve = bezier_eval(ctrl, ts)

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

# Benutzung
#
# RAW
# python src/fit_beziers.py
#
# Densified
# python src/fit_beziers.py --densify-step-px 2
#
# All
# python src/fit_beziers.py --densify-step-px 10 --n-control-points 3 4 5 6
