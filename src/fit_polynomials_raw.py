import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fitting.common_helpers import (
    get_points_of_all_3_lanes,
    mae_from_errors,
    maxae_from_errors,
    medae_from_errors,
    rmse_from_errors,
    x_errors_x_of_y,
)
from fitting.fit_polynomial import fit_poly_x_of_y
from fitting.load_data import get_current_data


def rmse_x_of_y(points: List[Tuple[float, float]], poly) -> float:
    """
    Compute RMSE in x-direction for x = f(y) evaluated at the original points.

    Args:
        points: List of (x, y) points (ground truth).
        poly: A NumPy Polynomial-like object callable as poly(y).

    Returns:
        RMSE in pixels.
    """
    if not points:
        return float("nan")

    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)

    x_hat = poly(ys)
    return float(np.sqrt(np.mean((x_hat - xs) ** 2)))


def lane_points_from_row(row) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract lane points for a specific lane from a dataset row.

    Args:
        row:
            Pandas Series or dict-like object containing lane annotations.
        lane_name:
            Name of the lane ("left", "center", or "right").

    Returns:
        list[tuple[float, float]]:
            List of (x, y) points for the requested lane.
    """
    left, center, right = get_points_of_all_3_lanes(row)
    return {
        "left": left or [],
        "center": center or [],
        "right": right or [],
    }


if __name__ == "__main__":
    data = get_current_data()

    # For now: small sample while developing
    # sample_data = data.sample(n=50, random_state=0)

    degrees = [2, 3, 4]
    results: List[Dict[str, Any]] = []

    for _, row in data.iterrows():
        meta = {
            "task_name": row.get("task_name", ""),
            "image_id": row.get("image_id", ""),
            "image_path": row.get("image_path", ""),
        }

        lanes = lane_points_from_row(row)

        for lane_name, pts in lanes.items():
            n_in = len(pts)

            for deg in degrees:
                min_required_points = deg + 1
                fit = fit_poly_x_of_y(pts, deg)

                rec: Dict[str, Any] = {
                    **meta,
                    "lane": lane_name,
                    "method": "poly",
                    "variant": "raw",
                    "degree": deg,
                    "min_required_points": min_required_points,
                    "n_points_in": n_in,
                    "lane_present": n_in > 0,
                    "enough_points": n_in >= min_required_points,
                    "fit_success": fit is not None,
                    "coef_c0": float("nan"),
                    "coef_c1": float("nan"),
                    "coef_c2": float("nan"),
                    "coef_c3": float("nan"),
                    "coef_c4": float("nan"),
                    "rmse_x_px": float("nan"),  # root mean squared error
                    "mae_x_px": float("nan"),  # mean absolute error
                    "medae_x_px": float("nan"),  # median absolute error
                    "maxae_x_px": float("nan"),  # maximum absolute error
                }

                if fit is not None:
                    # --- coefficients (ascending order: c0..cn) --- Puts in the saved polynomial coefficients into the result output
                    for i in range(min(len(fit.poly_std_coeffs), 5)):
                        rec[f"coef_c{i}"] = float(fit.poly_std_coeffs[i])

                    # --- error metrics ---
                    err = x_errors_x_of_y(pts, fit.poly)
                    rec["rmse_x_px"] = rmse_from_errors(err)
                    rec["mae_x_px"] = mae_from_errors(err)
                    rec["medae_x_px"] = medae_from_errors(err)
                    rec["maxae_x_px"] = maxae_from_errors(err)

                results.append(rec)

    df_results = pd.DataFrame(results)

    # Quick sanity print
    print(df_results.head(10))
    print("\nFit success rate:")
    print(df_results.groupby(["lane", "degree"])["fit_success"].mean())

    # Save for later reporting
    out_csv = "artifacts/results_poly_raw.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
