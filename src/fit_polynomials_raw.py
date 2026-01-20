import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fitting.common_helpers import (
    densify_polyline,
    get_points_of_all_3_lanes,
    mae_from_errors,
    maxae_from_errors,
    medae_from_errors,
    rmse_from_errors,
    x_errors_x_of_y,
)
from fitting.fit_polynomial import fit_poly_x_of_y
from fitting.load_data import get_current_data

Point = Tuple[float, float]


def lane_points_from_row(row) -> Dict[str, List[Point]]:
    """
    Extract all lane point lists from a dataset row.

    Args:
        row:
            Pandas Series or dict-like object containing lane annotations.

    Returns:
        dict[str, list[tuple[float, float]]]:
            Mapping of lane name to point list. Missing lanes are returned as empty lists.
    """
    left, center, right = get_points_of_all_3_lanes(row)
    return {
        "left": left or [],
        "center": center or [],
        "right": right or [],
    }


def maybe_densify_points(
    points: List[Point], densify_step_px: Optional[float]
) -> List[Point]:
    """
    Optionally densify a polyline by linear interpolation.

    Args:
        points:
            Original (x, y) points.
        densify_step_px:
            If provided, points are densified with approximately this spacing (in pixels).
            If None, points are returned unchanged.

    Returns:
        list[tuple[float, float]]:
            Points used for fitting (raw or densified).
    """
    if densify_step_px is None:
        return points
    if not points:
        return []
    return densify_polyline(points, step_px=float(densify_step_px))


def build_output_path(densify_step_px: Optional[float]) -> str:
    """
    Build the output CSV path based on the chosen preprocessing variant.

    Args:
        densify_step_px:
            Densify step size in pixels, or None for raw mode.

    Returns:
        str:
            Output path for the results CSV.
    """
    if densify_step_px is None:
        return "artifacts/results_poly_raw.csv"
    step = float(densify_step_px)
    step_tag = str(step).replace(".", "p")
    return f"artifacts/results_poly_densified_step{step_tag}px.csv"


def main() -> None:
    """Run polynomial fitting over the dataset and write results to a CSV (single source of truth)."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--densify-step-px",
        type=float,
        default=None,
        help="If set, densify lane polylines to ~this point spacing (pixels) before fitting.",
    )
    p.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Polynomial degrees to fit (default: 2 3 4).",
    )
    args = p.parse_args()

    densify_step_px: Optional[float] = args.densify_step_px
    degrees: List[int] = list(args.degrees)

    data = get_current_data()
    results: List[Dict[str, Any]] = []

    variant = (
        "raw" if densify_step_px is None else f"densified_step{densify_step_px:g}px"
    )

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

            for deg in degrees:
                min_required_points = deg + 1
                fit = fit_poly_x_of_y(pts_used, deg)

                rec: Dict[str, Any] = {
                    **meta,
                    "lane": lane_name,
                    "method": "poly",
                    "variant": variant,
                    "degree": deg,
                    "densify_step_px": float("nan")
                    if densify_step_px is None
                    else float(densify_step_px),
                    "min_required_points": min_required_points,
                    "n_points_in_raw": n_raw,
                    "n_points_in_used": n_used,
                    "lane_present": n_raw > 0,
                    "enough_points": n_used >= min_required_points,
                    "fit_success": fit is not None,
                    "coef_c0": float("nan"),
                    "coef_c1": float("nan"),
                    "coef_c2": float("nan"),
                    "coef_c3": float("nan"),
                    "coef_c4": float("nan"),
                    # errors computed on the points used for fitting (raw or densified)
                    "rmse_x_px_used": float("nan"),
                    "mae_x_px_used": float("nan"),
                    "medae_x_px_used": float("nan"),
                    "maxae_x_px_used": float("nan"),
                    # errors computed only on original ground-truth points (always raw)
                    "rmse_x_px_gt": float("nan"),
                    "mae_x_px_gt": float("nan"),
                    "medae_x_px_gt": float("nan"),
                    "maxae_x_px_gt": float("nan"),
                }

                if fit is not None:
                    for i in range(min(len(fit.poly_std_coeffs), 5)):
                        rec[f"coef_c{i}"] = float(fit.poly_std_coeffs[i])

                    err_used = x_errors_x_of_y(pts_used, fit.poly)
                    rec["rmse_x_px_used"] = rmse_from_errors(err_used)
                    rec["mae_x_px_used"] = mae_from_errors(err_used)
                    rec["medae_x_px_used"] = medae_from_errors(err_used)
                    rec["maxae_x_px_used"] = maxae_from_errors(err_used)

                    err_gt = x_errors_x_of_y(pts_raw, fit.poly)
                    rec["rmse_x_px_gt"] = rmse_from_errors(err_gt)
                    rec["mae_x_px_gt"] = mae_from_errors(err_gt)
                    rec["medae_x_px_gt"] = medae_from_errors(err_gt)
                    rec["maxae_x_px_gt"] = maxae_from_errors(err_gt)

                results.append(rec)

    df_results = pd.DataFrame(results)

    # Quick sanity print
    print(df_results.head(10))
    print("\nFit success rate:")
    print(df_results.groupby(["lane", "degree"])["fit_success"].mean())

    out_csv = build_output_path(densify_step_px)
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()

# Beispiel Nutzung:
# python src/fit_polynomials_raw.py
# python src/fit_polynomials_raw.py --densify-step-px 10
