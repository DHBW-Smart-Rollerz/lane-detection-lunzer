"""
Create compact training ground-truth files from results CSVs (Bezier/B-spline).

Input:
- results CSV from fit_beziers.py or fit_bsplines.py (densified or raw)
  (one row per image + lane + n_control_points)

Output:
- CSV: one row per image with normalized control points (easy to inspect)
- NPZ (optional): training-ready tensors:
    image_path: (N,)
    lane_present: (N,3)
    ctrl_points: (N,3,m,2)

Normalization uses fixed image size (default: 2064x1544).
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LANES = ["left", "center", "right"]


def _ensure_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def _normalize_xy(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    return float(x) / float(w), float(y) / float(h)


def main() -> None:
    """
    Generate training ground truth files from curve fitting results.

    Converts Bezier or B-spline fitting results into normalized
    control point datasets and exports them as CSV and NPZ files
    used for neural network training.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results", required=True, help="Path to results CSV (bezier/bspline)."
    )
    p.add_argument("--out", required=True, help="Output GT CSV path.")
    p.add_argument(
        "--out-npz",
        default=None,
        help="Optional: also write training-ready NPZ to this path.",
    )
    p.add_argument(
        "--n-control-points",
        type=int,
        required=True,
        help="Control point count m to export (e.g. 6 or 8).",
    )
    p.add_argument(
        "--img-w", type=int, default=2064, help="Image width for normalization."
    )
    p.add_argument(
        "--img-h", type=int, default=1544, help="Image height for normalization."
    )
    p.add_argument(
        "--require-fit-success",
        action="store_true",
        help="Only use ctrl points when fit_success==True; else treat lane as missing in GT.",
    )
    p.add_argument(
        "--method",
        default=None,
        help="Optional: filter method column (e.g. 'bezier' or 'bspline') if present.",
    )
    args = p.parse_args()

    m = int(args.n_control_points)
    w = int(args.img_w)
    h = int(args.img_h)

    df = pd.read_csv(args.results)

    # Basic required columns
    required = ["task_name", "image_id", "image_path", "lane", "n_control_points"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column '{c}' in results CSV.")

    # Filter ctrl-count
    df["n_control_points"] = pd.to_numeric(
        df["n_control_points"], errors="coerce"
    ).astype("Int64")
    df = df[df["n_control_points"] == m].copy()

    # Optional method filter
    if args.method is not None and "method" in df.columns:
        df = df[df["method"].astype(str) == str(args.method)].copy()

    # Parse booleans if present
    if "lane_present" in df.columns:
        df["lane_present"] = _ensure_bool_series(df["lane_present"])
    else:
        df["lane_present"] = True

    if "fit_success" in df.columns:
        df["fit_success"] = _ensure_bool_series(df["fit_success"])
    else:
        df["fit_success"] = True

    # Validate ctrl columns exist
    for i in range(m):
        cx = f"ctrl_p{i}_x"
        cy = f"ctrl_p{i}_y"
        if cx not in df.columns or cy not in df.columns:
            raise RuntimeError(
                f"Missing ctrl columns for i={i}: expected '{cx}' and '{cy}'."
            )

    # One row per image (task_name,image_id,image_path)
    key_cols = ["task_name", "image_id", "image_path"]
    images = df[key_cols].drop_duplicates().reset_index(drop=True)

    out_rows: List[Dict[str, object]] = []

    for _, img_row in images.iterrows():
        task = img_row["task_name"]
        img_id = img_row["image_id"]
        path = img_row["image_path"]

        sub = df[
            (df["task_name"] == task)
            & (df["image_id"] == img_id)
            & (df["image_path"] == path)
        ]

        rec: Dict[str, object] = {
            "task_name": task,
            "image_id": img_id,
            "image_path": path,
            "n_control_points": m,
            "img_w": w,
            "img_h": h,
        }

        # defaults
        for lane in LANES:
            rec[f"lane_present_{lane}"] = False
            for i in range(m):
                rec[f"{lane}_p{i}_x"] = float("nan")
                rec[f"{lane}_p{i}_y"] = float("nan")

        for lane in LANES:
            lane_rows = sub[sub["lane"].astype(str) == lane]
            if lane_rows.empty:
                continue

            r0 = lane_rows.iloc[0]
            lane_present = bool(r0.get("lane_present", True))
            fit_success = bool(r0.get("fit_success", True))

            rec[f"lane_present_{lane}"] = bool(lane_present)

            # We only export points if lane is present and (optionally) fit succeeded
            if not lane_present:
                continue
            if args.require_fit_success and not fit_success:
                continue

            for i in range(m):
                x = r0.get(f"ctrl_p{i}_x", np.nan)
                y = r0.get(f"ctrl_p{i}_y", np.nan)
                if x is None or y is None:
                    continue
                try:
                    xf = float(x)
                    yf = float(y)
                except Exception:
                    continue
                if not (math.isfinite(xf) and math.isfinite(yf)):
                    continue
                xn, yn = _normalize_xy(xf, yf, w=w, h=h)
                rec[f"{lane}_p{i}_x"] = float(xn)
                rec[f"{lane}_p{i}_y"] = float(yn)

        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)

    # stable column order for CSV
    cols = ["task_name", "image_id", "image_path", "n_control_points", "img_w", "img_h"]
    for lane in LANES:
        cols.append(f"lane_present_{lane}")
    for lane in LANES:
        for i in range(m):
            cols += [f"{lane}_p{i}_x", f"{lane}_p{i}_y"]
    out_df = out_df[cols]

    # --- Write CSV
    out_df.to_csv(args.out, index=False)
    print(f"[make_gt_ctrlpoints] wrote CSV: {args.out}  ({len(out_df)} rows)")

    # --- Optionally write NPZ
    if args.out_npz:
        N = len(out_df)

        image_path = out_df["image_path"].astype(str).to_numpy()
        task_name = out_df["task_name"].astype(str).to_numpy()
        image_id = out_df["image_id"].astype(str).to_numpy()

        lane_present = np.stack(
            [out_df[f"lane_present_{l}"].astype(bool).to_numpy() for l in LANES],
            axis=1,
        ).astype(np.bool_)

        ctrl = np.full((N, 3, m, 2), np.nan, dtype=np.float32)
        for li, lane in enumerate(LANES):
            for i in range(m):
                ctrl[:, li, i, 0] = out_df[f"{lane}_p{i}_x"].to_numpy(np.float32)
                ctrl[:, li, i, 1] = out_df[f"{lane}_p{i}_y"].to_numpy(np.float32)

        np.savez_compressed(
            args.out_npz,
            image_path=image_path,
            task_name=task_name,
            image_id=image_id,
            lane_present=lane_present,
            ctrl_points=ctrl,
            img_wh=np.asarray([w, h], dtype=np.int32),
            n_control_points=np.asarray([m], dtype=np.int32),
        )
        print(
            f"[make_gt_ctrlpoints] wrote NPZ: {args.out_npz}  ctrl_points={ctrl.shape}"
        )


if __name__ == "__main__":
    main()


##Verwendung nacheinander um alle 4 Dateien zu erstellen (Bspline 6+8 Punkte, Bezier 6+8 Punkte)
"""
python src/generate_newGT_from_results.py \
  --results artifacts/results_bezier_densified_step10p0px.csv \
  --out artifacts/gt_bezier_m6.csv \
  --out-npz artifacts/gt_bezier_m6.npz \
  --n-control-points 6 \
  --method bezier \
  --require-fit-success

python src/generate_newGT_from_results.py \
  --results artifacts/results_bezier_densified_step10p0px.csv \
  --out artifacts/gt_bezier_m8.csv \
  --out-npz artifacts/gt_bezier_m8.npz \
  --n-control-points 8 \
  --method bezier \
  --require-fit-success

python src/generate_newGT_from_results.py \
  --results artifacts/results_bspline_densified_step10p0px.csv \
  --out artifacts/gt_bspline_m6.csv \
  --out-npz artifacts/gt_bspline_m6.npz \
  --n-control-points 6 \
  --method bspline \
  --require-fit-success

python src/generate_newGT_from_results.py \
  --results artifacts/results_bspline_densified_step10p0px.csv \
  --out artifacts/gt_bspline_m8.csv \
  --out-npz artifacts/gt_bspline_m8.npz \
  --n-control-points 8 \
  --method bspline \
  --require-fit-success
"""
