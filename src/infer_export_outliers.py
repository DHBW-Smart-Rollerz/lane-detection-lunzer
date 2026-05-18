from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import BSpline  # type: ignore
except Exception:
    BSpline = None


LANE_ORDER = ["left", "center", "right"]
LANE_TO_IDX = {n: i for i, n in enumerate(LANE_ORDER)}


# ----------------------------
# Curve evaluation
# ----------------------------
def _bernstein_matrix(t: np.ndarray, degree: int) -> np.ndarray:
    import math

    t = np.asarray(t, dtype=float).reshape(-1)
    n = int(degree)
    A = np.empty((t.shape[0], n + 1), dtype=float)
    one_minus = 1.0 - t
    for i in range(n + 1):
        c = math.comb(n, i)
        A[:, i] = c * (t**i) * (one_minus ** (n - i))
    return A


def bezier_eval(ctrl: np.ndarray, n_samples: int = 400) -> np.ndarray:
    """
    Evaluate Bezier curve from control points.

    Returns sampled points used for visualizing
    predicted lane geometry.
    """
    ctrl = np.asarray(ctrl, dtype=float)
    m = ctrl.shape[0]
    degree = m - 1
    t = np.linspace(0.0, 1.0, n_samples, dtype=float)
    A = _bernstein_matrix(t, degree)
    return A @ ctrl


def _clamped_uniform_knot_vector(n_ctrl: int, degree: int) -> np.ndarray:
    m = int(n_ctrl)
    p = int(degree)
    n_knots = m + p + 1
    n_internal = n_knots - 2 * (p + 1)
    if n_internal < 0:
        raise ValueError("n_ctrl too small for chosen degree")
    if n_internal == 0:
        internal = np.asarray([], dtype=float)
    else:
        internal = np.linspace(0.0, 1.0, n_internal + 2, dtype=float)[1:-1]
    return np.concatenate([np.zeros(p + 1), internal, np.ones(p + 1)])


def bspline_eval(ctrl: np.ndarray, degree: int = 3, n_samples: int = 400) -> np.ndarray:
    """
    Evaluate cubic B-spline curve from control points.

    Generates sampled curve points for visualization
    or geometric error analysis.
    """
    if BSpline is None:
        raise RuntimeError("scipy not available; cannot evaluate B-splines.")
    ctrl = np.asarray(ctrl, dtype=float)
    knots = _clamped_uniform_knot_vector(ctrl.shape[0], degree)
    bx = BSpline(knots, ctrl[:, 0], degree, extrapolate=True)
    by = BSpline(knots, ctrl[:, 1], degree, extrapolate=True)
    t = np.linspace(0.0, 1.0, n_samples, dtype=float)
    return np.stack([bx(t), by(t)], axis=1)


def eval_curve(method: str, ctrl_px: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Evaluate curve depending on selected method.

    Supports Bezier and B-spline curve evaluation.
    """
    if method == "bezier":
        return bezier_eval(ctrl_px, n_samples=n_samples)
    if method == "bspline":
        return bspline_eval(ctrl_px, degree=3, n_samples=n_samples)
    raise ValueError(method)


def denorm_ctrl(ctrl_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Convert normalized control points to pixel coordinates.

    Scales control points according to image width and height.
    """
    c = np.asarray(ctrl_norm, dtype=float).copy()
    c[..., 0] *= float(w)
    c[..., 1] *= float(h)
    return c


# ----------------------------
# Drawing helpers
# ----------------------------
def draw_poly_curve(
    img: np.ndarray, xs: List[float], ys: List[float], color, thickness=3, clip=True
) -> np.ndarray:
    """
    Draw a polyline curve on an image.

    Used to render sampled lane curves for visualization.
    """
    h, w = img.shape[:2]
    pts = []
    for x, y in zip(xs, ys):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if clip:
            if 0 <= xi < w and 0 <= yi < h:
                pts.append((xi, yi))
        else:
            pts.append((xi, yi))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
    return img


def annotate(img: np.ndarray, text: str, row: int = 1) -> None:
    """
    Overlay annotation text on an image.

    Displays information such as RMSE values,
    prediction confidence and lane type.
    """
    y = 28 + (row - 1) * 30
    cv2.putText(
        img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_ctrl_polygon(
    img: np.ndarray, ctrl_px: np.ndarray, color=(0, 165, 255)
) -> None:
    """
    Draw control polygon connecting predicted control points.

    Used to visualize the underlying control structure
    of predicted lane curves.
    """
    h, w = img.shape[:2]
    pts = []
    for x, y in ctrl_px:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            pts.append((xi, yi))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cv2.line(img, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (x, y), 4, color, -1, lineType=cv2.LINE_AA)


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(int).astype(bool)
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["1", "true", "t", "yes", "y"])


# ----------------------------
# Selection logic
# ----------------------------
def select_rows(df: pd.DataFrame, mode: str, k: int, mid_frac: float) -> pd.DataFrame:
    """
    Select subset of rows for qualitative visualization.

    Filters rows based on selection mode such as
    best predictions, worst predictions, false positives
    or false negatives.
    """
    df = df.copy()
    df["gt_exists"] = _to_bool_series(df["gt_exists"])
    df["pred_exists"] = _to_bool_series(df["pred_exists"])

    if mode in ["worst", "best", "mid"]:
        ok = df[
            (df["gt_exists"] == True) & (df["pred_exists"] == True)
        ].copy()  # noqa: E712
        ok = ok[np.isfinite(ok["rmse_sym_px"].to_numpy(float))]
        if len(ok) == 0:
            return ok

        ok = ok.sort_values("rmse_sym_px", ascending=True)

        if mode == "best":
            return ok.head(k)

        if mode == "worst":
            return ok.tail(k).sort_values("rmse_sym_px", ascending=False)

        # mid: take around median quantile band
        vals = ok["rmse_sym_px"].to_numpy(float)
        q_lo = np.quantile(vals, 0.5 - mid_frac)
        q_hi = np.quantile(vals, 0.5 + mid_frac)
        mid = ok[(ok["rmse_sym_px"] >= q_lo) & (ok["rmse_sym_px"] <= q_hi)].copy()
        # if too many, sample evenly by sorting then taking head
        mid = mid.sort_values("rmse_sym_px", ascending=True).head(k)
        return mid

    if mode == "fp":
        # pred exists, gt not
        fp = df[
            (df["pred_exists"] == True) & (df["gt_exists"] == False)
        ].copy()  # noqa: E712
        # Sort by confidence if available
        if "pred_prob" in fp.columns:
            fp = fp.sort_values("pred_prob", ascending=False)
        return fp.head(k)

    if mode == "fn":
        # gt exists, pred not
        fn = df[
            (df["gt_exists"] == True) & (df["pred_exists"] == False)
        ].copy()  # noqa: E712
        if "pred_prob" in fn.columns:
            fn = fn.sort_values("pred_prob", ascending=True)
        return fn.head(k)

    raise ValueError(f"Unknown mode: {mode}")


# ----------------------------
# Main
# ----------------------------
def main():
    """
    Export qualitative visualizations of selected predictions.

    Loads inference results and renders images highlighting
    interesting examples such as worst errors or false detections.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infer-dir",
        required=True,
        help="folder containing preds.npz and metrics_lane.csv",
    )
    ap.add_argument("--method", choices=["bezier", "bspline"], required=True)
    ap.add_argument(
        "--mode", choices=["worst", "best", "mid", "fp", "fn"], default="worst"
    )
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--lane", choices=["all", "left", "center", "right"], default="all")
    ap.add_argument("--curve-samples", type=int, default=400)
    ap.add_argument(
        "--draw-mode",
        choices=["orig", "train"],
        default="orig",
        help="Draw on original image or on training canvas size stored in preds.npz",
    )
    ap.add_argument(
        "--mid-frac",
        type=float,
        default=0.05,
        help="For mode=mid: +/- band around median quantile",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output directory (default: <infer-dir>/outliers/<mode>)",
    )
    args = ap.parse_args()

    infer_dir = Path(args.infer_dir)
    preds_path = infer_dir / "preds.npz"
    metrics_path = infer_dir / "metrics_lane.csv"
    if not preds_path.exists():
        raise FileNotFoundError(preds_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path)
    if args.lane != "all":
        df = df[df["lane"] == args.lane].copy()

    # pick rows
    sel = select_rows(df, mode=args.mode, k=int(args.k), mid_frac=float(args.mid_frac))
    if len(sel) == 0:
        print(f"[vis_outliers] no rows selected for mode={args.mode}")
        return

    z = np.load(preds_path, allow_pickle=True)
    indices = z["indices"].astype(int)  # (K,)
    image_paths = z["image_path"].astype(str)  # (K,)
    gt_present = z["gt_present"].astype(np.float32)  # (K,3)
    gt_ctrl = z["gt_ctrl"].astype(np.float32)  # (K,3,m,2)
    pred_prob = z["pred_prob"].astype(np.float32)  # (K,3)
    pred_ctrl = z["pred_ctrl"].astype(np.float32)  # (K,3,m,2)
    img_w_train = int(z["img_w"])
    img_h_train = int(z["img_h"])
    m = int(z["m"])

    # Map original image_id -> local index in preds arrays
    id_to_k: Dict[int, int] = {int(iid): kk for kk, iid in enumerate(indices)}

    out_dir = Path(args.out) if args.out else (infer_dir / "outliers" / args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    col_gt = (0, 255, 0)
    col_pr = (0, 255, 255)
    col_ctrl = (0, 165, 255)

    # Save a CSV of what we exported (nice for paper)
    sel_out_csv = out_dir / "selected_rows.csv"
    sel.to_csv(sel_out_csv, index=False)

    for rank, row in enumerate(sel.itertuples(index=False), start=1):
        image_id = int(getattr(row, "image_id"))
        lane = str(getattr(row, "lane"))
        lane_idx = LANE_TO_IDX.get(lane, None)
        if lane_idx is None:
            continue

        kk = id_to_k.get(image_id, None)
        if kk is None:
            # Shouldn't happen if infer_export_nn used same ids
            continue

        img_path = str(getattr(row, "image_path"))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] could not load: {img_path}")
            continue

        # choose drawing canvas
        if args.draw_mode == "orig":
            draw_img = img.copy()
            H, W = draw_img.shape[:2]
        else:
            draw_img = cv2.resize(
                img, (img_w_train, img_h_train), interpolation=cv2.INTER_AREA
            )
            H, W = draw_img.shape[:2]

        # denorm ctrl points into chosen pixel space
        gt_ctrl_px = denorm_ctrl(gt_ctrl[kk], W, H)  # (3,m,2)
        pr_ctrl_px = denorm_ctrl(pred_ctrl[kk], W, H)  # (3,m,2)

        # draw only the chosen lane (cleaner for outliers)
        gt_ex = bool(gt_present[kk, lane_idx] >= 0.5)
        pr_ex = bool(pred_prob[kk, lane_idx] >= 0.5)
        rmse = float(getattr(row, "rmse_sym_px", float("nan")))
        prob = float(pred_prob[kk, lane_idx])

        # curves
        if gt_ex:
            curve_gt = eval_curve(
                args.method, gt_ctrl_px[lane_idx], n_samples=int(args.curve_samples)
            )
            draw_poly_curve(
                draw_img,
                curve_gt[:, 0].tolist(),
                curve_gt[:, 1].tolist(),
                color=col_gt,
                thickness=3,
                clip=True,
            )

        if pr_ex:
            curve_pr = eval_curve(
                args.method, pr_ctrl_px[lane_idx], n_samples=int(args.curve_samples)
            )
            draw_poly_curve(
                draw_img,
                curve_pr[:, 0].tolist(),
                curve_pr[:, 1].tolist(),
                color=col_pr,
                thickness=4,
                clip=True,
            )
            draw_ctrl_polygon(draw_img, pr_ctrl_px[lane_idx], color=col_ctrl)

        # annotate
        base = os.path.basename(img_path)
        annotate(
            draw_img,
            f"{rank:03d}/{len(sel)}  mode={args.mode}  {args.method}  m={m}  lane={lane}  draw={args.draw_mode}",
            row=1,
        )
        annotate(draw_img, f"{base}", row=2)
        annotate(
            draw_img,
            f"gt={int(gt_ex)} pred={int(pr_ex)}  prob={prob:.3f}  rmse_sym_px={rmse:.3f}",
            row=3,
        )

        out_name = (
            f"{rank:03d}_id{image_id}_{lane}_rmse{rmse:.2f}_{Path(base).stem}.png"
        )
        cv2.imwrite(str(out_dir / out_name), draw_img)

    print(f"[vis_outliers] wrote {len(sel)} images to: {out_dir.resolve()}")
    print(f"[vis_outliers] wrote selection CSV: {sel_out_csv.resolve()}")


if __name__ == "__main__":
    main()


"""
Examples:

# Worst 20 TP-lanes (by rmse_sym_px), draw on original resolution:
python src/infer_export_outliers.py \
  --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
  --method bspline --mode worst --k 20 --draw-mode orig

# Best 20 TP-lanes:
python src/infer_export_outliers.py \
  --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
  --method bspline --mode best --k 20 --draw-mode orig

# Mid bucket around median (+/-5% quantile band):
python src/infer_export_outliers.py \
  --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
  --method bspline --mode mid --mid-frac 0.05 --k 20 --draw-mode orig

# False positives (pred=1, gt=0), sorted by highest confidence:
python src/infer_export_outliers.py \
  --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
  --method bspline --mode fp --k 20 --draw-mode orig

# False negatives (gt=1, pred=0), sorted by lowest confidence:
python src/infer_export_outliers.py \
  --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
  --method bspline --mode fn --k 20 --draw-mode orig

# Only one lane:
python src/infer_export_outliers.py \
  --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
  --method bspline --mode worst --k 20 --lane center --draw-mode orig
"""
