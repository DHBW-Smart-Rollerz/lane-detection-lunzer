from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from fitting.common_helpers import (
    mae_from_errors,
    maxae_from_errors,
    medae_from_errors,
    rmse_from_errors,
)
from train_lane_ctrlpoints import LaneNet

try:
    from scipy.interpolate import BSpline  # type: ignore
except Exception:
    BSpline = None


LANE_ORDER = ["left", "center", "right"]


# ----------------------------
# Curve evaluation
# ----------------------------
def _bernstein_matrix(t: np.ndarray, degree: int) -> np.ndarray:
    import math as pymath

    t = np.asarray(t, dtype=float).reshape(-1)
    n = int(degree)
    A = np.empty((t.shape[0], n + 1), dtype=float)
    one_minus = 1.0 - t
    for i in range(n + 1):
        c = pymath.comb(n, i)
        A[:, i] = c * (t**i) * (one_minus ** (n - i))
    return A


def bezier_eval(ctrl: np.ndarray, n_samples: int = 400) -> np.ndarray:
    """
    Evaluate a Bezier curve from control points.

    Generates sampled points along the Bezier curve defined
    by the given control points using Bernstein polynomials.
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
    Evaluate a cubic B-spline curve from control points.

    Creates a sampled polyline representation of a clamped
    uniform B-spline curve using SciPy's BSpline implementation.
    """
    if BSpline is None:
        raise RuntimeError("scipy not available; cannot evaluate B-splines.")
    ctrl = np.asarray(ctrl, dtype=float)
    m = ctrl.shape[0]
    knots = _clamped_uniform_knot_vector(m, degree)
    bx = BSpline(knots, ctrl[:, 0], degree, extrapolate=True)
    by = BSpline(knots, ctrl[:, 1], degree, extrapolate=True)
    t = np.linspace(0.0, 1.0, n_samples, dtype=float)
    return np.stack([bx(t), by(t)], axis=1)


def eval_curve(method: str, ctrl_px: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Evaluate a curve depending on the chosen representation.

    Dispatch function that evaluates either a Bezier curve
    or B-spline curve based on the selected method.
    """
    if method == "bezier":
        return bezier_eval(ctrl_px, n_samples=n_samples)
    if method == "bspline":
        return bspline_eval(ctrl_px, degree=3, n_samples=n_samples)
    raise ValueError(method)


def denorm_ctrl(ctrl_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Convert normalized control points into pixel coordinates.

    Transforms control points from normalized image coordinates
    ([0,1] range) to pixel space using image width and height.
    """
    c = np.asarray(ctrl_norm, dtype=float).copy()
    c[..., 0] *= float(w)
    c[..., 1] *= float(h)
    return c


# ----------------------------
# Geometry: point-to-polyline distances (same method as your fitting eval)
# ----------------------------
Point = Tuple[float, float]


def point_to_polyline_distances(
    points: List[Point], polyline: np.ndarray
) -> np.ndarray:
    """
    Euclidean distance from each point to closest segment of a polyline.
    points: list of (x,y)
    polyline: (M,2) sampled curve points in order
    returns: (N,) distances in px
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


def sym_curve_metrics(curve_gt: np.ndarray, curve_pr: np.ndarray) -> Dict[str, float]:
    """
    Symmetric curve error using point-to-polyline distances both directions.
    Returns RMSE/MAE/MedAE/MaxAE (px) + debug one-way RMSEs.
    """
    d_pr_to_gt = point_to_polyline_distances(curve_pr.tolist(), curve_gt)
    d_gt_to_pr = point_to_polyline_distances(curve_gt.tolist(), curve_pr)

    if d_pr_to_gt.size == 0 or d_gt_to_pr.size == 0:
        return {
            "rmse_sym_px": float("nan"),
            "mae_sym_px": float("nan"),
            "medae_sym_px": float("nan"),
            "maxae_sym_px": float("nan"),
            "rmse_pr_to_gt_px": float("nan"),
            "rmse_gt_to_pr_px": float("nan"),
        }

    rmse_sym = float(
        np.sqrt(0.5 * (np.mean(d_pr_to_gt**2) + np.mean(d_gt_to_pr**2)))
    )
    mae_sym = float(0.5 * (mae_from_errors(d_pr_to_gt) + mae_from_errors(d_gt_to_pr)))
    medae_sym = float(
        0.5 * (medae_from_errors(d_pr_to_gt) + medae_from_errors(d_gt_to_pr))
    )
    maxae_sym = float(max(maxae_from_errors(d_pr_to_gt), maxae_from_errors(d_gt_to_pr)))

    return {
        "rmse_sym_px": rmse_sym,
        "mae_sym_px": mae_sym,
        "medae_sym_px": medae_sym,
        "maxae_sym_px": maxae_sym,
        "rmse_pr_to_gt_px": rmse_from_errors(d_pr_to_gt),
        "rmse_gt_to_pr_px": rmse_from_errors(d_gt_to_pr),
    }


# ----------------------------
# Main
# ----------------------------
def _instantiate_model(m: int, ctrl_margin: float | None, pretrained: bool) -> Any:
    """
    Be robust against slightly different LaneNet signatures across your iterations.
    Tries: (m, ctrl_margin, pretrained) -> (m, pretrained) -> (m)
    """
    if ctrl_margin is not None:
        try:
            return LaneNet(m=m, ctrl_margin=float(ctrl_margin), pretrained=pretrained)
        except TypeError:
            pass
    try:
        return LaneNet(m=m, pretrained=pretrained)
    except TypeError:
        return LaneNet(m=m)


def main():
    """
    Run neural network inference and export evaluation data.

    Loads a trained model checkpoint, performs inference on a dataset,
    computes geometric curve errors and exports prediction and
    evaluation metrics for later analysis.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt/last.pt")
    ap.add_argument("--method", choices=["bezier", "bspline"], required=True)
    ap.add_argument(
        "--npz", default=None, help="override NPZ used for GT (else ckpt['npz'])"
    )
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--out", default="artifacts/infer", help="output root dir")
    ap.add_argument("--curve-samples", type=int, default=400)
    ap.add_argument("--exist-thr", type=float, default=0.5)
    ap.add_argument(
        "--pretrained",
        action="store_true",
        help="use pretrained backbone (usually False for eval)",
    )
    ap.add_argument(
        "--eval-space",
        choices=["train", "orig"],
        default="orig",
        help="Compute geometric errors in train canvas px or original image px.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    npz_path = str(args.npz or ckpt.get("npz") or ckpt.get("cfg", {}).get("npz") or "")
    if not npz_path:
        raise RuntimeError("No NPZ path resolved (pass --npz or ensure ckpt['npz']).")

    img_w = int(ckpt.get("img_w") or ckpt.get("cfg", {}).get("img_w") or 640)
    img_h = int(ckpt.get("img_h") or ckpt.get("cfg", {}).get("img_h") or 384)
    m = int(ckpt.get("m") or ckpt.get("cfg", {}).get("m") or 6)
    ctrl_margin = ckpt.get("ctrl_margin", ckpt.get("cfg", {}).get("ctrl_margin", None))

    z = np.load(npz_path, allow_pickle=True)
    image_path = z["image_path"].astype(str)
    lane_present = z["lane_present"].astype(np.float32)  # (N,3)
    ctrl_points = z["ctrl_points"].astype(np.float32)  # (N,3,m,2)

    N = len(image_path)

    # indices
    if args.split == "all":
        idxs = np.arange(N, dtype=int)
    else:
        key = "val_indices" if args.split == "val" else "train_indices"
        idxs = np.array(ckpt.get(key, []), dtype=int)
        if idxs.size == 0:
            raise RuntimeError(
                f"Checkpoint has no '{key}'. Cannot guarantee split consistency."
            )

    # model load
    model = _instantiate_model(
        m=m, ctrl_margin=ctrl_margin, pretrained=bool(args.pretrained)
    ).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    out_dir = (
        Path(args.out) / Path(args.ckpt).parent.name / f"{args.method}_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # save preds for later (no need to re-run net)
    pred_prob_all = np.zeros((len(idxs), 3), dtype=np.float32)
    pred_ctrl_all = np.zeros((len(idxs), 3, m, 2), dtype=np.float32)

    rows_lane: List[Dict[str, Any]] = []

    for k, idx in enumerate(idxs):
        p = str(image_path[idx])
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # NN input always training size
        img_in = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        x = x.unsqueeze(0).to(device)

        with torch.inference_mode():
            exist_logits, ctrl_pred = model(x)
            prob = (
                torch.sigmoid(exist_logits)[0].detach().cpu().numpy().astype(np.float32)
            )  # (3,)
            ctrl = ctrl_pred[0].detach().cpu().numpy().astype(np.float32)  # (3,m,2)

        pred_prob_all[k] = prob
        pred_ctrl_all[k] = ctrl

        gt_present = lane_present[idx].astype(np.float32)  # (3,)
        gt_ctrl = ctrl_points[idx].astype(np.float32)  # (3,m,2)

        # evaluate in according to eval_space parameter (standard is original image size, because all the fitting errors are calculated im original image pixels)
        if args.eval_space == "train":
            w, h = img_w, img_h
        else:
            # original size from image
            h, w = img.shape[:2]

        gt_ctrl_px = denorm_ctrl(gt_ctrl, w, h)
        pr_ctrl_px = denorm_ctrl(ctrl, w, h)

        for li, lane in enumerate(LANE_ORDER):
            gt_exists = bool(gt_present[li] >= 0.5)
            pr_exists = bool(prob[li] >= float(args.exist_thr))

            rec: Dict[str, Any] = {
                "image_id": int(idx),
                "image_path": p,
                "lane": lane,
                "gt_exists": gt_exists,
                "pred_exists": pr_exists,
                "pred_prob": float(prob[li]),
                "rmse_sym_px": float("nan"),
                "mae_sym_px": float("nan"),
                "medae_sym_px": float("nan"),
                "maxae_sym_px": float("nan"),
                "rmse_pr_to_gt_px": float("nan"),
                "rmse_gt_to_pr_px": float("nan"),
            }

            # your chosen rule: "existieren UND vorhergesagt wurden"
            if gt_exists and pr_exists:
                curve_gt = eval_curve(
                    args.method, gt_ctrl_px[li], n_samples=int(args.curve_samples)
                )
                curve_pr = eval_curve(
                    args.method, pr_ctrl_px[li], n_samples=int(args.curve_samples)
                )

                met = sym_curve_metrics(curve_gt, curve_pr)
                rec.update(met)

            rows_lane.append(rec)

        if (k + 1) % 200 == 0:
            print(f"[infer] {k+1}/{len(idxs)}")

    # Write frozen inference outputs
    np.savez_compressed(
        out_dir / "preds.npz",
        indices=idxs,
        image_path=image_path[idxs],
        gt_present=lane_present[idxs],
        gt_ctrl=ctrl_points[idxs],
        pred_prob=pred_prob_all,
        pred_ctrl=pred_ctrl_all,
        img_w=img_w,
        img_h=img_h,
        m=m,
        ctrl_margin=ctrl_margin if ctrl_margin is None else float(ctrl_margin),
        method=args.method,
        split=args.split,
        exist_thr=float(args.exist_thr),
        curve_samples=int(args.curve_samples),
        ckpt_path=str(args.ckpt),
        npz_path=str(npz_path),
    )

    df_lane = pd.DataFrame(rows_lane)
    df_lane.to_csv(out_dir / "metrics_lane.csv", index=False)

    # Per-image aggregation (only lanes where gt & pred exist)
    ok = df_lane[
        (df_lane["gt_exists"] == True) & (df_lane["pred_exists"] == True)
    ].copy()  # noqa: E712
    img_agg = ok.groupby(["image_id", "image_path"], as_index=False).agg(
        rmse_sym_px_mean=("rmse_sym_px", "mean"),
        mae_sym_px_mean=("mae_sym_px", "mean"),
        medae_sym_px_mean=("medae_sym_px", "mean"),
        maxae_sym_px_mean=("maxae_sym_px", "max"),
        n_lanes_ok=("lane", "count"),
    )
    img_agg.to_csv(out_dir / "metrics_image.csv", index=False)

    # Tiny text summary (handy for paper)
    vals = ok["rmse_sym_px"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    summary = {
        "model_dir": str(Path(args.ckpt).parent),
        "method": args.method,
        "split": args.split,
        "n_images": int(len(idxs)),
        "n_lanes_ok": int(vals.size),
        "rmse_sym_mean_px": float(np.mean(vals)) if vals.size else None,
        "rmse_sym_median_px": float(np.median(vals)) if vals.size else None,
        "rmse_sym_p90_px": float(np.percentile(vals, 90)) if vals.size else None,
        "curve_samples": int(args.curve_samples),
        "exist_thr": float(args.exist_thr),
        "img_w": img_w,
        "img_h": img_h,
        "m": m,
    }
    (out_dir / "summary.json").write_text(pd.Series(summary).to_json(indent=2))

    print(f"[done] wrote: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


##Verwendung
"""
# 1) Inference + Export (1x pro Modell)
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m6/best.pt  --method bezier  --split val
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m8/best.pt  --method bezier  --split val
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m6/best.pt --method bspline --split val
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m8/best.pt --method bspline --split val

2) Nach Umentscheidung doch noch M4 und M5 zu machen um Trend zu bestätigen:
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m4/best.pt  --method bezier  --split val --eval-space orig
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m5/best.pt  --method bezier  --split val --eval-space orig
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m4/best.pt --method bspline --split val --eval-space orig
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m5/best.pt --method bspline --split val --eval-space orig
"""
