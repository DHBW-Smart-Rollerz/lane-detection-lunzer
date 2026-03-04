from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from fitting.visualize import draw_poly_curve, load_image, send_frame_to_server
from train_lane_ctrlpoints import LaneNet

try:
    from scipy.interpolate import BSpline  # type: ignore
except Exception:
    BSpline = None


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
    Evaluate a Bezier curve from predicted control points.

    Returns sampled points along the Bezier curve for visualization.
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

    Produces sampled curve points suitable for visualization
    of predicted lane geometry.
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


def _eval_curve(method: str, ctrl_px: np.ndarray, samples: int) -> np.ndarray:
    if method == "bezier":
        return bezier_eval(ctrl_px, n_samples=samples)
    if method == "bspline":
        return bspline_eval(ctrl_px, degree=3, n_samples=samples)
    raise ValueError(method)


# ----------------------------
# Drawing helpers
# ----------------------------
def _annotate(img, text: str, row=1):
    cv2.putText(
        img,
        text,
        (10, 28 + ((row - 1) * 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (10, 28 + ((row - 1) * 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _draw_ctrl_points(
    img: np.ndarray, ctrl: np.ndarray, color=(0, 165, 255)
) -> np.ndarray:
    h, w = img.shape[:2]
    pts = []
    for x, y in ctrl:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h:
            pts.append((xi, yi))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cv2.line(img, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
    return img


def _denorm_ctrl(ctrl_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    ctrl = np.asarray(ctrl_norm, dtype=float).copy()
    ctrl[..., 0] *= float(w)
    ctrl[..., 1] *= float(h)
    return ctrl


# ----------------------------
# Main
# ----------------------------
def main():
    """
    Visualize predicted and ground truth lane curves.

    Runs inference on a subset of images and overlays predicted
    curves, ground truth curves and control polygons for
    qualitative inspection of model performance.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["bezier", "bspline"], required=True)
    p.add_argument("--npz", default=None)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curve-samples", type=int, default=400)
    p.add_argument("--out", default="artifacts/infer_vis")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--interval", type=float, default=0.6)

    p.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="val",
        help="Which split to visualize (requires ckpt to contain indices).",
    )
    p.add_argument(
        "--draw-mode",
        choices=["train", "orig"],
        default="orig",
        help="Draw on resized training canvas or original image.",
    )

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    npz_path = str(args.npz or ckpt.get("npz"))
    if not npz_path:
        raise RuntimeError("No NPZ path resolved (pass --npz or ensure ckpt['npz']).")

    img_w_train = int(ckpt.get("img_w") or ckpt.get("cfg", {}).get("img_w") or 640)
    img_h_train = int(ckpt.get("img_h") or ckpt.get("cfg", {}).get("img_h") or 384)
    m = int(ckpt.get("m") or ckpt.get("cfg", {}).get("m") or 6)
    ctrl_margin = float(
        ckpt.get("ctrl_margin", ckpt.get("cfg", {}).get("ctrl_margin", 0.25))
    )

    data = np.load(npz_path, allow_pickle=True)
    image_path = data["image_path"].astype(str)
    lane_present = data["lane_present"].astype(np.float32)  # (N,3)
    ctrl_points = data["ctrl_points"].astype(np.float32)  # (N,3,m,2)

    N = len(image_path)

    # Select indices by split (if available)
    if args.split != "all":
        key = "val_indices" if args.split == "val" else "train_indices"
        idxs = list(ckpt.get(key, []))
        if not idxs:
            print(f"[warn] ckpt has no '{key}'. Falling back to all.")
            idxs = list(range(N))
    else:
        idxs = list(range(N))

    random.shuffle(idxs)
    idxs = idxs[: min(args.n, len(idxs))]

    model = LaneNet(m=m, ctrl_margin=ctrl_margin, pretrained=False).to(device)
    msd = ckpt.get("model_state", ckpt)
    model.load_state_dict(msd, strict=True)
    model.eval()

    out_dir = (
        Path(args.out)
        / Path(args.ckpt).parent.name
        / f"{args.method}_{args.split}_{args.draw_mode}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    col_gt = (0, 255, 0)
    col_pred = (0, 255, 255)
    col_ctrl = (0, 165, 255)

    for j, idx in enumerate(idxs, start=1):
        path = str(image_path[idx])

        img_orig = load_image(path)
        if img_orig is None:
            print(f"[skip] could not load {path}")
            continue

        # canvas to draw on
        if args.draw_mode == "orig":
            draw_img = img_orig.copy()
            draw_h, draw_w = draw_img.shape[:2]
        else:
            draw_img = cv2.resize(
                img_orig, (img_w_train, img_h_train), interpolation=cv2.INTER_AREA
            )
            draw_h, draw_w = draw_img.shape[:2]

        # model input always training size
        img_in = cv2.resize(
            img_orig, (img_w_train, img_h_train), interpolation=cv2.INTER_AREA
        )
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            exist_logits, ctrl_pred = model(x)
            exist_prob = torch.sigmoid(exist_logits)[0].detach().cpu().numpy()  # (3,)
            ctrl_pred = (
                ctrl_pred[0].detach().cpu().numpy()
            )  # (3,m,2) normalized (may be <0 or >1)

        gt_present = lane_present[idx]
        gt_ctrl_norm = ctrl_points[idx]

        pred_ctrl_px = _denorm_ctrl(ctrl_pred, draw_w, draw_h)
        gt_ctrl_px = _denorm_ctrl(gt_ctrl_norm, draw_w, draw_h)

        lanes = ["left", "center", "right"]
        for li, _ in enumerate(lanes):
            pred_exists = exist_prob[li] >= 0.5
            gt_exists = gt_present[li] >= 0.5

            if gt_exists:
                curve_gt = _eval_curve(
                    args.method, gt_ctrl_px[li], samples=args.curve_samples
                )
                draw_img = draw_poly_curve(
                    draw_img,
                    curve_gt[:, 0].tolist(),
                    curve_gt[:, 1].tolist(),
                    color=col_gt,
                    thickness=3,
                    clip=True,
                )

            if pred_exists:
                curve_pr = _eval_curve(
                    args.method, pred_ctrl_px[li], samples=args.curve_samples
                )
                draw_img = draw_poly_curve(
                    draw_img,
                    curve_pr[:, 0].tolist(),
                    curve_pr[:, 1].tolist(),
                    color=col_pred,
                    thickness=4,
                    clip=True,
                )
                draw_img = _draw_ctrl_points(draw_img, pred_ctrl_px[li], color=col_ctrl)

        base = os.path.basename(path)
        _annotate(
            draw_img,
            f"{j:03d}/{len(idxs)}  {args.method}  m={m}  split={args.split}  draw={args.draw_mode}",
            row=1,
        )
        _annotate(draw_img, f"{base}", row=2)
        _annotate(
            draw_img,
            f"GT={gt_present.astype(int).tolist()}  pred_prob={[round(float(p), 2) for p in exist_prob]}  margin={ctrl_margin}",
            row=3,
        )

        out_path = out_dir / f"{j:03d}_{Path(base).stem}.png"
        cv2.imwrite(str(out_path), draw_img)

        if args.stream:
            send_frame_to_server(draw_img)
            import time

            time.sleep(float(args.interval))

    print(f"[done] wrote {len(idxs)} images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


# Verwendung
# python src/infer_vis_ctrl.py --ckpt artifacts/train_bezier_m6/best.pt --method bezier --split val --draw-mode orig --n 120
# python src/infer_vis_ctrl.py --ckpt artifacts/train_bezier_m6/best.pt --method bezier --split val --draw-mode orig --n 300 --stream --interval 3
# python src/infer_vis_ctrl.py --ckpt artifacts/train_bspline_m6/best.pt --method bspline --split val --draw-mode orig --n 300 --stream --interval 3
