from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from train_lane_ctrlpoints import LaneNet

try:
    from scipy.interpolate import BSpline  # type: ignore
except Exception:
    BSpline = None


LANE_ORDER = ["left", "center", "right"]


# ----------------------------
# Curve eval
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
    Evaluate Bezier curve from predicted control points.

    Generates sampled curve points for overlay on video frames.
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

    Produces sampled curve points used to render lane predictions.
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


def _denorm_ctrl(ctrl_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    c = np.asarray(ctrl_norm, dtype=float).copy()
    c[..., 0] *= float(w)
    c[..., 1] *= float(h)
    return c


# ----------------------------
# Drawing
# ----------------------------
def draw_poly_curve(
    img: np.ndarray,
    xs: List[float],
    ys: List[float],
    color,
    thickness: int = 3,
    clip: bool = True,
) -> np.ndarray:
    """
    Draw sampled curve points as polyline on image frame.

    Used for visualizing predicted lane curves in videos.
    """
    h, w = img.shape[:2]
    pts = []
    for x, y in zip(xs, ys):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if clip:
            if xi < 0 or yi < 0 or xi >= w or yi >= h:
                continue
        pts.append((xi, yi))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
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


def _annotate(img: np.ndarray, text: str, row: int = 1) -> None:
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


# ----------------------------
# IO helpers
# ----------------------------
def _list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def _instantiate_model(m: int, ctrl_margin: float | None, pretrained: bool) -> LaneNet:
    if ctrl_margin is not None:
        try:
            return LaneNet(m=m, ctrl_margin=float(ctrl_margin), pretrained=pretrained)
        except TypeError:
            pass
    try:
        return LaneNet(m=m, pretrained=pretrained)
    except TypeError:
        return LaneNet(m=m)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    """
    Run neural network inference on video or image frames.

    Processes frames sequentially, predicts lane curves
    and renders them into an output video visualization.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt/last.pt")
    ap.add_argument("--method", choices=["bezier", "bspline"], required=True)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--frames", default=None, help="folder with extracted frames (jpg/png)"
    )
    src.add_argument("--video", default=None, help="input video file")

    ap.add_argument("--out", default="artifacts/demo/demo.mp4")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--curve-samples", type=int, default=400)
    ap.add_argument("--exist-thr", type=float, default=0.5)

    ap.add_argument(
        "--draw-ctrl", action="store_true", help="draw control polygon/points"
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--start", type=int, default=0, help="start frame index (for folder or video)"
    )
    ap.add_argument(
        "--every", type=int, default=1, help="process every Nth frame (speedup)"
    )
    ap.add_argument(
        "--encode",
        choices=["opencv", "ffmpeg"],
        default="ffmpeg",
        help="opencv: write mp4 via OpenCV (less compatible). ffmpeg: write temp AVI and transcode to H.264 yuv420p (recommended).",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 quality for ffmpeg (lower=better, larger=file).",
    )
    ap.add_argument(
        "--preset", default="fast", help="ffmpeg x264 preset (ultrafast..veryslow)."
    )

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    img_w_train = int(ckpt.get("img_w") or ckpt.get("cfg", {}).get("img_w") or 640)
    img_h_train = int(ckpt.get("img_h") or ckpt.get("cfg", {}).get("img_h") or 384)
    m = int(ckpt.get("m") or ckpt.get("cfg", {}).get("m") or 6)
    ctrl_margin = ckpt.get("ctrl_margin", ckpt.get("cfg", {}).get("ctrl_margin", None))

    model = _instantiate_model(m=m, ctrl_margin=ctrl_margin, pretrained=False).to(
        device
    )
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Colors (BGR)
    col_pred = (0, 255, 255)  # yellow
    col_ctrl = (0, 165, 255)  # orange

    writer = None

    tmp_avi_path = None  # will be used if args.encode == "ffmpeg"

    def ensure_writer(w: int, h: int) -> cv2.VideoWriter:
        nonlocal writer, tmp_avi_path

        if writer is not None:
            return writer

        if args.encode == "opencv":
            # This often creates mp4v (MPEG-4 Part 2) -> not always NAS/browser friendly
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (w, h))
        else:
            # Write a temp AVI with MJPG (very robust), then transcode via ffmpeg at the end.
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_avi_path = str(out_path.with_suffix(".tmp.avi"))
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(tmp_avi_path, fourcc, float(args.fps), (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {out_path}")
        return writer

    frame_count = 0

    if args.frames:
        files = _list_images(Path(args.frames))
        if args.start:
            files = files[args.start :]
        if args.every > 1:
            files = files[:: args.every]
        if args.max_frames and args.max_frames > 0:
            files = files[: args.max_frames]

        for i, p in enumerate(files):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue

            H, W = img.shape[:2]
            ensure_writer(W, H)

            # model input always training size
            img_in = cv2.resize(
                img, (img_w_train, img_h_train), interpolation=cv2.INTER_AREA
            )
            img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
            x = x.unsqueeze(0).to(device)

            with torch.inference_mode():
                exist_logits, ctrl_pred = model(x)
                prob = torch.sigmoid(exist_logits)[0].detach().cpu().numpy()  # (3,)
                ctrl = ctrl_pred[0].detach().cpu().numpy()  # (3,m,2)

            # IMPORTANT: draw in ORIGINAL size
            ctrl_px = _denorm_ctrl(ctrl, W, H)

            draw_img = img.copy()
            for li, lane in enumerate(LANE_ORDER):
                if float(prob[li]) < float(args.exist_thr):
                    continue
                curve = _eval_curve(
                    args.method, ctrl_px[li], samples=int(args.curve_samples)
                )
                draw_img = draw_poly_curve(
                    draw_img,
                    curve[:, 0].tolist(),
                    curve[:, 1].tolist(),
                    color=col_pred,
                    thickness=4,
                    clip=True,
                )
                if args.draw_ctrl:
                    draw_img = _draw_ctrl_points(draw_img, ctrl_px[li], color=col_ctrl)

            _annotate(
                draw_img,
                f"{Path(args.ckpt).parent.name}  {args.method} m={m}  thr={args.exist_thr:.2f}",
                row=1,
            )
            _annotate(
                draw_img, f"{p.name}  prob={[round(float(x),2) for x in prob]}", row=2
            )

            writer.write(draw_img)
            frame_count += 1
            if frame_count % 200 == 0:
                print(f"[video] {frame_count} frames")

    else:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        print(f"[video] input frames={total} fps={fps_in}")

        idx = 0
        out_written = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break

            if idx < args.start:
                idx += 1
                continue
            if args.every > 1 and ((idx - args.start) % args.every != 0):
                idx += 1
                continue
            if args.max_frames and out_written >= args.max_frames:
                break

            H, W = img.shape[:2]
            ensure_writer(W, H)

            img_in = cv2.resize(
                img, (img_w_train, img_h_train), interpolation=cv2.INTER_AREA
            )
            img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
            x = x.unsqueeze(0).to(device)

            with torch.inference_mode():
                exist_logits, ctrl_pred = model(x)
                prob = torch.sigmoid(exist_logits)[0].detach().cpu().numpy()
                ctrl = ctrl_pred[0].detach().cpu().numpy()

            ctrl_px = _denorm_ctrl(ctrl, W, H)

            draw_img = img.copy()
            for li in range(3):
                if float(prob[li]) < float(args.exist_thr):
                    continue
                curve = _eval_curve(
                    args.method, ctrl_px[li], samples=int(args.curve_samples)
                )
                draw_img = draw_poly_curve(
                    draw_img,
                    curve[:, 0].tolist(),
                    curve[:, 1].tolist(),
                    color=col_pred,
                    thickness=4,
                    clip=True,
                )
                if args.draw_ctrl:
                    draw_img = _draw_ctrl_points(draw_img, ctrl_px[li], color=col_ctrl)

            _annotate(
                draw_img,
                f"{Path(args.ckpt).parent.name}  {args.method} m={m}  thr={args.exist_thr:.2f}",
                row=1,
            )
            _annotate(
                draw_img,
                f"frame={idx}  prob={[round(float(x),2) for x in prob]}",
                row=2,
            )

            writer.write(draw_img)
            out_written += 1
            idx += 1

            if out_written % 200 == 0:
                print(f"[video] wrote {out_written} frames")

        cap.release()

    if writer is not None:
        writer.release()

    # If requested: transcode to H.264 (MP4) for maximum compatibility (Synology/Browser)
    if args.encode == "ffmpeg":
        if not tmp_avi_path or not os.path.exists(tmp_avi_path):
            raise RuntimeError("Temporary AVI was not created; cannot transcode.")

        # ffmpeg -> H.264 + yuv420p + faststart
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tmp_avi_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(int(args.crf)),
            "-preset",
            str(args.preset),
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        print("[ffmpeg] " + " ".join(cmd))
        subprocess.run(cmd, check=True)

        # cleanup temp
        try:
            os.remove(tmp_avi_path)
        except Exception:
            pass

    print(
        f"[done] wrote video: {out_path}  frames={frame_count if args.frames else 'see logs'}"
    )


if __name__ == "__main__":
    main()


"""
VERWENDUNG:
Aus Frame-Ordner:

python src/infer_export_video_nn.py \
  --ckpt artifacts/train_bspline_m4/best.pt \
  --method bspline \
  --frames /workspace/artifacts/infer_videodemo/input/xyz \
  --out /workspace/artifacts/infer_videodemo/output/xyz.mp4 \
  --fps 20 \
  --exist-thr 0.5 \
  --draw-ctrl \
  --encode ffmpeg

Aus video Datei:

python src/infer_export_video_nn.py \
  --ckpt artifacts/train_bspline_m4/best.pt \
  --method bspline \
  --video /workspace/artifacts/infer_videodemo/input/xyz.mp4 \
  --out /workspace/artifacts/infer_videodemo/output/2024-04-04-11-57-06.mp4 \
  --fps 20 \
  --exist-thr 0.5 \
  --encode ffmpeg

"""
