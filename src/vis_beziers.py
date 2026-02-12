import argparse
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from fitting.common_helpers import (
    stitch_side_by_side,  # unused right now but handy later
)
from fitting.load_data import get_current_data
from fitting.visualize import (
    draw_lanes,
    draw_poly_curve,
    load_image,
    send_frame_to_server,
)

# Group identity for one "image + ctrlpoints + method + variant"
GROUP_KEYS = [
    "task_name",
    "image_id",
    "image_path",
    "method",
    "variant",
    "n_control_points",
]


@contextmanager
def raw_terminal():
    """Put terminal into raw mode temporarily."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key_nonblocking():
    """Return a single character if available, else None."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def _infer_variant_from_results_path(results_csv: str) -> str:
    """
    Infer visualization variant label from the results CSV filename.

    Used to organize artifacts under:
        artifacts/vis_beziers/<variant>/...

    Heuristic:
    - If the filename contains "dens" or "densified" -> "densified"
    - Else -> "raw"
    """
    name = os.path.basename(results_csv).lower()
    if "dens" in name or "densified" in name:
        return "densified"
    return "raw"


def _annotate(img, text: str, row=1):
    """Draw readable annotation text on an image (shadow + foreground)."""
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


def _prepare_results(results_csv: str, metric: str) -> pd.DataFrame:
    """Load results CSV and ensure selected metric column is numeric."""
    res = pd.read_csv(results_csv)
    res[metric] = pd.to_numeric(res[metric], errors="coerce")

    # robust bool parsing from CSV
    if "lane_present" in res.columns:
        res["lane_present"] = (
            res["lane_present"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
    if "fit_success" in res.columns:
        res["fit_success"] = (
            res["fit_success"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
    return res


def _filter_only_full_groups(res: pd.DataFrame, n_control_points: int) -> pd.DataFrame:
    """
    Keep only images where all present lanes have a successful fit.

    Condition per image group:
        for each lane: fit_success == lane_present
    """
    r = res[res["n_control_points"] == n_control_points].copy()
    r["mismatch"] = r["fit_success"] != r["lane_present"]

    full = (
        r.groupby(GROUP_KEYS, as_index=False)["mismatch"]
        .any()
        .rename(columns={"mismatch": "any_mismatch"})
    )
    full = full[full["any_mismatch"] == False].drop(
        columns=["any_mismatch"]
    )  # noqa: E712
    return full


def _groups_for_ctrlpoints(
    res: pd.DataFrame, n_control_points: int, metric: str, only_full: bool
) -> pd.DataFrame:
    """
    Build one-row-per-image groups for a given control-point count, ranked by mean metric.

    Ranking logic:
    - Only lanes with fit_success == True contribute to the mean metric.
    - If only_full is True, only groups where (fit_success == lane_present)
      holds for all lanes are kept.

    Returns:
        DataFrame with columns:
            GROUP_KEYS + ["metric_mean", "n_lanes_ok"]
    """
    r = res.copy()

    if only_full:
        full_groups = _filter_only_full_groups(r, n_control_points)
        r = r.merge(full_groups, on=GROUP_KEYS, how="inner")

    r = r[
        (r["n_control_points"] == n_control_points) & (r["fit_success"] == True)
    ]  # noqa: E712
    r[metric] = pd.to_numeric(r[metric], errors="coerce")
    r = r[np.isfinite(r[metric])]

    if r.empty:
        return pd.DataFrame(columns=GROUP_KEYS + ["metric_mean", "n_lanes_ok"])

    agg = r.groupby(GROUP_KEYS, as_index=False).agg(
        metric_mean=(metric, "mean"),
        n_lanes_ok=("lane", "count"),
    )
    return agg


# ---------------------------
# Bezier reconstruction + drawing
# ---------------------------


def bezier_ctrl_from_result_row(row: pd.Series) -> np.ndarray:
    """Reconstruct (m,2) control point array from a results row."""
    m = int(row["n_control_points"])
    ctrl = np.zeros((m, 2), dtype=float)
    for i in range(m):
        ctrl[i, 0] = float(row[f"ctrl_p{i}_x"])
        ctrl[i, 1] = float(row[f"ctrl_p{i}_y"])
    return ctrl


def bernstein_matrix(t: np.ndarray, degree: int) -> np.ndarray:
    """Bernstein basis matrix (N, degree+1)."""
    import math

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
    """Evaluate Bezier curve for given control points and t array."""
    ctrl = np.asarray(ctrl, dtype=float)
    m = ctrl.shape[0]
    degree = m - 1
    A = bernstein_matrix(np.asarray(t, dtype=float), degree)
    return A @ ctrl  # (N,2)


def draw_control_polygon(
    img: np.ndarray, ctrl: np.ndarray, color=(0, 165, 255), thickness=2
) -> np.ndarray:
    """Draw control polygon P0->P1->...->Pn (orange) and control points."""
    h, w = img.shape[:2]
    pts = []
    for x, y in ctrl:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        pts.append((xi, yi))

    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)

    for x, y in pts:
        cv2.circle(img, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
    return img


def render_overlay_all_lanes(
    group_row: pd.Series,
    res: pd.DataFrame,
    gt_df: pd.DataFrame,
    metric: str,
    curve_samples: int = 400,
    show_ctrl: bool = True,
) -> np.ndarray:
    """Render GT lanes and all Bezier predictions for one image and control-point count."""
    img_path = group_row["image_path"]
    m = int(group_row["n_control_points"])
    metric_mean = float(group_row["metric_mean"])
    task = str(group_row["task_name"])
    img_id = str(group_row["image_id"])

    gt_match = gt_df.loc[gt_df["image_path"] == img_path]
    if gt_match.empty:
        raise RuntimeError(f"GT row not found for image_path: {img_path}")
    gt_row = gt_match.iloc[0]

    img = load_image(img_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    # 1) draw GT first
    img = draw_lanes(img, gt_row)

    # 2) draw predictions in yellow
    mask = (
        (res["image_path"] == img_path)
        & (res["n_control_points"] == m)
        & (res["fit_success"] == True)  # noqa: E712
    )
    sub = res.loc[mask].copy()

    pred_color = (0, 255, 255)  # yellow (BGR)
    ts = np.linspace(0.0, 1.0, num=int(curve_samples), dtype=float)

    for _, lane_row in sub.iterrows():
        ctrl = bezier_ctrl_from_result_row(lane_row)
        curve = bezier_eval(ctrl, ts)  # (N,2)
        xs = curve[:, 0].tolist()
        ys = curve[:, 1].tolist()
        img = draw_poly_curve(img, xs, ys, color=pred_color, thickness=4, clip=True)
        if show_ctrl:
            img = draw_control_polygon(img, ctrl)

    # 3) draw GT again so points stay visible
    img = draw_lanes(img, gt_row)

    base = os.path.basename(img_path)
    img = _annotate(
        img,
        f"ctrl={m} mean{metric}={metric_mean:.2f}px lanesOK={int(group_row['n_lanes_ok'])}  "
        f"{task}:{img_id}  {base}",
    )
    return img


def export_best_worst(
    results_csv: str,
    out_dir: str,
    n_each: int,
    metric: str,
    n_control_points_list: Iterable[int],
    only_full: bool,
    curve_samples: int,
    show_ctrl: bool,
):
    """
    Export visualizations of the best and worst Bezier fits.

    Output structure:
        <out_dir>/<variant>/
            best/ctrl{m}/...
            worst/ctrl{m}/...
    """
    variant = _infer_variant_from_results_path(results_csv)

    out_base = Path(out_dir) / variant
    out_best = out_base / "best"
    out_worst = out_base / "worst"
    out_best.mkdir(parents=True, exist_ok=True)
    out_worst.mkdir(parents=True, exist_ok=True)

    res = _prepare_results(results_csv, metric)
    gt_df = get_current_data()

    for m in n_control_points_list:
        groups = _groups_for_ctrlpoints(res, int(m), metric, only_full=only_full)
        if groups.empty:
            continue

        best = groups.nsmallest(n_each, "metric_mean")
        worst = groups.nlargest(n_each, "metric_mean")

        best_dir = out_best / f"ctrl{m}"
        worst_dir = out_worst / f"ctrl{m}"
        best_dir.mkdir(parents=True, exist_ok=True)
        worst_dir.mkdir(parents=True, exist_ok=True)

        def _save(df, folder: Path):
            for _, g in df.iterrows():
                img = render_overlay_all_lanes(
                    g,
                    res,
                    gt_df,
                    metric,
                    curve_samples=curve_samples,
                    show_ctrl=show_ctrl,
                )
                val = float(g["metric_mean"])
                task = str(g["task_name"])
                img_id = str(g["image_id"])
                fname = f"{val:07.2f}_task-{task}_img-{img_id}_ctrl{m}.png"
                cv2.imwrite(str(folder / fname), img)

        _save(best, best_dir)
        _save(worst, worst_dir)


def stream(
    results_csv: str,
    interval_s: float,
    metric: str,
    mode: str,
    n_control_points: int,
    only_full: bool,
    curve_samples: int,
    show_ctrl: bool,
    skip_missing: bool = False,
):
    """
    Continuously stream Bezier fit visualizations to an image server.

    Modes:
      - mixed: random sample of up to 500 groups
      - best : 200 best groups by metric_mean
      - worst: 200 worst groups by metric_mean
      - order: iterate in GT order (as returned by get_current_data()), image by image

    Controls:
        - SPACE: pause/resume
        - CTRL+C: exit
    """
    res = _prepare_results(results_csv, metric)
    gt_df = get_current_data()

    print("▶ Stream started")
    print("⏸ Press SPACE to pause/resume, CTRL+C to exit")

    paused = False

    # -----------------------
    # ORDER mode: iterate GT order
    # -----------------------
    if mode == "order":
        gt_paths = gt_df["image_path"].tolist()
        if not gt_paths:
            raise RuntimeError("GT dataframe has no image_path rows.")

        i = 0
        with raw_terminal():
            while True:
                key = read_key_nonblocking()
                if key == " ":
                    paused = not paused
                    print("⏸ Paused" if paused else "▶ Resumed")

                if paused:
                    time.sleep(interval_s)
                    continue

                img_path = gt_paths[i % len(gt_paths)]
                i += 1

                # All result rows for this image and ctrl-count (one per lane)
                sub_mask = (res["image_path"] == img_path) & (
                    res["n_control_points"] == n_control_points
                )
                sub_all = res.loc[sub_mask].copy()

                # If no results exist for this image/ctrl-count
                if sub_all.empty:
                    if skip_missing:
                        continue

                    # Render GT only (no predictions); still annotate.
                    gt_row = gt_df.loc[gt_df["image_path"] == img_path].iloc[0]
                    g = pd.Series(
                        {
                            "task_name": gt_row.get("task_name", ""),
                            "image_id": gt_row.get("image_id", ""),
                            "image_path": img_path,
                            "method": "bezier",
                            "variant": "order",
                            "n_control_points": n_control_points,
                            "metric_mean": float("nan"),
                            "n_lanes_ok": 0,
                        }
                    )
                    try:
                        img = render_overlay_all_lanes(
                            g,
                            res,
                            gt_df,
                            metric,
                            curve_samples=curve_samples,
                            show_ctrl=show_ctrl,
                        )
                        _annotate(img, "NO FIT ROWS FOUND", row=2)
                        send_frame_to_server(img)
                    except Exception as e:
                        print(f"render/send failed: {e}")

                    time.sleep(interval_s)
                    continue

                # Determine "only_full" mismatch: fit_success must equal lane_present for each lane row
                if only_full:
                    any_mismatch = bool(
                        (sub_all["fit_success"] != sub_all["lane_present"]).any()
                    )
                    if any_mismatch and skip_missing:
                        continue

                # metric aggregation over successful lanes
                sub_ok = sub_all[sub_all["fit_success"] == True].copy()  # noqa: E712
                sub_ok[metric] = pd.to_numeric(sub_ok[metric], errors="coerce")
                sub_ok = sub_ok[np.isfinite(sub_ok[metric])]

                metric_mean = (
                    float(sub_ok[metric].mean()) if len(sub_ok) else float("nan")
                )
                n_lanes_ok = int(len(sub_ok))

                # Use first result row for meta fields
                first = sub_all.iloc[0]
                g = pd.Series(
                    {
                        "task_name": first.get("task_name", ""),
                        "image_id": first.get("image_id", ""),
                        "image_path": img_path,
                        "method": first.get("method", "bezier"),
                        "variant": first.get("variant", "order"),
                        "n_control_points": n_control_points,
                        "metric_mean": metric_mean,
                        "n_lanes_ok": n_lanes_ok,
                    }
                )

                try:
                    img = render_overlay_all_lanes(
                        g,
                        res,
                        gt_df,
                        metric,
                        curve_samples=curve_samples,
                        show_ctrl=show_ctrl,
                    )
                    if (
                        only_full
                        and (sub_all["fit_success"] != sub_all["lane_present"]).any()
                    ):
                        _annotate(img, "NOT FULL (only_full enabled)", row=2)
                    send_frame_to_server(img)
                except Exception as e:
                    print(f"render/send failed: {e}")

                time.sleep(interval_s)

        return

    # -----------------------
    # EXISTING modes: mixed / best / worst
    # -----------------------
    groups = _groups_for_ctrlpoints(res, n_control_points, metric, only_full=only_full)
    if groups.empty:
        raise RuntimeError(
            f"No successful fits found for n_control_points={n_control_points}."
        )

    if mode == "best":
        pool = (
            groups.sort_values("metric_mean", ascending=True)
            .head(200)
            .reset_index(drop=True)
        )
    elif mode == "worst":
        pool = (
            groups.sort_values("metric_mean", ascending=False)
            .head(200)
            .reset_index(drop=True)
        )
    else:
        pool = groups.sample(min(len(groups), 500)).reset_index(drop=True)

    i = 0
    with raw_terminal():
        while True:
            key = read_key_nonblocking()
            if key == " ":
                paused = not paused
                print("⏸ Paused" if paused else "▶ Resumed")

            if not paused:
                g = pool.iloc[i % len(pool)]
                try:
                    img = render_overlay_all_lanes(
                        g,
                        res,
                        gt_df,
                        metric,
                        curve_samples=curve_samples,
                        show_ctrl=show_ctrl,
                    )
                    send_frame_to_server(img)
                except Exception as e:
                    print(f"render/send failed: {e}")

                i += 1

            time.sleep(interval_s)


def main():
    """CLI entry point."""
    p = argparse.ArgumentParser()

    p.add_argument("--results", required=True, help="CSV from fit_beziers.py")
    p.add_argument("--out", default="artifacts/vis_beziers")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--metric", default="rmse_dist_px_gt")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument(
        "--mode", choices=["mixed", "best", "worst", "order"], default="mixed"
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="In order-mode: skip images with no/failed fits.",
    )
    p.add_argument(
        "--n-control-points",
        type=int,
        default=4,
        help="Control point count to visualize in stream mode (default: 4).",
    )
    p.add_argument(
        "--ctrl-list",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10],
        help="Control point counts to export (default: 3 4 5 6 7 8 9 10).",
    )
    p.add_argument(
        "--only-full",
        action="store_true",
        help="Only use images where all present lanes have a successful fit.",
    )
    p.add_argument(
        "--curve-samples",
        type=int,
        default=400,
        help="Bezier sampling resolution for drawing (default: 400).",
    )
    p.add_argument(
        "--no-ctrl",
        action="store_true",
        help="Disable drawing of control polygon / control points.",
    )

    args = p.parse_args()

    show_ctrl = not args.no_ctrl

    if args.stream:
        stream(
            results_csv=args.results,
            interval_s=args.interval,
            metric=args.metric,
            mode=args.mode,
            n_control_points=args.n_control_points,
            only_full=args.only_full,
            curve_samples=args.curve_samples,
            show_ctrl=show_ctrl,
        )
    else:
        export_best_worst(
            results_csv=args.results,
            out_dir=args.out,
            n_each=args.n,
            metric=args.metric,
            n_control_points_list=args.ctrl_list,
            only_full=args.only_full,
            curve_samples=args.curve_samples,
            show_ctrl=show_ctrl,
        )


if __name__ == "__main__":
    main()

    # Examples:
    # python /workspace/src/vis_beziers.py --results /workspace/artifacts/results_bezier_raw.csv --only-full
    # python /workspace/src/vis_beziers.py --results /workspace/artifacts/results_bezier_densified_step10p0px.csv --only-full
    # python /workspace/src/vis_beziers.py --results /workspace/artifacts/results_bezier_densified_step10p0px.csv --stream --n-control-points 4
    #
    # IN ORDER
    # python src/vis_beziers.py --results artifacts/results_bezier_densified_step10p0px.csv --stream --mode order --n-control-points 4 --interval 1
