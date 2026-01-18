import argparse
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fitting.common_helpers import (
    get_points_of_all_3_lanes,
    poly_from_result_row,
    sample_poly_x_of_y,
)
from fitting.load_data import get_current_data
from fitting.visualize import (
    draw_lanes,
    draw_poly_curve,
    load_image,
    send_frame_to_server,
)

LANE_ORDER = ["left", "center", "right"]
LANE_COLOR = {
    "left": (0, 0, 255),  # red (BGR)
    "center": (0, 255, 0),  # green
    "right": (255, 0, 0),  # blue
}

GROUP_KEYS = ["task_name", "image_id", "image_path", "method", "variant", "degree"]


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


def _filter_only_full_groups(res: pd.DataFrame, degree: int) -> pd.DataFrame:
    """
    Keep only images where all present lanes have a successful fit.

    Condition per image:
        for each lane: fit_success == lane_present
    """
    r = res[res["degree"] == degree].copy()

    # Robust bool parsing from CSV
    r["lane_present"] = (
        r["lane_present"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    r["fit_success"] = (
        r["fit_success"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    # mismatch per row (lane)
    r["mismatch"] = r["fit_success"] != r["lane_present"]

    # if any mismatch in a group -> not full
    full = (
        r.groupby(GROUP_KEYS, as_index=False)["mismatch"]
        .any()
        .rename(columns={"mismatch": "any_mismatch"})
    )

    # keep only groups with no mismatch
    full = full[full["any_mismatch"] == False].drop(
        columns=["any_mismatch"]
    )  # noqa: E712
    return full


def _y_range_from_points(points):
    ys = [p[1] for p in points if p is not None and len(p) == 2]
    if not ys:
        return None
    return float(np.min(ys)), float(np.max(ys))


def _annotate(img, text: str):
    cv2.putText(
        img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        img,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _prepare_results(results_csv: str, metric: str) -> pd.DataFrame:
    res = pd.read_csv(results_csv)
    res[metric] = pd.to_numeric(res[metric], errors="coerce")
    return res


def _groups_for_degree(
    res: pd.DataFrame, degree: int, metric: str, only_full: bool
) -> pd.DataFrame:
    r = res.copy()

    # Optional: keep only "full" images
    if only_full:
        full_groups = _filter_only_full_groups(r, degree)
        # Join to keep only rows belonging to those groups
        r = r.merge(full_groups, on=GROUP_KEYS, how="inner")

    # For ranking: only successful lanes contribute to mean
    r = r[(r["degree"] == degree) & (r["fit_success"] == True)]  # noqa: E712
    r[metric] = pd.to_numeric(r[metric], errors="coerce")
    r = r[np.isfinite(r[metric])]

    if r.empty:
        return pd.DataFrame(columns=GROUP_KEYS + ["rmse_mean", "n_lanes_ok"])

    agg = r.groupby(GROUP_KEYS, as_index=False).agg(
        rmse_mean=(metric, "mean"),
        n_lanes_ok=("lane", "count"),
    )
    return agg


def render_overlay_all_lanes(
    group_row: pd.Series, res: pd.DataFrame, gt_df: pd.DataFrame, metric: str
) -> np.ndarray:
    """
    Render ground truth lanes and all polynomial predictions for one image.

    The function overlays:
    1) Ground truth lane points for left, center, and right lanes.
    2) Polynomial predictions for all successfully fitted lanes (same degree).
    3) A textual annotation containing degree, mean error metric, and identifiers.

    Args:
        group_row:
            Aggregated result row representing one image and polynomial degree.
        res:
            Full results DataFrame containing per-lane polynomial fits.
        gt_df:
            Ground truth DataFrame containing lane annotations.
        metric:
            Error metric name used for annotation (e.g. "rmse_x_px").

    Returns:
        numpy.ndarray:
            Image with ground truth and prediction overlays.
    """
    img_path = group_row["image_path"]
    degree = int(group_row["degree"])
    rmse_mean = float(group_row["rmse_mean"])
    task = str(group_row["task_name"])
    img_id = str(group_row["image_id"])

    gt_match = gt_df.loc[gt_df["image_path"] == img_path]
    if gt_match.empty:
        raise RuntimeError(f"GT row not found for image_path: {img_path}")
    gt_row = gt_match.iloc[0]

    img = load_image(img_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    # 1) Ground truth points first
    img = draw_lanes(img, gt_row)

    # 2) Draw predictions in yellow
    mask = (
        (res["image_path"] == img_path)
        & (res["degree"] == degree)
        & (res["fit_success"] == True)  # noqa: E712
    )
    sub = res.loc[mask].copy()

    left_pts, center_pts, right_pts = get_points_of_all_3_lanes(gt_row)
    lane_pts_map = {"left": left_pts, "center": center_pts, "right": right_pts}

    pred_color = (0, 255, 255)  # yellow in BGR

    for _, lane_row in sub.iterrows():
        lane = str(lane_row["lane"])
        pts = lane_pts_map.get(lane, [])
        yr = _y_range_from_points(pts)
        if yr is None:
            continue

        y_min, y_max = yr
        poly = poly_from_result_row(lane_row)
        xs, ys = sample_poly_x_of_y(poly, y_min, y_max, n=300)
        img = draw_poly_curve(img, xs, ys, color=pred_color, thickness=4, clip=True)

    # 3) Draw GT points again on top so they remain visible
    img = draw_lanes(img, gt_row)

    base = os.path.basename(img_path)
    img = _annotate(
        img,
        f"deg={degree} meanRMSE={rmse_mean:.2f}px lanesOK={int(group_row['n_lanes_ok'])}  {task}:{img_id}  {base}",
    )
    return img


def export_best_worst(
    results_csv: str, out_dir: str, n_each: int, metric: str, degrees, only_full: bool
):
    """
    Export visualizations of the best and worst polynomial fits.

    For each requested polynomial degree, images are ranked by the mean
    error metric across all fitted lanes. The top and bottom N images
    are rendered and saved to disk.

    Args:
        results_csv:
            Path to the CSV file produced by fit_polynomials_raw.
        out_dir:
            Output directory for visualizations.
        n_each:
            Number of best and worst images to export per degree.
        metric:
            Error metric used for ranking (e.g. "rmse_x_px").
        degrees:
            Iterable of polynomial degrees to process.
        only_full:
            If True, only images where all present lanes were successfully
            fitted are considered.
    """
    out_best = Path(out_dir) / "best"
    out_worst = Path(out_dir) / "worst"
    out_best.mkdir(parents=True, exist_ok=True)
    out_worst.mkdir(parents=True, exist_ok=True)

    res = _prepare_results(results_csv, metric)
    gt_df = get_current_data()

    for deg in degrees:
        groups = _groups_for_degree(res, deg, metric, only_full)
        if groups.empty:
            continue

        best = groups.nsmallest(n_each, "rmse_mean")
        worst = groups.nlargest(n_each, "rmse_mean")

        best_dir = out_best / f"deg{deg}"
        worst_dir = out_worst / f"deg{deg}"
        best_dir.mkdir(parents=True, exist_ok=True)
        worst_dir.mkdir(parents=True, exist_ok=True)

        def _save(df, folder: Path):
            for _, g in df.iterrows():
                img = render_overlay_all_lanes(g, res, gt_df, metric)
                val = float(g["rmse_mean"])
                task = str(g["task_name"])
                img_id = str(g["image_id"])
                fname = f"{val:07.2f}_task-{task}_img-{img_id}_deg{deg}.png"
                cv2.imwrite(str(folder / fname), img)

        _save(best, best_dir)
        _save(worst, worst_dir)


def stream(
    results_csv: str,
    interval_s: float,
    metric: str,
    mode: str,
    degree: int,
    only_full: bool,
):
    """
    Continuously stream polynomial fit visualizations to an image server.

    Images are selected from the result set according to the chosen mode
    (best, worst, or mixed) and rendered sequentially at a fixed interval.

    Args:
        results_csv:
            Path to the CSV file produced by fit_polynomials_raw.
        interval_s:
            Time delay between frames in seconds.
        metric:
            Error metric used for ranking and annotation.
        mode:
            Selection mode ("best", "worst", or "mixed").
        degree:
            Polynomial degree to visualize.
        only_full:
            If True, only images with successful fits for all present lanes
            are included.
    """
    res = _prepare_results(results_csv, metric)
    gt_df = get_current_data()

    groups = _groups_for_degree(res, degree, metric, only_full=only_full)
    if groups.empty:
        raise RuntimeError(f"No successful fits found for degree={degree}.")

    if mode == "best":
        pool = (
            groups.sort_values("rmse_mean", ascending=True)
            .head(200)
            .reset_index(drop=True)
        )
    elif mode == "worst":
        pool = (
            groups.sort_values("rmse_mean", ascending=False)
            .head(200)
            .reset_index(drop=True)
        )
    else:
        pool = groups.sample(min(len(groups), 500)).reset_index(drop=True)

    paused = False
    i = 0

    print("▶ Stream started")
    print("⏸ Press SPACE to pause/resume, CTRL+C to exit")

    with raw_terminal():
        while True:
            key = read_key_nonblocking()
            if key == " ":
                paused = not paused
                print("⏸ Paused" if paused else "▶ Resumed")

            if not paused:
                g = pool.iloc[i % len(pool)]
                try:
                    img = render_overlay_all_lanes(g, res, gt_df, metric)
                    send_frame_to_server(img)
                except Exception as e:
                    print(f"render/send failed: {e}")

                i += 1

            time.sleep(interval_s)


def main():
    """
    Entry point for polynomial fit visualization.

    Parses command-line arguments and either:
    - exports best and worst visualizations to disk, or
    - starts a live visualization stream for interactive inspection.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="CSV from fit_polynomials_raw")
    p.add_argument("--out", default="artifacts/vis_polynomials_raw")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--metric", default="rmse_x_px")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--mode", choices=["mixed", "best", "worst"], default="mixed")
    p.add_argument(
        "--degree",
        type=int,
        choices=[2, 3, 4],
        help="Polynomial degree to visualize (required for --stream).",
    )
    p.add_argument(
        "--only-full",
        action="store_true",
        help="Only use images where all present lanes have a successful fit.",
    )
    args = p.parse_args()

    if args.stream:
        if args.degree is None:
            raise SystemExit("--degree is required in --stream mode")
        stream(
            args.results,
            args.interval,
            args.metric,
            args.mode,
            degree=args.degree,
            only_full=args.only_full,
        )
    else:
        degrees = [args.degree] if args.degree is not None else [2, 3, 4]
        export_best_worst(
            args.results,
            args.out,
            args.n,
            args.metric,
            degrees=degrees,
            only_full=args.only_full,
        )


if __name__ == "__main__":
    main()

    # Verwendungsbeispiele:
    # /usr/local/bin/python /workspace/src/vis_polynomials_raw.py --results /workspace/artifacts/results_poly_raw.csv --only-full
    # /usr/local/bin/python /workspace/src/vis_polynomials_raw.py --results /workspace/artifacts/results_poly_raw.csv
    # /usr/local/bin/python /workspace/src/vis_polynomials_raw.py --results /workspace/artifacts/results_poly_raw.csv --only-full --stream --degree 3
