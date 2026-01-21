import argparse
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from fitting.common_helpers import (
    get_points_of_all_3_lanes,
    poly_from_result_row,
    sample_poly_x_of_y,
    stitch_side_by_side,
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

# Group identity for one "image + degree + method + variant"
GROUP_KEYS = ["task_name", "image_id", "image_path", "method", "variant", "degree"]
COMPARE_KEYS = ["task_name", "image_id", "image_path", "degree"]


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

    This is used to organize artifacts under:
        artifacts/vis_polynomials/<variant>/...

    Heuristic:
    - If the filename contains "dens" or "densified" -> "densified"
    - Else -> "raw"
    """
    name = os.path.basename(results_csv).lower()
    if "dens" in name or "densified" in name:
        return "densified"
    return "raw"


def _filter_only_full_groups(res: pd.DataFrame, degree: int) -> pd.DataFrame:
    """
    Keep only images where all present lanes have a successful fit.

    Condition per image group:
        for each lane: fit_success == lane_present

    Args:
        res:
            Results DataFrame (per-lane rows).
        degree:
            Polynomial degree to filter.

    Returns:
        pd.DataFrame:
            A DataFrame containing only group identifier columns (GROUP_KEYS)
            for "full" groups.
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
    full = full[full["any_mismatch"] == False].drop(  # noqa: E712
        columns=["any_mismatch"]
    )
    return full


def _y_range_from_points(points) -> Optional[Tuple[float, float]]:
    """Return (y_min, y_max) for valid points, or None if no valid y-values exist."""
    ys = [p[1] for p in points if p is not None and len(p) == 2]
    if not ys:
        return None
    return float(np.min(ys)), float(np.max(ys))


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
    return res


def _groups_for_degree(
    res: pd.DataFrame, degree: int, metric: str, only_full: bool
) -> pd.DataFrame:
    """
    Build one-row-per-image groups for a given degree, ranked by mean metric.

    Ranking logic:
    - Only lanes with fit_success == True contribute to the mean metric.
    - If only_full is True, only groups where (fit_success == lane_present)
      holds for all lanes are kept.

    Returns:
        DataFrame with columns:
            GROUP_KEYS + ["rmse_mean", "n_lanes_ok"]
    """
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
    3) Ground truth points again so they remain visible.
    4) A textual annotation containing degree, mean error metric, and identifiers.

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
    metric_mean = float(group_row["rmse_mean"])
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
        f"deg={degree} mean{metric}={metric_mean:.2f}px lanesOK={int(group_row['n_lanes_ok'])}  "
        f"{task}:{img_id}  {base}",
    )
    return img


def export_best_worst(
    results_csv: str,
    out_dir: str,
    n_each: int,
    metric: str,
    degrees: Iterable[int],
    only_full: bool,
):
    """
    Export visualizations of the best and worst polynomial fits.

    For each requested polynomial degree, images are ranked by the mean
    error metric across all successfully fitted lanes. The top and bottom N
    images are rendered and saved to disk.

    Output structure:
        <out_dir>/<variant>/
            best/deg{d}/...
            worst/deg{d}/...

    Args:
        results_csv:
            Path to the CSV file produced by fit_polynomials.
        out_dir:
            Output base directory for visualizations (e.g. artifacts/vis_polynomials).
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
    variant = _infer_variant_from_results_path(results_csv)

    out_base = Path(out_dir) / variant
    out_best = out_base / "best"
    out_worst = out_base / "worst"
    out_best.mkdir(parents=True, exist_ok=True)
    out_worst.mkdir(parents=True, exist_ok=True)

    res = _prepare_results(results_csv, metric)
    gt_df = get_current_data()

    for deg in degrees:
        groups = _groups_for_degree(res, deg, metric, only_full=only_full)
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


def export_compare_best_worst(
    *,
    results_raw_csv: str,
    results_dens_csv: str,
    out_dir: str,
    n_each: int,
    metric: str,
    degrees: Iterable[int],
    only_full: bool,
    # optional: if you want to rank by something else later
    rank_by: str = "delta",  # "delta" | "raw" | "dens"
):
    """
    Export side-by-side visualizations comparing RAW vs DENSIFIED fits.

    Ranking is performed per image (and degree) using the mean metric across lanes:
        raw_mean = mean(metric) over successful lanes in raw
        dens_mean = mean(metric) over successful lanes in densified
        delta = raw_mean - dens_mean

    - "best": highest delta (densified improved most)
    - "worst": lowest delta (densified improved least / got worse)

    Output structure:
        <out_dir>/compare/deg{d}/best/*.png
        <out_dir>/compare/deg{d}/worst/*.png
    """
    out_base = Path(out_dir) / "compare"
    out_base.mkdir(parents=True, exist_ok=True)

    # load + keep metric numeric
    res_raw = _prepare_results(results_raw_csv, metric)
    res_dens = _prepare_results(results_dens_csv, metric)

    # GT
    gt_df = get_current_data()

    for deg in degrees:
        # aggregate groups per image for each results table
        g_raw = _groups_for_degree(res_raw, deg, metric, only_full=only_full).copy()
        g_dens = _groups_for_degree(res_dens, deg, metric, only_full=only_full).copy()

        if g_raw.empty or g_dens.empty:
            continue

        # Drop keys that prevent matching (variant/method differ)
        # _groups_for_degree returns GROUP_KEYS + ["rmse_mean", "n_lanes_ok"]
        # GROUP_KEYS = ["task_name","image_id","image_path","method","variant","degree"]
        # For compare we match on COMPARE_KEYS only.
        keep_cols = COMPARE_KEYS + ["rmse_mean", "n_lanes_ok"]
        g_raw2 = g_raw[
            ["task_name", "image_id", "image_path", "degree", "rmse_mean", "n_lanes_ok"]
        ].copy()
        g_dens2 = g_dens[
            ["task_name", "image_id", "image_path", "degree", "rmse_mean", "n_lanes_ok"]
        ].copy()

        g_raw2 = g_raw2.rename(
            columns={"rmse_mean": "raw_mean", "n_lanes_ok": "raw_lanes_ok"}
        )
        g_dens2 = g_dens2.rename(
            columns={"rmse_mean": "dens_mean", "n_lanes_ok": "dens_lanes_ok"}
        )

        merged = g_raw2.merge(g_dens2, on=COMPARE_KEYS, how="inner")
        if merged.empty:
            continue

        merged["delta"] = merged["raw_mean"] - merged["dens_mean"]

        if rank_by == "raw":
            merged = merged.sort_values("raw_mean", ascending=True)
        elif rank_by == "dens":
            merged = merged.sort_values("dens_mean", ascending=True)
        else:
            # default: biggest improvement first
            merged = merged.sort_values("delta", ascending=False)

        best = merged.head(n_each).reset_index(drop=True)
        worst = merged.tail(n_each).reset_index(drop=True)

        best_dir = out_base / f"deg{deg}" / "best"
        worst_dir = out_base / f"deg{deg}" / "worst"
        best_dir.mkdir(parents=True, exist_ok=True)
        worst_dir.mkdir(parents=True, exist_ok=True)

        def _render_pair(row: pd.Series) -> np.ndarray:
            # Build pseudo group_row for raw/dens, compatible with render_overlay_all_lanes
            group_raw = pd.Series(
                {
                    "task_name": row["task_name"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "degree": row["degree"],
                    "rmse_mean": row["raw_mean"],
                    "n_lanes_ok": row["raw_lanes_ok"],
                }
            )
            group_dens = pd.Series(
                {
                    "task_name": row["task_name"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "degree": row["degree"],
                    "rmse_mean": row["dens_mean"],
                    "n_lanes_ok": row["dens_lanes_ok"],
                }
            )

            img_raw = render_overlay_all_lanes(group_raw, res_raw, gt_df, metric)
            img_dens = render_overlay_all_lanes(group_dens, res_dens, gt_df, metric)

            stitched = stitch_side_by_side(img_raw, img_dens)

            # Optional header annotation on stitched image
            delta = float(row["delta"])
            raw_m = float(row["raw_mean"])
            dens_m = float(row["dens_mean"])
            text = f"deg={deg}  raw={raw_m:.2f}px  dens={dens_m:.2f}px  delta={delta:+.2f}px"
            stitched = _annotate(stitched, text)
            return stitched

        def _save(df: pd.DataFrame, folder: Path, tag: str):
            for _, r in df.iterrows():
                img = _render_pair(r)
                task = str(r["task_name"])
                img_id = str(r["image_id"])
                delta = float(r["delta"])
                fname = f"{delta:+07.2f}_task-{task}_img-{img_id}_deg{deg}_{tag}.png"
                cv2.imwrite(str(folder / fname), img)

        _save(best, best_dir, "best")
        _save(worst, worst_dir, "worst")


def build_compare_groups(
    res_raw: pd.DataFrame,
    res_dens: pd.DataFrame,
    degree: int,
    metric: str,
    *,
    only_full: bool,
    rank_by: str = "delta",  # "delta" | "raw" | "dens"
) -> pd.DataFrame:
    """
    Build a compare group table that exists in both raw and densified results.

    Produces one row per image (and degree) with:
        raw_mean, dens_mean, delta (raw - dens)

    Important:
        Matching is done on COMPARE_KEYS (task_name, image_id, image_path, degree)
        and NOT on GROUP_KEYS, because 'variant' differs between raw and densified runs.

    Returns:
        DataFrame with columns:
            COMPARE_KEYS + [raw_mean, dens_mean, delta, n_lanes_ok_raw, n_lanes_ok_dens]
    """
    g_raw = _groups_for_degree(res_raw, degree, metric, only_full=only_full).copy()
    g_dens = _groups_for_degree(res_dens, degree, metric, only_full=only_full).copy()

    if g_raw.empty or g_dens.empty:
        return pd.DataFrame(
            columns=COMPARE_KEYS
            + ["raw_mean", "dens_mean", "delta", "n_lanes_ok_raw", "n_lanes_ok_dens"]
        )

    # keep only columns we can actually match on
    g_raw2 = g_raw[
        ["task_name", "image_id", "image_path", "degree", "rmse_mean", "n_lanes_ok"]
    ].rename(columns={"rmse_mean": "raw_mean", "n_lanes_ok": "n_lanes_ok_raw"})
    g_dens2 = g_dens[
        ["task_name", "image_id", "image_path", "degree", "rmse_mean", "n_lanes_ok"]
    ].rename(columns={"rmse_mean": "dens_mean", "n_lanes_ok": "n_lanes_ok_dens"})

    g = g_raw2.merge(g_dens2, on=COMPARE_KEYS, how="inner")
    if g.empty:
        return g

    g["delta"] = g["raw_mean"] - g["dens_mean"]

    if rank_by == "raw":
        g = g.sort_values("raw_mean", ascending=True)
    elif rank_by == "dens":
        g = g.sort_values("dens_mean", ascending=True)
    else:
        g = g.sort_values("delta", ascending=False)

    return g.reset_index(drop=True)


def render_compare_overlay(
    group_row: pd.Series,
    res_raw: pd.DataFrame,
    res_dens: pd.DataFrame,
    gt_df: pd.DataFrame,
    metric: str,
) -> np.ndarray:
    """
    Render side-by-side comparison image: raw vs densified for the same image/degree.

    group_row comes from build_compare_groups() and contains:
        raw_mean, dens_mean, delta, n_lanes_ok_raw, n_lanes_ok_dens
    """
    # Build two pseudo group rows compatible with render_overlay_all_lanes()
    group_raw = pd.Series(
        {
            "task_name": group_row["task_name"],
            "image_id": group_row["image_id"],
            "image_path": group_row["image_path"],
            "degree": group_row["degree"],
            "rmse_mean": group_row["raw_mean"],
            "n_lanes_ok": group_row["n_lanes_ok_raw"],
        }
    )
    group_dens = pd.Series(
        {
            "task_name": group_row["task_name"],
            "image_id": group_row["image_id"],
            "image_path": group_row["image_path"],
            "degree": group_row["degree"],
            "rmse_mean": group_row["dens_mean"],
            "n_lanes_ok": group_row["n_lanes_ok_dens"],
        }
    )

    img_left = render_overlay_all_lanes(group_raw, res_raw, gt_df, metric)
    img_right = render_overlay_all_lanes(group_dens, res_dens, gt_df, metric)

    # small labels
    img_left = _annotate(img_left, "RAW", row=2)
    img_right = _annotate(img_right, "DENSIFIED", row=2)

    stitched = stitch_side_by_side(img_left, img_right)

    delta = float(group_row["delta"])
    raw_m = float(group_row["raw_mean"])
    dens_m = float(group_row["dens_mean"])
    stitched = _annotate(
        stitched, f"raw={raw_m:.2f}px  dens={dens_m:.2f}px  delta={delta:+.2f}px", row=3
    )

    return stitched


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

    Controls:
        - SPACE: pause/resume
        - CTRL+C: exit

    Args:
        results_csv:
            Path to the CSV file produced by fit_polynomials.
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


def stream_compare(
    results_raw_csv: str,
    results_dens_csv: str,
    interval_s: float,
    metric: str,
    mode: str,
    degree: int,
    only_full: bool,
    rank_by: str,
):
    """
    Stream side-by-side visualizations comparing raw and densified polynomial fits.

    For each selected image and polynomial degree, this function renders a
    stitched visualization showing:
        - left: polynomial fits based on raw lane points
        - right: polynomial fits based on densified lane points

    Images are selected according to the chosen ranking strategy:
        - "delta": largest improvement from raw to densified (raw_mean - dens_mean)
        - "raw": lowest raw error first
        - "dens": lowest densified error first

    The stream runs continuously and sends frames to the image server at a
    fixed interval. Keyboard controls allow interactive inspection.

    Controls:
        SPACE   Pause or resume the stream
        CTRL+C Exit the stream

    Args:
        results_raw_csv:
            CSV file containing results from raw polynomial fitting.
        results_dens_csv:
            CSV file containing results from densified polynomial fitting.
        interval_s:
            Delay between frames in seconds.
        metric:
            Error metric column used for ranking and annotation
            (e.g. "rmse_x_px_gt").
        mode:
            Selection mode for streaming ("best", "worst", or "mixed").
        degree:
            Polynomial degree to visualize.
        only_full:
            If True, only images where all present lanes were successfully
            fitted in both raw and densified results are included.
        rank_by:
            Ranking criterion for comparison ("delta", "raw", or "dens").

    Raises:
        RuntimeError:
            If no overlapping image groups exist between raw and densified
            results after filtering.
    """
    res_raw = _prepare_results(results_raw_csv, metric)
    res_dens = _prepare_results(results_dens_csv, metric)
    gt_df = get_current_data()

    groups = build_compare_groups(
        res_raw, res_dens, degree, metric, only_full=only_full, rank_by=rank_by
    )
    if groups.empty:
        raise RuntimeError("No overlapping groups between raw and densified results.")

    if mode == "best":
        pool = groups.head(200).reset_index(drop=True)
    elif mode == "worst":
        pool = groups.tail(200).reset_index(drop=True)
    else:
        pool = groups.sample(min(len(groups), 500)).reset_index(drop=True)

    paused = False
    i = 0
    print("▶ Compare stream started (RAW | DENSIFIED)")
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
                    img = render_compare_overlay(g, res_raw, res_dens, gt_df, metric)
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

    # --- common ---
    p.add_argument("--results", help="CSV from fit_polynomials (single mode)")
    p.add_argument("--out", default="artifacts/vis_polynomials")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--metric", default="rmse_x_px_gt")
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

    # --- compare mode ---
    p.add_argument(
        "--compare",
        action="store_true",
        help="Enable side-by-side comparison (raw vs densified).",
    )
    p.add_argument(
        "--results-raw",
        help="CSV from raw polynomial fitting (required for --compare).",
    )
    p.add_argument(
        "--results-dens",
        help="CSV from densified polynomial fitting (required for --compare).",
    )
    p.add_argument(
        "--rank-by",
        choices=["delta", "raw", "dens"],
        default="delta",
        help="Ranking strategy for compare mode.",
    )

    args = p.parse_args()

    # ----------------------------
    # Compare mode
    # ----------------------------
    if args.compare:
        if args.results_raw is None or args.results_dens is None:
            raise SystemExit("--compare requires --results-raw and --results-dens")

        if args.stream:
            if args.degree is None:
                raise SystemExit("--degree is required in --compare mode")
            stream_compare(
                results_raw_csv=args.results_raw,
                results_dens_csv=args.results_dens,
                interval_s=args.interval,
                metric=args.metric,
                mode=args.mode,
                degree=args.degree,
                only_full=args.only_full,
                rank_by=args.rank_by,
            )
        else:
            export_compare_best_worst(
                results_raw_csv=args.results_raw,
                results_dens_csv=args.results_dens,
                out_dir=args.out,
                n_each=args.n,
                metric=args.metric,
                degrees=[args.degree] if args.degree is not None else [2, 3, 4],
                only_full=args.only_full,
                rank_by=args.rank_by,
            )
        return

    # ----------------------------
    # Normal (single CSV) mode
    # ----------------------------
    if args.results is None:
        raise SystemExit("--results is required unless --compare is used")

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
    # /usr/local/bin/python /workspace/src/vis_polynomials.py --results /workspace/artifacts/results_poly_raw.csv --only-full
    # /usr/local/bin/python /workspace/src/vis_polynomials.py --results /workspace/artifacts/results_poly_raw.csv
    # /usr/local/bin/python /workspace/src/vis_polynomials.py --results /workspace/artifacts/results_poly_raw.csv --only-full --stream --degree 3
    #
    #
    # Densified (Ordner wird automatisch artifacts/vis_polynomials/densified/...):
    # /usr/local/bin/python /workspace/src/vis_polynomials.py --results /workspace/artifacts/results_poly_densified.csv --only-full
    #
    # Compare output: (sollte man nur mit only-full verwenden)
    # /usr/local/bin/python /workspace/src/vis_polynomials.py --compare --results-raw artifacts/results_poly_raw.csv --results-dens artifacts/results_poly_densified_step10p0px.csv --rank-by delta --only-full
    #
    # Compare stream: (sollte man nur mit only-full verwenden)
    # /usr/local/bin/python /workspace/src/vis_polynomials.py --compare --results-raw artifacts/results_poly_raw.csv --results-dens artifacts/results_poly_densified_step10p0px.csv --rank-by delta --only-full --stream --degree 3
