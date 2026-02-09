"""
Evaluate polynomial fitting results (raw or densified).

This script summarizes fit coverage and error metrics (RMSE/MAE/MedAE/MaxAE)
for polynomial lane fits across degrees. It exports CSV summaries and saves
plots as static images for later reporting and comparison.

Key features (updated):
- Automatically splits outputs into:
    artifacts/eval_polynomials/raw/...
    artifacts/eval_polynomials/densified/...
  (inferred from the results CSV filename)
- Computes and exports ALL summaries/plots twice:
    *_gt   : errors against original GT points
    *_used : errors against the points actually used for fitting (raw==used for raw runs)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANE_ORDER = ["left", "center", "right"]
DEGREES_DEFAULT = [2, 3, 4, 5, 6]
GROUP_KEYS = ["task_name", "image_id", "image_path", "method", "variant", "degree"]


def _infer_variant_from_results_path(results_csv: str) -> str:
    """
    Infer evaluation subfolder from results filename.

    Heuristic:
    - If the filename contains "dens" or "densified" -> "densified"
    - Else -> "raw"
    """
    name = os.path.basename(results_csv).lower()
    if "dens" in name or "densified" in name:
        return "densified"
    return "raw"


def _ensure_bool(s: pd.Series) -> pd.Series:
    """
    Convert a Series to boolean in a CSV-friendly way.

    Accepts True/False, 1/0, "true"/"false", "True"/"False", "yes"/"no".
    """
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load_results(results_csv: str, metric_cols: Iterable[str]) -> pd.DataFrame:
    """
    Load the results CSV and coerce important columns.

    Args:
        results_csv: Path to CSV produced by fit_polynomials.
        metric_cols: Metric columns to coerce to numeric.

    Returns:
        DataFrame with cleaned dtypes for degree/booleans/metrics.
    """
    df = pd.read_csv(results_csv)

    if "degree" in df.columns:
        df["degree"] = pd.to_numeric(df["degree"], errors="coerce").astype("Int64")

    for c in ["lane_present", "enough_points", "fit_success"]:
        if c in df.columns:
            df[c] = _ensure_bool(df[c])

    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def is_full_image_group(g: pd.DataFrame) -> bool:
    """
    Check if an image group is 'fully fitted'.

    Rule:
      For all lanes where lane_present == True, fit_success must be True.

    Args:
        g: DataFrame slice of one image group.

    Returns:
        True if the group is considered fully fitted.
    """
    present = g["lane_present"].to_numpy(bool)
    success = g["fit_success"].to_numpy(bool)
    return bool(success[present].all())


def add_full_image_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'is_full_image' flag to each row based on group completeness.

    Args:
        df: Results DataFrame.

    Returns:
        Copy of df with 'is_full_image' boolean column.
    """
    r = df.copy()
    flags = (
        r.groupby(GROUP_KEYS, as_index=False)
        .apply(lambda g: is_full_image_group(g), include_groups=False)
        .rename(columns={None: "is_full_image"})
    )
    if "is_full_image" not in flags.columns:
        flags = flags.rename(columns={0: "is_full_image"})

    r = r.merge(flags, on=GROUP_KEYS, how="left")
    r["is_full_image"] = r["is_full_image"].fillna(False).astype(bool)
    return r


def summarize_coverage_lane_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize lane-level coverage statistics by degree and lane.

    Metrics reported:
        - n_total: all lane records
        - n_present: lane_present == True
        - n_present_fail_points: present but not enough points
        - n_present_fit_fail: present + enough points but fit failed
        - n_fit_success: present & fit_success
        - rates computed relative to n_present

    Args:
        df: Results DataFrame.

    Returns:
        Coverage summary DataFrame.
    """
    rows: List[Dict[str, object]] = []
    for (deg, lane), g in df.groupby(["degree", "lane"], dropna=True):
        n_total = int(len(g))
        present = g["lane_present"] == True  # noqa: E712
        enough = (g["lane_present"] == True) & (
            g["enough_points"] == True
        )  # noqa: E712
        success = (g["lane_present"] == True) & (g["fit_success"] == True)  # noqa: E712

        n_present = int(present.sum())
        n_enough = int(enough.sum())
        n_success = int(success.sum())

        n_present_fail_points = int(
            (
                (g["lane_present"] == True) & (g["enough_points"] == False)
            ).sum()  # noqa: E712
        )
        n_present_fit_fail = int(
            (
                (g["lane_present"] == True)
                & (g["enough_points"] == True)
                & (g["fit_success"] == False)
            ).sum()  # noqa: E712
        )

        denom = n_present if n_present > 0 else np.nan
        rows.append(
            {
                "degree": int(deg),
                "lane": str(lane),
                "n_total": n_total,
                "n_present": n_present,
                "n_enough_points": n_enough,
                "n_present_fail_points": n_present_fail_points,
                "n_present_fit_fail": n_present_fit_fail,
                "n_fit_success": n_success,
                "rate_fit_success_given_present": (n_success / denom)
                if n_present
                else np.nan,
                "rate_fail_points_given_present": (n_present_fail_points / denom)
                if n_present
                else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["lane"] = pd.Categorical(out["lane"], categories=LANE_ORDER, ordered=True)
    out = out.sort_values(["degree", "lane"]).reset_index(drop=True)
    return out


def summarize_full_image_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the fraction of 'fully fitted' images by degree.

    Args:
        df: Results DataFrame with 'is_full_image' column.

    Returns:
        DataFrame with counts and rates per degree.
    """
    groups = df.groupby(GROUP_KEYS, as_index=False).agg(
        is_full_image=("is_full_image", "first")
    )
    out = groups.groupby("degree", as_index=False).agg(
        n_images=("is_full_image", "size"),
        n_full_images=("is_full_image", "sum"),
    )
    out["rate_full_images"] = out["n_full_images"] / out["n_images"]
    out = out.sort_values("degree").reset_index(drop=True)
    return out


def summarize_errors_lane_level(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    only_full: bool,
) -> pd.DataFrame:
    """
    Summarize error metrics over all successful lane fits.

    Args:
        df: Results DataFrame (must contain is_full_image).
        metrics: List of metric column names to summarize.
        only_full: If True, restrict to rows from fully fitted images.

    Returns:
        DataFrame with mean/median/p90/count per degree and metric.
    """
    r = df.copy()
    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712

    rows: List[Dict[str, object]] = []
    for deg, g in r.groupby("degree", dropna=True):
        for m in metrics:
            if m not in g.columns:
                continue
            vals = g[m].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                rows.append(
                    {
                        "degree": int(deg),
                        "metric": m,
                        "n": 0,
                        "mean": np.nan,
                        "median": np.nan,
                        "p90": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "degree": int(deg),
                    "metric": m,
                    "n": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p90": float(np.percentile(vals, 90)),
                }
            )

    out = pd.DataFrame(rows).sort_values(["metric", "degree"]).reset_index(drop=True)
    return out


def summarize_errors_image_level(
    df: pd.DataFrame,
    metric: str,
    *,
    only_full: bool,
) -> pd.DataFrame:
    """
    Summarize an error metric at image level.

    Per image, we compute:
        metric_mean_image = mean(metric over successful lanes in that image)
        n_lanes_ok = count(successful lanes)

    Then we summarize metric_mean_image over images per degree.

    Args:
        df: Results DataFrame (must contain is_full_image).
        metric: Metric column to aggregate (e.g. "rmse_x_px_gt").
        only_full: If True, use only fully fitted images.

    Returns:
        DataFrame with mean/median/p90/count of image-level means per degree.
    """
    r = df.copy()
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712

    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    r = r[np.isfinite(r[metric])]

    if r.empty:
        return pd.DataFrame(
            columns=["degree", "n_images", "mean", "median", "p90", "mean_n_lanes_ok"]
        )

    per_image = r.groupby(GROUP_KEYS, as_index=False).agg(
        metric_mean=(metric, "mean"),
        n_lanes_ok=("lane", "count"),
    )

    out_rows: List[Dict[str, object]] = []
    for deg, g in per_image.groupby("degree", dropna=True):
        vals = g["metric_mean"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out_rows.append(
                {
                    "degree": int(deg),
                    "n_images": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "p90": np.nan,
                    "mean_n_lanes_ok": np.nan,
                }
            )
            continue
        out_rows.append(
            {
                "degree": int(deg),
                "n_images": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90)),
                "mean_n_lanes_ok": float(np.mean(g["n_lanes_ok"].to_numpy(float))),
            }
        )

    out = pd.DataFrame(out_rows).sort_values("degree").reset_index(drop=True)
    return out


def export_worst_cases(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    degrees: List[int],
    *,
    only_full: bool,
    top_k: int = 25,
) -> None:
    """
    Export worst-case lane fits per degree to CSV.

    Args:
        df: Results DataFrame.
        out_dir: Output directory.
        metric: Metric column to rank by.
        degrees: Degrees to export.
        only_full: If True, restrict to fully fitted images.
        top_k: Number of worst rows per degree.
    """
    r = df.copy()
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712
    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    r = r[np.isfinite(r[metric])]

    for deg in degrees:
        sub = r[r["degree"] == deg].copy()
        if sub.empty:
            continue
        worst = sub.nlargest(top_k, metric)
        cols = ["task_name", "image_id", "image_path", "lane", "degree", metric]
        for extra in [
            "n_points_in_raw",
            "n_points_in_used",
            "min_required_points",
            "densify_step_px",
        ]:
            if extra in worst.columns and extra not in cols:
                cols.append(extra)
        worst[cols].to_csv(out_dir / f"worst_cases_{metric}_deg{deg}.csv", index=False)


def plot_coverage_present_vs_degree(cov: pd.DataFrame, out_path: Path) -> None:
    """
    Plot lane-level fit success rate and point-failure rate vs degree.

    Args:
        cov: Coverage summary DataFrame.
        out_path: Where to save the figure (PNG).
    """
    agg = cov.groupby("degree", as_index=False).apply(
        lambda g: pd.Series(
            {
                "n_present": float(g["n_present"].sum()),
                "rate_fit_success": float(
                    (g["n_fit_success"].sum() / g["n_present"].sum())
                    if g["n_present"].sum() > 0
                    else np.nan
                ),
                "rate_fail_points": float(
                    (g["n_present_fail_points"].sum() / g["n_present"].sum())
                    if g["n_present"].sum() > 0
                    else np.nan
                ),
            }
        ),
        include_groups=False,
    )

    degrees = agg["degree"].to_list()
    x = np.arange(len(degrees))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(
        x - 0.2, agg["rate_fit_success"], width=0.4, label="fit_success | lane_present"
    )
    ax.bar(
        x + 0.2,
        agg["rate_fail_points"],
        width=0.4,
        label="fail_due_to_points | lane_present",
    )

    ax.set_xticks(x, [str(d) for d in degrees])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Rate")
    ax.set_title("Lane-level coverage (rates among present lanes)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_full_image_rate(full: pd.DataFrame, out_path: Path) -> None:
    """
    Plot the fraction of fully fitted images vs degree.

    Args:
        full: Full-image summary DataFrame.
        out_path: Where to save the figure (PNG).
    """
    degrees = full["degree"].to_list()
    x = np.arange(len(degrees))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x, full["rate_full_images"])
    ax.set_xticks(x, [str(d) for d in degrees])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Rate")
    ax.set_title("Full image fit rate (all present lanes successfully fitted)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_stats_by_degree(
    stats: pd.DataFrame, out_path: Path, metric: str
) -> None:
    """
    Plot mean/median/p90 of a metric vs degree.

    Args:
        stats: Lane-level error stats DataFrame (degree/metric/mean/median/p90).
        out_path: Where to save the figure.
        metric: Metric name to plot.
    """
    sub = stats[stats["metric"] == metric].copy().sort_values("degree")
    degrees = sub["degree"].to_list()
    x = np.arange(len(degrees))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, sub["mean"], marker="o", label="mean")
    ax.plot(x, sub["median"], marker="o", label="median")
    ax.plot(x, sub["p90"], marker="o", label="p90")

    ax.set_xticks(x, [str(d) for d in degrees])
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(f"{metric} (px)")
    ax.set_title(f"{metric} statistics over successful lane fits")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_boxplot(
    df: pd.DataFrame,
    out_path: Path,
    metric: str,
    *,
    only_full: bool,
) -> None:
    """
    Plot distributions per degree as a boxplot for a given metric.

    Args:
        df: Results DataFrame.
        out_path: Where to save the plot.
        metric: Metric column name (e.g. rmse_x_px_gt).
        only_full: If True, restrict to fully fitted images.
    """
    r = df.copy()
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712
    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    if metric not in r.columns:
        return
    r = r[np.isfinite(r[metric])]

    data: List[np.ndarray] = []
    labels: List[str] = []
    for deg in sorted(r["degree"].dropna().unique()):
        vals = r.loc[r["degree"] == deg, metric].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(str(int(deg)))

    if not data:
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(f"{metric} (px)")
    ax.set_title(f"{metric} distribution over successful lane fits")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Entry point for evaluation of polynomial fitting results."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results", required=True, help="CSV from fit_polynomials (raw or densified)"
    )
    p.add_argument(
        "--out",
        default="artifacts/eval_polynomials",
        help="Base output directory (variant subfolder will be appended: raw/ or densified/)",
    )
    p.add_argument(
        "--degrees",
        nargs="*",
        type=int,
        default=DEGREES_DEFAULT,
        help="Degrees to include (default: 2 3 4 5 6)",
    )
    p.add_argument(
        "--only-full",
        action="store_true",
        help="Restrict error summaries/plots to fully fitted images",
    )
    p.add_argument(
        "--topk", type=int, default=25, help="Worst cases exported per degree"
    )
    args = p.parse_args()

    variant = _infer_variant_from_results_path(args.results)
    out_dir = Path(args.out) / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # We evaluate BOTH point-sets (gt + used).
    # These columns must exist in the CSV; for raw runs, *_gt and *_used are typically identical.
    metric_sets: Dict[str, List[str]] = {
        "gt": ["rmse_x_px_gt", "mae_x_px_gt", "medae_x_px_gt", "maxae_x_px_gt"],
        "used": [
            "rmse_x_px_used",
            "mae_x_px_used",
            "medae_x_px_used",
            "maxae_x_px_used",
        ],
    }

    # Load once with all metric columns we care about
    all_metrics = sorted({m for ms in metric_sets.values() for m in ms})
    df = load_results(args.results, all_metrics)
    df = df[df["degree"].isin(args.degrees)].copy()

    # Add image-level completeness flag
    df = add_full_image_flag(df)

    # Coverage / full image rate (independent of gt/used)
    cov = summarize_coverage_lane_level(df)
    full = summarize_full_image_rate(df)

    cov.to_csv(out_dir / "summary_coverage_lane_level.csv", index=False)
    full.to_csv(out_dir / "summary_full_image_rate.csv", index=False)

    plot_coverage_present_vs_degree(
        cov, out_dir / "fig_coverage_rates_present_vs_degree.png"
    )
    plot_full_image_rate(full, out_dir / "fig_full_image_rate_vs_degree.png")

    # Now do EVERYTHING twice: gt + used
    for tag, metrics in metric_sets.items():
        # Skip set if none of its columns exist (helps during transition)
        existing = [m for m in metrics if m in df.columns]
        if not existing:
            print(f"[eval] warning: no '{tag}' metric columns found, skipping.")
            continue

        lane_stats = summarize_errors_lane_level(df, existing, only_full=args.only_full)
        lane_stats.to_csv(out_dir / f"summary_errors_lane_level_{tag}.csv", index=False)

        # image-level stats for RMSE only (per your previous approach)
        rmse_col = f"rmse_x_px_{tag}"
        img_stats = summarize_errors_image_level(df, rmse_col, only_full=args.only_full)
        img_stats.to_csv(
            out_dir / f"summary_errors_image_level_rmse_{tag}.csv", index=False
        )

        # worst-cases by degree (lane-level)
        export_worst_cases(
            df,
            out_dir,
            metric=rmse_col,
            degrees=args.degrees,
            only_full=args.only_full,
            top_k=args.topk,
        )

        # Plots for each metric stats line-plot
        for m in existing:
            plot_metric_stats_by_degree(
                lane_stats, out_dir / f"fig_{m}_stats_vs_degree.png", metric=m
            )

        # Boxplot for RMSE distribution
        plot_metric_boxplot(
            df,
            out_dir / f"fig_{rmse_col}_boxplot_vs_degree.png",
            metric=rmse_col,
            only_full=args.only_full,
        )

    print(f"[eval] wrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


# Beispielnutzung:
# python /workspace/src/eval_polynomials.py --results /workspace/artifacts/results_poly_raw.csv
# python /workspace/src/eval_polynomials.py --results /workspace/artifacts/results_poly_densified_step2p0px
# python /workspace/src/eval_polynomials.py --results /workspace/artifacts/results_poly_raw.csv --only-full
