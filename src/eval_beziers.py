"""
Evaluate Bezier fitting results (raw or densified).

This script summarizes fit coverage and error metrics (RMSE/MAE/MedAE/MaxAE)
for Bezier lane fits across different numbers of control points.

It exports CSV summaries and saves plots as static images for later reporting
and comparison.

Key features:
- Automatically splits outputs into:
    artifacts/eval_beziers/raw/...
    artifacts/eval_beziers/densified/...
  (inferred from the results CSV filename)
- Computes and exports ALL summaries/plots twice:
    *_gt   : errors against original GT points
    *_used : errors against the points actually used for fitting (raw==used for raw runs)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANE_ORDER = ["left", "center", "right"]
CTRLPTS_DEFAULT = [3, 4, 5, 6]

# one fit-row is one (image, lane, n_control_points)
GROUP_KEYS = [
    "task_name",
    "image_id",
    "image_path",
    "method",
    "variant",
    "n_control_points",
]


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
        results_csv: Path to CSV produced by fit_beziers.py
        metric_cols: Metric columns to coerce to numeric.

    Returns:
        DataFrame with cleaned dtypes for ctrlpoint count / booleans / metrics.
    """
    df = pd.read_csv(results_csv)

    if "n_control_points" in df.columns:
        df["n_control_points"] = pd.to_numeric(
            df["n_control_points"], errors="coerce"
        ).astype("Int64")

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
    """
    present = g["lane_present"].to_numpy(bool)
    success = g["fit_success"].to_numpy(bool)
    return bool(success[present].all())


def add_full_image_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'is_full_image' flag to each row based on group completeness."""
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
    Summarize lane-level coverage statistics by n_control_points and lane.

    Metrics reported:
        - n_total: all lane records
        - n_present: lane_present == True
        - n_present_fail_points: present but not enough points
        - n_present_fit_fail: present + enough points but fit failed
        - n_fit_success: present & fit_success
        - rates computed relative to n_present
    """
    rows: List[Dict[str, object]] = []
    for (m, lane), g in df.groupby(["n_control_points", "lane"], dropna=True):
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
                "n_control_points": int(m),
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
    out = out.sort_values(["n_control_points", "lane"]).reset_index(drop=True)
    return out


def summarize_full_image_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the fraction of 'fully fitted' images by n_control_points."""
    groups = df.groupby(GROUP_KEYS, as_index=False).agg(
        is_full_image=("is_full_image", "first")
    )
    out = groups.groupby("n_control_points", as_index=False).agg(
        n_images=("is_full_image", "size"),
        n_full_images=("is_full_image", "sum"),
    )
    out["rate_full_images"] = out["n_full_images"] / out["n_images"]
    out = out.sort_values("n_control_points").reset_index(drop=True)
    return out


def summarize_errors_lane_level(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    only_full: bool,
) -> pd.DataFrame:
    """
    Summarize error metrics over all successful lane fits.

    Returns mean/median/p90/count per n_control_points and metric.
    """
    r = df.copy()
    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712

    rows: List[Dict[str, object]] = []
    for m, g in r.groupby("n_control_points", dropna=True):
        for met in metrics:
            if met not in g.columns:
                continue
            vals = g[met].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                rows.append(
                    {
                        "n_control_points": int(m),
                        "metric": met,
                        "n": 0,
                        "mean": np.nan,
                        "median": np.nan,
                        "p90": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "n_control_points": int(m),
                    "metric": met,
                    "n": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "p90": float(np.percentile(vals, 90)),
                }
            )

    out = (
        pd.DataFrame(rows)
        .sort_values(["metric", "n_control_points"])
        .reset_index(drop=True)
    )
    return out


def summarize_errors_image_level(
    df: pd.DataFrame,
    metric: str,
    *,
    only_full: bool,
) -> pd.DataFrame:
    """
    Summarize an error metric at image level.

    Per image group we compute:
        metric_mean_image = mean(metric over successful lanes)
        n_lanes_ok = count(successful lanes)

    Then summarize metric_mean_image over images per n_control_points.
    """
    r = df.copy()
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712

    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    if metric not in r.columns:
        return pd.DataFrame(
            columns=[
                "n_control_points",
                "n_images",
                "mean",
                "median",
                "p90",
                "mean_n_lanes_ok",
            ]
        )
    r = r[np.isfinite(r[metric])]

    if r.empty:
        return pd.DataFrame(
            columns=[
                "n_control_points",
                "n_images",
                "mean",
                "median",
                "p90",
                "mean_n_lanes_ok",
            ]
        )

    per_image = r.groupby(GROUP_KEYS, as_index=False).agg(
        metric_mean=(metric, "mean"),
        n_lanes_ok=("lane", "count"),
    )

    out_rows: List[Dict[str, object]] = []
    for m, g in per_image.groupby("n_control_points", dropna=True):
        vals = g["metric_mean"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out_rows.append(
                {
                    "n_control_points": int(m),
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
                "n_control_points": int(m),
                "n_images": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90)),
                "mean_n_lanes_ok": float(np.mean(g["n_lanes_ok"].to_numpy(float))),
            }
        )

    out = pd.DataFrame(out_rows).sort_values("n_control_points").reset_index(drop=True)
    return out


def export_worst_cases(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    ctrl_list: List[int],
    *,
    only_full: bool,
    top_k: int = 25,
) -> None:
    """Export worst-case lane fits per n_control_points to CSV."""
    r = df.copy()
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712
    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    if metric not in r.columns:
        return
    r = r[np.isfinite(r[metric])]

    for m in ctrl_list:
        sub = r[r["n_control_points"] == m].copy()
        if sub.empty:
            continue
        worst = sub.nlargest(top_k, metric)
        cols = [
            "task_name",
            "image_id",
            "image_path",
            "lane",
            "n_control_points",
            metric,
        ]
        for extra in [
            "n_points_in_raw",
            "n_points_in_used",
            "min_required_points",
            "densify_step_px",
        ]:
            if extra in worst.columns and extra not in cols:
                cols.append(extra)
        worst[cols].to_csv(out_dir / f"worst_cases_{metric}_ctrl{m}.csv", index=False)


def plot_coverage_present_vs_ctrlpts(cov: pd.DataFrame, out_path: Path) -> None:
    """Plot lane-level fit success rate and point-failure rate vs n_control_points."""
    agg = cov.groupby("n_control_points", as_index=False).apply(
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

    ctrl = agg["n_control_points"].to_list()
    x = np.arange(len(ctrl))

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

    ax.set_xticks(x, [str(int(m)) for m in ctrl])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Bezier control points")
    ax.set_ylabel("Rate")
    ax.set_title("Lane-level coverage (rates among present lanes)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_full_image_rate(full: pd.DataFrame, out_path: Path) -> None:
    """Plot the fraction of fully fitted images vs n_control_points."""
    ctrl = full["n_control_points"].to_list()
    x = np.arange(len(ctrl))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x, full["rate_full_images"])
    ax.set_xticks(x, [str(int(m)) for m in ctrl])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Bezier control points")
    ax.set_ylabel("Rate")
    ax.set_title("Full image fit rate (all present lanes successfully fitted)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_stats_by_ctrlpts(
    stats: pd.DataFrame, out_path: Path, metric: str
) -> None:
    """Plot mean/median/p90 of a metric vs n_control_points."""
    sub = stats[stats["metric"] == metric].copy().sort_values("n_control_points")
    ctrl = sub["n_control_points"].to_list()
    x = np.arange(len(ctrl))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, sub["mean"], marker="o", label="mean")
    ax.plot(x, sub["median"], marker="o", label="median")
    ax.plot(x, sub["p90"], marker="o", label="p90")

    ax.set_xticks(x, [str(int(m)) for m in ctrl])
    ax.set_xlabel("Bezier control points")
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
    """Plot distributions per n_control_points as a boxplot for a given metric."""
    r = df.copy()
    if only_full:
        r = r[r["is_full_image"] == True]  # noqa: E712
    r = r[(r["lane_present"] == True) & (r["fit_success"] == True)]  # noqa: E712
    if metric not in r.columns:
        return
    r = r[np.isfinite(r[metric])]

    data: List[np.ndarray] = []
    labels: List[str] = []
    for m in sorted(r["n_control_points"].dropna().unique()):
        vals = r.loc[r["n_control_points"] == m, metric].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(str(int(m)))

    if not data:
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_xlabel("Bezier control points")
    ax.set_ylabel(f"{metric} (px)")
    ax.set_title(f"{metric} distribution over successful lane fits")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Executes script with parameters."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results", required=True, help="CSV from fit_beziers.py (raw or densified)"
    )
    p.add_argument(
        "--out",
        default="artifacts/eval_beziers",
        help="Base output directory (variant subfolder will be appended: raw/ or densified/)",
    )
    p.add_argument(
        "--n-control-points",
        nargs="*",
        type=int,
        default=CTRLPTS_DEFAULT,
        help="Control point counts to include (default: 3 4 5 6)",
    )
    p.add_argument(
        "--only-full",
        action="store_true",
        help="Restrict error summaries/plots to fully fitted images",
    )
    p.add_argument(
        "--topk", type=int, default=25, help="Worst cases exported per ctrl-count"
    )
    args = p.parse_args()

    variant = _infer_variant_from_results_path(args.results)
    out_dir = Path(args.out) / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate BOTH point-sets (gt + used).
    metric_sets: Dict[str, List[str]] = {
        "gt": [
            "rmse_dist_px_gt",
            "mae_dist_px_gt",
            "medae_dist_px_gt",
            "maxae_dist_px_gt",
        ],
        "used": [
            "rmse_dist_px_used",
            "mae_dist_px_used",
            "medae_dist_px_used",
            "maxae_dist_px_used",
        ],
    }

    all_metrics = sorted({m for ms in metric_sets.values() for m in ms})
    df = load_results(args.results, all_metrics)
    df = df[df["n_control_points"].isin(args.n_control_points)].copy()

    df = add_full_image_flag(df)

    # Coverage summaries (independent of gt/used)
    cov = summarize_coverage_lane_level(df)
    full = summarize_full_image_rate(df)

    cov.to_csv(out_dir / "summary_coverage_lane_level.csv", index=False)
    full.to_csv(out_dir / "summary_full_image_rate.csv", index=False)

    plot_coverage_present_vs_ctrlpts(
        cov, out_dir / "fig_coverage_rates_present_vs_ctrlpts.png"
    )
    plot_full_image_rate(full, out_dir / "fig_full_image_rate_vs_ctrlpts.png")

    # Do EVERYTHING twice: gt + used
    for tag, metrics in metric_sets.items():
        existing = [m for m in metrics if m in df.columns]
        if not existing:
            print(f"[eval] warning: no '{tag}' metric columns found, skipping.")
            continue

        lane_stats = summarize_errors_lane_level(df, existing, only_full=args.only_full)
        lane_stats.to_csv(out_dir / f"summary_errors_lane_level_{tag}.csv", index=False)

        rmse_col = f"rmse_dist_px_{tag}"
        img_stats = summarize_errors_image_level(df, rmse_col, only_full=args.only_full)
        img_stats.to_csv(
            out_dir / f"summary_errors_image_level_rmse_{tag}.csv", index=False
        )

        export_worst_cases(
            df,
            out_dir,
            metric=rmse_col,
            ctrl_list=args.n_control_points,
            only_full=args.only_full,
            top_k=args.topk,
        )

        for m in existing:
            plot_metric_stats_by_ctrlpts(
                lane_stats, out_dir / f"fig_{m}_stats_vs_ctrlpts.png", metric=m
            )

        plot_metric_boxplot(
            df,
            out_dir / f"fig_{rmse_col}_boxplot_vs_ctrlpts.png",
            metric=rmse_col,
            only_full=args.only_full,
        )

    print(f"[eval] wrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


# Examples:
# python /workspace/src/eval_beziers.py --results /workspace/artifacts/results_bezier_raw.csv
# python /workspace/src/eval_beziers.py --results /workspace/artifacts/results_bezier_densified_step10p0px.csv
# python /workspace/src/eval_beziers.py --results /workspace/artifacts/results_bezier_densified_step10p0px.csv --only-full
