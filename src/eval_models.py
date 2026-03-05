from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------
# helpers
# ----------------------------
def _to_bool_series(s: pd.Series) -> pd.Series:
    """
    Robust conversion of gt_exists/pred_exists columns.
    Handles bool, 0/1, 'True'/'False', 'true'/'false'.
    """
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(int).astype(bool)

    # string-like
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["1", "true", "t", "yes", "y"])


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def _confusion_counts(gt: np.ndarray, pr: np.ndarray) -> Dict[str, int]:
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    tp = int(np.sum(gt & pr))
    fp = int(np.sum(~gt & pr))
    fn = int(np.sum(gt & ~pr))
    tn = int(np.sum(~gt & ~pr))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = float(2 * prec * rec / (prec + rec))
    return prec, rec, f1


def _quantiles(vals: np.ndarray, qs: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        for q in qs:
            out[f"rmse_sym_p{int(q*100)}_px"] = float("nan")
        return out
    for q in qs:
        out[f"rmse_sym_p{int(q*100)}_px"] = float(np.percentile(vals, q * 100))
    return out


def _model_name_from_path(metrics_path: Path) -> str:
    # metrics_path = .../infer/train_bspline_m6/bspline_val/metrics_lane.csv
    return metrics_path.parent.parent.name + "/" + metrics_path.parent.name


def _pretty_name(name: str) -> str:
    name = name.lower()

    if "bezier" in name:
        method = "Bezier"
    elif "bspline" in name:
        method = "B-Spline"
    else:
        method = "Model"

    import re

    m = re.search(r"m(\d+)", name)
    cp = m.group(1) if m else "?"

    return f"{method} ({cp} CP)"


# ----------------------------
# per-model summarization
# ----------------------------
def summarize_model(metrics_lane_csv: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Compute summary statistics for a trained model evaluation.

    Aggregates lane-level metrics such as RMSE, MAE, MedianAE and MaxAE
    and produces a compact summary used for reporting and comparison
    between different model configurations.
    """
    p = Path(metrics_lane_csv)
    df = pd.read_csv(p)

    # Normalize bool columns
    df["gt_exists"] = _to_bool_series(df["gt_exists"])
    df["pred_exists"] = _to_bool_series(df["pred_exists"])

    # --- classification summary (overall) ---
    gt = df["gt_exists"].to_numpy(bool)
    pr = df["pred_exists"].to_numpy(bool)
    cc = _confusion_counts(gt, pr)
    prec, rec, f1 = _precision_recall_f1(cc["tp"], cc["fp"], cc["fn"])

    # --- regression summary (TP only) ---
    ok = df[
        (df["gt_exists"] == True) & (df["pred_exists"] == True)
    ].copy()  # noqa: E712
    vals = ok["rmse_sym_px"].to_numpy(float)
    vals = vals[np.isfinite(vals)]

    model = _model_name_from_path(p)

    row = {
        "model": model,
        "metrics_path": str(p),
        "n_rows": int(len(df)),
        "n_tp_lanes": int(vals.size),  # lanes used for geo stats
        "rmse_sym_mean_px": float(np.mean(vals)) if vals.size else float("nan"),
        "rmse_sym_median_px": float(np.median(vals)) if vals.size else float("nan"),
        "rmse_sym_std_px": float(np.std(vals)) if vals.size else float("nan"),
        "rmse_sym_min_px": float(np.min(vals)) if vals.size else float("nan"),
        "rmse_sym_max_px": float(np.max(vals)) if vals.size else float("nan"),
        **_quantiles(vals, qs=[0.75, 0.90, 0.95, 0.99]),
        # classification
        "tp": cc["tp"],
        "fp": cc["fp"],
        "fn": cc["fn"],
        "tn": cc["tn"],
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    # Also compute per-lane classification table (for optional CSV)
    per_lane_rows = []
    for lane, g in df.groupby("lane"):
        gt_l = g["gt_exists"].to_numpy(bool)
        pr_l = g["pred_exists"].to_numpy(bool)
        cc_l = _confusion_counts(gt_l, pr_l)
        prec_l, rec_l, f1_l = _precision_recall_f1(cc_l["tp"], cc_l["fp"], cc_l["fn"])
        per_lane_rows.append(
            {
                "model": model,
                "lane": lane,
                "tp": cc_l["tp"],
                "fp": cc_l["fp"],
                "fn": cc_l["fn"],
                "tn": cc_l["tn"],
                "precision": prec_l,
                "recall": rec_l,
                "f1": f1_l,
                "support_gt_pos": int(np.sum(gt_l)),
                "support_pred_pos": int(np.sum(pr_l)),
            }
        )

    df_per_lane = pd.DataFrame(per_lane_rows)
    return row, df_per_lane


# ----------------------------
# plotting
# ----------------------------
def plot_grouped_rmse(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Plot grouped RMSE statistics across model configurations.

    Creates a comparison plot showing mean RMSE values grouped by
    method, control point count, or dataset variant.
    Used to visually compare model performance.
    """
    # grouped bars: mean, median, p90
    fig = plt.figure()
    ax = fig.add_subplot(111)

    models = summary["model"].tolist()
    mean = summary["rmse_sym_mean_px"].to_numpy(float)
    median = summary["rmse_sym_median_px"].to_numpy(float)
    p90 = (
        summary["rmse_sym_p90_px"].to_numpy(float)
        if "rmse_sym_p90_px" in summary.columns
        else summary["rmse_sym_p90_px".lower()].to_numpy(float)
    )
    models_pretty = [
        f"{_pretty_name(m)}\nμ={mean[i]:.2f}px" for i, m in enumerate(models)
    ]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, mean, width=width, label="Mean")
    ax.bar(x, median, width=width, label="Median")
    ax.bar(x + width, p90, width=width, label="P90")

    ax.set_xticks(x, models_pretty, fontsize=6)
    ax.set_ylabel("Symmetric curve RMSE (px)")
    ax.set_title("RMSE (sym) summary on validation set")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_boxplot_rmse(tp_vals: Dict[str, np.ndarray], out_path: Path) -> None:
    """
    Create boxplot visualization of RMSE distributions.

    Displays the distribution of symmetric RMSE errors for each
    model configuration to highlight variance and outliers.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --- compute mean rmse per model ---
    sort_items = []
    for k, v in tp_vals.items():
        vals = v[np.isfinite(v)]
        mean = np.mean(vals) if vals.size else np.inf
        sort_items.append((k, mean))

    # --- sort by rmse (best model left) ---
    sort_items.sort(key=lambda x: x[1])

    labels = [k for k, _ in sort_items]
    labels_pretty = [
        f"{_pretty_name(k)}\nμ={np.mean(tp_vals[k]):.2f}px" for k in labels
    ]
    data = [tp_vals[k][np.isfinite(tp_vals[k])] for k in labels]

    ax.boxplot(data, labels=labels_pretty, showfliers=True)

    ax.set_ylabel("Symmetric curve RMSE (px)")
    ax.set_title("RMSE (sym) distribution (TP lanes only)")
    ax.tick_params(axis="x", labelsize=6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_hist_overlay(
    a_name: str, a_vals: np.ndarray, b_name: str, b_vals: np.ndarray, out_path: Path
) -> None:
    """
    Plot overlaid histograms of RMSE distributions.

    Allows comparison of multiple models by overlaying their
    RMSE error distributions in a single histogram plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    a = a_vals[np.isfinite(a_vals)]
    b = b_vals[np.isfinite(b_vals)]

    # Freedman–Diaconis-ish binning fallback
    allv = np.concatenate([a, b]) if (a.size and b.size) else (a if a.size else b)
    if allv.size == 0:
        return
    # bins: keep it readable
    bins = 40

    ax.hist(a, bins=bins, alpha=0.5, label=a_name)
    ax.hist(b, bins=bins, alpha=0.5, label=b_name)

    ax.set_xlabel("Symmetric curve RMSE (px)")
    ax.set_ylabel("Count")
    ax.set_title("RMSE (sym) histogram overlay (TP lanes only)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_f1(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Compute and visualize F1 score for lane existence prediction.

    Evaluates the binary lane existence classification performance
    and generates plots summarizing precision, recall and F1 score.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    models = summary["model"].tolist()
    f1 = summary["f1"].to_numpy(float)

    x = np.arange(len(models))
    ax.bar(x, f1)
    ax.set_xticks(x, models, rotation=18, ha="right")
    ax.set_ylabel("F1 score (existence)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Existence classification (F1)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ----------------------------
# main
# ----------------------------
def main():
    """
    Entry point for model evaluation script.

    Loads inference results, computes summary metrics and
    generates plots and CSV outputs used for analysis
    and reporting of lane detection performance.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="paths to metrics_lane.csv (one per model)",
    )
    ap.add_argument("--out", default="artifacts/infer/compare")
    ap.add_argument(
        "--make-hist",
        action="store_true",
        help="also create histogram overlay (best bspline vs best bezier)",
    )
    ap.add_argument(
        "--make-f1-plot", action="store_true", help="also create an F1 bar plot"
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summaries
    summary_rows = []
    per_lane_tables = []

    # Also collect TP RMSE arrays for boxplot + hist
    tp_rmse: Dict[str, np.ndarray] = {}

    for mp in args.metrics:
        row, df_per_lane = summarize_model(mp)
        summary_rows.append(row)
        per_lane_tables.append(df_per_lane)

        # load RMSE values for TP lanes
        df = pd.read_csv(mp)
        df["gt_exists"] = _to_bool_series(df["gt_exists"])
        df["pred_exists"] = _to_bool_series(df["pred_exists"])
        ok = df[
            (df["gt_exists"] == True) & (df["pred_exists"] == True)
        ].copy()  # noqa: E712
        vals = ok["rmse_sym_px"].to_numpy(float)
        tp_rmse[row["model"]] = vals

    summary = pd.DataFrame(summary_rows).sort_values("rmse_sym_mean_px")
    summary.to_csv(out_dir / "summary_models.csv", index=False)

    per_lane = pd.concat(per_lane_tables, ignore_index=True)
    per_lane.to_csv(out_dir / "summary_existence_per_lane.csv", index=False)

    # convenience: also write a compact existence-only table
    existence_cols = ["model", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]
    summary[existence_cols].to_csv(
        out_dir / "summary_existence_overall.csv", index=False
    )

    # Main plots
    # 1) grouped bar: mean/median/p90
    plot_grouped_rmse(summary, out_dir / "fig_rmse_grouped_mean_median_p90.png")

    # 2) boxplot of TP rmse distributions
    plot_boxplot_rmse(tp_rmse, out_dir / "fig_rmse_boxplot_tp.png")

    # 3) optional histogram overlay: best bspline vs best bezier (by mean RMSE)
    if args.make_hist:
        # pick best bspline and best bezier by mean RMSE
        bs = summary[
            summary["model"].str.contains("bspline", case=False, na=False)
        ].copy()
        bz = summary[
            summary["model"].str.contains("bezier", case=False, na=False)
        ].copy()
        if len(bs) and len(bz):
            best_bs = bs.iloc[0]["model"]
            best_bz = bz.iloc[0]["model"]
            plot_hist_overlay(
                best_bs,
                tp_rmse[best_bs],
                best_bz,
                tp_rmse[best_bz],
                out_dir / "fig_rmse_hist_overlay_best_bspline_vs_best_bezier.png",
            )

    # 4) optional F1 plot
    if args.make_f1_plot:
        plot_f1(summary, out_dir / "fig_existence_f1.png")

    print(f"[eval_models] wrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


"""
Usage:

python src/eval_models.py --metrics \
  artifacts/infer/train_bezier_m6/bezier_val/metrics_lane.csv \
  artifacts/infer/train_bezier_m8/bezier_val/metrics_lane.csv \
  artifacts/infer/train_bspline_m6/bspline_val/metrics_lane.csv \
  artifacts/infer/train_bspline_m8/bspline_val/metrics_lane.csv \
  --make-hist --make-f1-plot

#Nachdem nun auch M4/5 betrachtet wird:

python src/eval_models.py --metrics \
  artifacts/infer/train_bezier_m4/bezier_val/metrics_lane.csv \
  artifacts/infer/train_bezier_m5/bezier_val/metrics_lane.csv \
  artifacts/infer/train_bezier_m6/bezier_val/metrics_lane.csv \
  artifacts/infer/train_bezier_m8/bezier_val/metrics_lane.csv \
  artifacts/infer/train_bspline_m4/bspline_val/metrics_lane.csv \
  artifacts/infer/train_bspline_m5/bspline_val/metrics_lane.csv \
  artifacts/infer/train_bspline_m6/bspline_val/metrics_lane.csv \
  artifacts/infer/train_bspline_m8/bspline_val/metrics_lane.csv \
  --make-hist --make-f1-plot
"""
