# lane-detection-lunzer

Development of a new lane detection methodology for the *smarty* project.

This repository contains the full pipeline of a study work (Studienarbeit) on lane detection. The workflow covers everything from parsing CVAT polyline annotations, fitting classical curve models (polynomials, Bezier curves, B-splines) to lane points, generating compact ground-truth files for neural networks, training control-point regression networks, and finally running and evaluating inference on validation data and video sequences.

The goal of this README is to make every result reproducible. Follow the sections in order — each step writes artifacts to disk that later steps consume.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Data Layout (required folder structure)](#data-layout-required-folder-structure)
4. [Quick Smoke Tests](#quick-smoke-tests)
5. [End-to-End Pipeline](#end-to-end-pipeline)
   - [Stage 1 — Classical Curve Fitting](#stage-1--classical-curve-fitting)
   - [Stage 2 — Evaluating Classical Fits](#stage-2--evaluating-classical-fits)
   - [Stage 3 — Visualizing Classical Fits](#stage-3--visualizing-classical-fits)
   - [Stage 4 — Generating NN Ground Truth](#stage-4--generating-nn-ground-truth)
   - [Stage 5 — Training the Networks](#stage-5--training-the-networks)
   - [Stage 6 — Inference & Per-Lane Metrics](#stage-6--inference--per-lane-metrics)
   - [Stage 7 — Model Comparison](#stage-7--model-comparison)
   - [Stage 8 — Qualitative Inspection](#stage-8--qualitative-inspection)
   - [Stage 9 — Video Inference](#stage-9--video-inference)
6. [Optional: Live Image Streaming Server](#optional-live-image-streaming-server)
7. [Helper Scripts](#helper-scripts)
8. [Output Directory Map](#output-directory-map)
9. [License](#license)

---

## Repository Structure

```
.
├── LICENSE
├── README.md
├── concat_proj.py                  # utility: concatenates the whole repo into one txt file
├── data folder.txt                 # notes on the expected data layout
└── src/
    ├── fitting/                    # shared library used by all fitting/visualization scripts
    │   ├── common_helpers.py       # densification, polyline math, error metrics, point parsing
    │   ├── fit_polynomial.py       # polynomial least-squares (x as a function of y)
    │   ├── img_stream_server.py    # FastAPI/SSE server for live image streaming
    │   ├── load_data.py            # CVAT XML parser + DataParser (defines `get_current_data()`)
    │   ├── plot_label.py           # matplotlib/plotly plotting of GT lanes
    │   └── visualize.py            # OpenCV drawing helpers + push-to-stream client
    │
    ├── fit_polynomials.py          # Stage 1a: fit polynomials, write results CSV
    ├── fit_beziers.py              # Stage 1b: fit Bezier curves, write results CSV
    ├── fit_bsplines.py             # Stage 1c: fit cubic B-splines, write results CSV
    │
    ├── eval_polynomials.py         # Stage 2a: aggregate metrics + plots over polynomial results
    ├── eval_beziers.py             # Stage 2b: aggregate metrics + plots over Bezier results
    ├── eval_bsplines.py            # Stage 2c: aggregate metrics + plots over B-spline results
    │
    ├── vis_polynomials.py          # Stage 3a: best/worst image renderings, streaming
    ├── vis_beziers.py              # Stage 3b: same for Bezier fits
    ├── vis_bsplines.py             # Stage 3c: same for B-spline fits
    │
    ├── generate_newGT_from_results.py  # Stage 4: turn fit CSVs into compact NN ground truth (CSV + NPZ)
    │
    ├── train_lane_ctrlpoints.py    # Stage 5: train ResNet-based control-point regressor
    ├── train_all.sh                # Stage 5 helper: trains all 8 networks in sequence
    │
    ├── infer_export_nn.py          # Stage 6: run inference + write metrics_lane.csv / preds.npz
    ├── eval_models.py              # Stage 7: cross-model comparison + summary plots
    ├── infer_vis_ctrl.py           # Stage 8a: side-by-side GT/pred rendering on validation images
    ├── infer_export_outliers.py    # Stage 8b: export best / worst / mid / FP / FN cases
    ├── infer_export_video_nn.py    # Stage 9: run a trained model on a video / frame folder
    │
    ├── test_loading_data.py        # smoke test: dump cleaned dataset to xlsx
    ├── test_all_img_exist.py       # smoke test: verify every image path resolves on disk
    └── test_plot_labels.py         # smoke test: plot one random labeled sample
```

All scripts expect to be run **from the repository root** (so that relative paths like `artifacts/...` resolve correctly).

---

## Setup

Tested with Python 3.10+.

```bash
git clone <this-repo-url>
cd lane-detection-lunzer

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install \
    numpy pandas matplotlib plotly \
    opencv-python pillow scipy \
    torch torchvision \
    fastapi uvicorn \
    openpyxl
```

If you want to encode videos to MP4 in [Stage 9](#stage-9--video-inference), `ffmpeg` must be available on `PATH`.

GPU training requires a working CUDA-enabled PyTorch install. CPU training works but is slow.

Create the artifacts directory once:

```bash
mkdir -p artifacts
```

---

## Data Layout (required folder structure)

`src/fitting/load_data.py` resolves the dataset relative to the repository root, at the directory `data/`. Each top-level subfolder corresponds to one **CVAT task** (named by year-month) and must contain two subfolders:

- `img/` — image frames, grouped per rosbag sub-recording
- `label/annotations.xml` — exported CVAT annotation file

```
data/
├── 2023-05/
│   ├── img/
│   │   ├── 2023-05-24-17-46-49/
│   │   │   ├── 2023-05-24-17-46-49_frame000001.png
│   │   │   ├── 2023-05-24-17-46-49_frame000002.png
│   │   │   └── ...
│   │   └── 2023-05-24-17-47-27/
│   │       └── ...
│   └── label/
│       └── annotations.xml
│
├── 2023-06/
│   ├── img/
│   │   └── 2023-06-28-15-07-49/...
│   └── label/
│       └── annotations.xml
│
├── 2023-10/
│   ├── img/...
│   └── label/annotations.xml
│
├── 2023-12/
│   ├── img/...
│   └── label/annotations.xml
│
└── 2025-05/
    ├── img/...
    └── label/annotations.xml
```

**Conventions used by the loader:**

- Each folder name maps to a CVAT *task* (label-server task export).
- The image frames inside `img/<rosbag-id>/` are downloaded from the NAS at `Daten/cvat/rosbags/<rosbag-id>/`.
- `annotations.xml` is the CVAT export of all *completed* jobs of the task.
- The default loader (`get_current_data()`) loads tasks `2023-05`, `2023-06`, `2023-10`, `2023-12`, `2025-05` and excludes jobs `[119, 120, 121, 111, 112, 113, 114, 115, 116, 117]`. Adjust these two lists in `src/fitting/load_data.py` if you want to use a different subset.
- Rows are kept only if **at least two of three lane columns** (`left lane`, `center lane`, `right lane`) contain valid polyline points — images with insufficient annotations are dropped automatically.

---

## Quick Smoke Tests

Before running the full pipeline it is worth verifying that the data is reachable:

```bash
# 1. Parse all CVAT tasks and dump the cleaned dataframe to xlsx
python src/test_loading_data.py
# -> writes artifacts/current_data.xlsx

# 2. Check that every image path referenced by the dataset actually exists on disk
python src/test_all_img_exist.py

# 3. Plot one random labeled sample (also pushes to the optional stream server)
python src/test_plot_labels.py
```

---

## End-to-End Pipeline

The pipeline produces two kinds of results that the Studienarbeit reports on:

- **Classical curve-fitting results** (polynomials / Bezier / B-spline directly fitted to GT polylines) — Stages 1–3.
- **Neural network results** (a ResNet regresses control points; the curve is reconstructed from them) — Stages 4–9.

The two halves connect at Stage 4: the fitted control points become the ground truth that the network is trained on.

### Stage 1 — Classical Curve Fitting

Each script reads the dataset via `get_current_data()` and writes one row per `(image, lane, parameter)` to a results CSV under `artifacts/`. The `--densify-step-px` flag inserts interpolated points between sparse GT vertices so the least-squares fit is not dominated by long straight segments.

**Polynomials** (degrees 2…6, fitted as `x = f(y)`):

```bash
# raw GT points
python src/fit_polynomials.py
# -> artifacts/results_poly_raw.csv

# densified to ~10 px point spacing
python src/fit_polynomials.py --densify-step-px 10
# -> artifacts/results_poly_densified_step10p0px.csv
```

**Bezier** (parametric Bezier with chord-length parameterization; m = number of control points, degree = m − 1):

```bash
python src/fit_beziers.py
# -> artifacts/results_bezier_raw.csv

python src/fit_beziers.py --densify-step-px 2
# -> artifacts/results_bezier_densified_step2px.csv

# full sweep used in the study
python src/fit_beziers.py --densify-step-px 10 --n-control-points 3 4 5 6 7 8 9 10
# -> artifacts/results_bezier_densified_step10p0px.csv
```

**Cubic clamped B-splines** (fixed degree 3, sweep over m):

```bash
python src/fit_bsplines.py
# -> artifacts/results_bspline_raw.csv

python src/fit_bsplines.py --densify-step-px 10
# -> artifacts/results_bspline_densified_step10p0px.csv

python src/fit_bsplines.py --densify-step-px 10 --n-control-points 4 5 6 7 8 9 10 --degree 3
# -> artifacts/results_bspline_densified_step10p0px.csv  (overwrites the above)
```

Each row of these CSVs carries: task / image metadata, lane name, the fitted coefficients or control points (`coef_c0…c6` for polynomials, `ctrl_p0_x/y … ctrl_pN_x/y` for Bezier and B-spline), and four error metrics (`rmse / mae / medae / maxae`) computed twice — once against the points used for fitting (`_used`) and once against the raw GT points (`_gt`).

### Stage 2 — Evaluating Classical Fits

These scripts aggregate the per-row Stage 1 CSVs into per-degree / per-control-point summaries, coverage tables, and plots.

```bash
# Polynomials
python src/eval_polynomials.py --results artifacts/results_poly_raw.csv
python src/eval_polynomials.py --results artifacts/results_poly_densified_step10p0px.csv
python src/eval_polynomials.py --results artifacts/results_poly_raw.csv --only-full

# Bezier
python src/eval_beziers.py --results artifacts/results_bezier_raw.csv
python src/eval_beziers.py --results artifacts/results_bezier_densified_step10p0px.csv
python src/eval_beziers.py --results artifacts/results_bezier_densified_step10p0px.csv --only-full

# B-spline
python src/eval_bsplines.py --results artifacts/results_bspline_raw.csv
python src/eval_bsplines.py --results artifacts/results_bspline_densified_step10p0px.csv
python src/eval_bsplines.py --results artifacts/results_bspline_densified_step10p0px.csv --only-full
```

Outputs go to `artifacts/eval_polynomials/<raw|densified>/...`, `artifacts/eval_beziers/...`, `artifacts/eval_bsplines/...` and include summary CSVs, line plots, boxplots, and "worst case" CSVs by metric/degree.

`--only-full` restricts the aggregation to images where all three lanes were fittable.

### Stage 3 — Visualizing Classical Fits

These scripts render the fitted curves on top of the original images. They can either export PNG batches of best/worst cases or stream live frames to the optional [streaming server](#optional-live-image-streaming-server).

```bash
# Polynomials — best/worst export
python src/vis_polynomials.py --results artifacts/results_poly_raw.csv --only-full
python src/vis_polynomials.py --results artifacts/results_poly_densified_step10p0px.csv --only-full

# stream a degree across the dataset
python src/vis_polynomials.py --results artifacts/results_poly_raw.csv --only-full --stream --degree 3

# compare raw vs densified for the same degree
python src/vis_polynomials.py --compare \
    --results-raw artifacts/results_poly_raw.csv \
    --results-dens artifacts/results_poly_densified_step10p0px.csv \
    --rank-by delta --only-full --stream --degree 3

# render all degrees of one specific image
python src/vis_polynomials.py \
    --results artifacts/results_poly_raw.csv \
    --specific-image data/2023-05/img/2023-05-24-17-54-35/2023-05-24-17-54-35_frame000245.png
```

```bash
# Bezier
python src/vis_beziers.py --results artifacts/results_bezier_densified_step10p0px.csv --only-full
python src/vis_beziers.py --results artifacts/results_bezier_densified_step10p0px.csv --stream --mode order --n-control-points 4 --interval 1
python src/vis_beziers.py --results artifacts/results_bezier_densified_step10p0px.csv \
    --specific-image data/2023-05/img/2023-05-24-17-54-35/2023-05-24-17-54-35_frame000245.png \
    --ctrl-list 4 6 8
```

```bash
# B-spline
python src/vis_bsplines.py --results artifacts/results_bspline_densified_step10p0px.csv --only-full
python src/vis_bsplines.py --results artifacts/results_bspline_densified_step10p0px.csv --stream --mode order --n-control-points 4 --interval 1
python src/vis_bsplines.py --results artifacts/results_bspline_densified_step10p0px.csv \
    --specific-image data/2023-05/img/2023-05-24-17-54-35/2023-05-24-17-54-35_frame000245.png \
    --ctrl-list 4 6 8
```

### Stage 4 — Generating NN Ground Truth

The network does not consume CVAT XML directly. Instead it consumes the fitted control points from Stage 1 — `generate_newGT_from_results.py` reduces a fit results CSV to one row per image with normalized control points and writes a compact NPZ that the trainer loads. Control points are normalized to `[0, 1]` using the original image size (default `2064 × 1544`).

Generate one GT file per (method, m) combination you want to train:

```bash
# Bezier, m=6
python src/generate_newGT_from_results.py \
    --results artifacts/results_bezier_densified_step10p0px.csv \
    --out artifacts/gt_bezier_m6.csv \
    --out-npz artifacts/gt_bezier_m6.npz \
    --n-control-points 6 \
    --method bezier \
    --require-fit-success

# Bezier, m=8
python src/generate_newGT_from_results.py \
    --results artifacts/results_bezier_densified_step10p0px.csv \
    --out artifacts/gt_bezier_m8.csv \
    --out-npz artifacts/gt_bezier_m8.npz \
    --n-control-points 8 \
    --method bezier \
    --require-fit-success

# B-spline, m=6
python src/generate_newGT_from_results.py \
    --results artifacts/results_bspline_densified_step10p0px.csv \
    --out artifacts/gt_bspline_m6.csv \
    --out-npz artifacts/gt_bspline_m6.npz \
    --n-control-points 6 \
    --method bspline \
    --require-fit-success

# B-spline, m=8
python src/generate_newGT_from_results.py \
    --results artifacts/results_bspline_densified_step10p0px.csv \
    --out artifacts/gt_bspline_m8.csv \
    --out-npz artifacts/gt_bspline_m8.npz \
    --n-control-points 8 \
    --method bspline \
    --require-fit-success
```

Repeat for m = 4, 5 if you also want the smaller-m sweep used later in the study. Use `--require-fit-success` so lanes where the classical fit failed are treated as missing rather than carrying garbage values into the training target.

Each NPZ contains:

- `image_path` — `(N,)` absolute paths
- `lane_present` — `(N, 3)` boolean existence flags for left/center/right
- `ctrl_points` — `(N, 3, m, 2)` normalized control points

### Stage 5 — Training the Networks

`train_lane_ctrlpoints.py` trains a ResNet-based regressor that, for an input image of size `--img-w × --img-h`, outputs both an existence probability per lane and `m` normalized control points per lane. `--ctrl-margin 0.25` allows control points to sit slightly outside the image frame (range `[-0.25, 1.25]`), which matters near the image borders.

Train all eight networks individually:

```bash
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m6.npz  --out artifacts/train_bezier_m6  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m8.npz  --out artifacts/train_bezier_m8  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m6.npz --out artifacts/train_bspline_m6 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m8.npz --out artifacts/train_bspline_m8 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
```

Or run the full m = 4, 5, 6, 8 sweep with the convenience script:

```bash
bash src/train_all.sh
```

Each training run writes `best.pt`, `last.pt`, and intermediate logs to its `--out` directory. The checkpoints embed everything the inference scripts need (`m`, `img_w`, `img_h`, `ctrl_margin`, the originating `npz` path).

### Stage 6 — Inference & Per-Lane Metrics

`infer_export_nn.py` runs each trained checkpoint over the validation split of the corresponding NPZ, computes symmetric point-to-curve errors between predicted and GT curves, and writes:

- `preds.npz` — frozen predictions used by all downstream visualization tools
- `metrics_lane.csv` — one row per `(image, lane)` with `gt_exists`, `pred_exists`, `rmse_sym_px`, …
- `metrics_image.csv` — per-image aggregation
- `summary.json` — compact text summary

Inference for the main four models:

```bash
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m6/best.pt  --method bezier  --split val
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m8/best.pt  --method bezier  --split val
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m6/best.pt --method bspline --split val
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m8/best.pt --method bspline --split val
```

For the extended sweep (m = 4, 5), force evaluation in the original image pixel space:

```bash
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m4/best.pt  --method bezier  --split val --eval-space orig
python src/infer_export_nn.py --ckpt artifacts/train_bezier_m5/best.pt  --method bezier  --split val --eval-space orig
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m4/best.pt --method bspline --split val --eval-space orig
python src/infer_export_nn.py --ckpt artifacts/train_bspline_m5/best.pt --method bspline --split val --eval-space orig
```

Outputs land in `artifacts/infer/<train_dir_name>/<method>_<split>/`.

### Stage 7 — Model Comparison

Once every model has its own `metrics_lane.csv`, `eval_models.py` consolidates them into a single comparison table, classification metrics (precision / recall / F1 of lane-existence prediction), grouped RMSE bar charts, boxplots, and (optionally) an overlay histogram of the best Bezier vs the best B-spline.

Main comparison (m = 6 and m = 8 only):

```bash
python src/eval_models.py --metrics \
    artifacts/infer/train_bezier_m6/bezier_val/metrics_lane.csv \
    artifacts/infer/train_bezier_m8/bezier_val/metrics_lane.csv \
    artifacts/infer/train_bspline_m6/bspline_val/metrics_lane.csv \
    artifacts/infer/train_bspline_m8/bspline_val/metrics_lane.csv \
    --make-hist --make-f1-plot
```

Full sweep (m = 4, 5, 6, 8):

```bash
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
```

Outputs land in the directory passed to `--out` (default printed to stdout) and contain `summary_models.csv`, `summary_existence_*.csv`, and the plots used in the report.

### Stage 8 — Qualitative Inspection

**Side-by-side GT/prediction renderings** for `--n` samples of a split, optionally streamed live:

```bash
python src/infer_vis_ctrl.py --ckpt artifacts/train_bezier_m6/best.pt  --method bezier  --split val --draw-mode orig --n 120
python src/infer_vis_ctrl.py --ckpt artifacts/train_bezier_m6/best.pt  --method bezier  --split val --draw-mode orig --n 300 --stream --interval 3
python src/infer_vis_ctrl.py --ckpt artifacts/train_bspline_m6/best.pt --method bspline --split val --draw-mode orig --n 300 --stream --interval 3
```

**Failure case / outlier galleries** built directly from the frozen `preds.npz` of Stage 6:

```bash
# Worst 20 lanes by symmetric RMSE
python src/infer_export_outliers.py --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
    --method bspline --mode worst --k 20 --draw-mode orig

# Best 20
python src/infer_export_outliers.py --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
    --method bspline --mode best --k 20 --draw-mode orig

# Median ±5% band
python src/infer_export_outliers.py --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
    --method bspline --mode mid --mid-frac 0.05 --k 20 --draw-mode orig

# False positives (pred=1, gt=0), sorted by highest predicted probability
python src/infer_export_outliers.py --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
    --method bspline --mode fp --k 20 --draw-mode orig

# False negatives (gt=1, pred=0), sorted by lowest predicted probability
python src/infer_export_outliers.py --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
    --method bspline --mode fn --k 20 --draw-mode orig

# Filter to one lane
python src/infer_export_outliers.py --infer-dir artifacts/infer/train_bspline_m6/bspline_val \
    --method bspline --mode worst --k 20 --lane center --draw-mode orig
```

### Stage 9 — Video Inference

Run any trained checkpoint over a video file or a folder of frames and write an annotated MP4. Requires `ffmpeg` when `--encode ffmpeg` is used.

From a folder of frames:

```bash
python src/infer_export_video_nn.py \
    --ckpt artifacts/train_bspline_m4/best.pt \
    --method bspline \
    --frames /workspace/artifacts/infer_videodemo/input/xyz \
    --out /workspace/artifacts/infer_videodemo/output/xyz.mp4 \
    --fps 20 \
    --exist-thr 0.5 \
    --draw-ctrl \
    --encode ffmpeg
```

From an existing video file:

```bash
python src/infer_export_video_nn.py \
    --ckpt artifacts/train_bspline_m4/best.pt \
    --method bspline \
    --video /workspace/artifacts/infer_videodemo/input/xyz.mp4 \
    --out /workspace/artifacts/infer_videodemo/output/2024-04-04-11-57-06.mp4 \
    --fps 20 \
    --exist-thr 0.5 \
    --encode ffmpeg
```

---

## Optional: Live Image Streaming Server

Several scripts (`--stream`, the test scripts, etc.) push image frames over HTTP to a lightweight FastAPI + SSE server defined in `src/fitting/img_stream_server.py`. The server keeps the latest frame in memory and notifies connected browsers, which then fetch it. This is handy for watching the dataset / fits / inference scroll by in real time without writing thousands of PNGs.

Start it in a separate terminal:

```bash
python src/fitting/img_stream_server.py
# listens on http://0.0.0.0:8000
```

Then open `http://localhost:8000/` in a browser and run any `--stream` command.

---

## Helper Scripts

- `concat_proj.py` — concatenates the entire project (source + selected text files) into one big `project_combined.txt`. Useful for shipping the whole codebase as a single attachment.

  ```bash
  python concat_proj.py
  python concat_proj.py --root . --include "**.py"
  python concat_proj.py --list-extensions
  ```

---

## Output Directory Map

Once the full pipeline has run, `artifacts/` looks roughly like:

```
artifacts/
├── current_data.xlsx                              # from test_loading_data.py

├── results_poly_raw.csv                           # Stage 1 (polynomial)
├── results_poly_densified_step10p0px.csv
├── results_bezier_raw.csv                         # Stage 1 (Bezier)
├── results_bezier_densified_step10p0px.csv
├── results_bspline_raw.csv                        # Stage 1 (B-spline)
├── results_bspline_densified_step10p0px.csv

├── eval_polynomials/{raw,densified}/...           # Stage 2
├── eval_beziers/{raw,densified}/...
├── eval_bsplines/{raw,densified}/...

├── vis_polynomials/{raw,densified}/...            # Stage 3
├── vis_beziers/{raw,densified}/...
├── vis_bsplines/{raw,densified}/...

├── gt_bezier_m{4,5,6,8}.csv                       # Stage 4
├── gt_bezier_m{4,5,6,8}.npz
├── gt_bspline_m{4,5,6,8}.csv
├── gt_bspline_m{4,5,6,8}.npz

├── train_bezier_m{4,5,6,8}/{best.pt,last.pt,...}  # Stage 5
├── train_bspline_m{4,5,6,8}/{best.pt,last.pt,...}

├── infer/                                         # Stage 6
│   ├── train_bezier_m6/bezier_val/{preds.npz,metrics_lane.csv,metrics_image.csv,summary.json}
│   └── ...

├── <eval_models output dir>/                      # Stage 7
│   ├── summary_models.csv
│   ├── summary_existence_overall.csv
│   ├── summary_existence_per_lane.csv
│   ├── fig_rmse_grouped_mean_median_p90.png
│   ├── fig_rmse_boxplot_tp.png
│   └── fig_existence_f1.png

└── infer_videodemo/                               # Stage 9
    ├── input/<rosbag-id>/...
    └── output/<rosbag-id>.mp4
```

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
