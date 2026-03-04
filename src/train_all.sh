#!/bin/bash

echo "===== TRAIN BEZIER M6 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m6.npz  --out artifacts/train_bezier_m6  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== TRAIN BEZIER M8 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m8.npz  --out artifacts/train_bezier_m8  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== TRAIN BSPLINE M6 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m6.npz --out artifacts/train_bspline_m6 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== TRAIN BSPLINE M8 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m8.npz --out artifacts/train_bspline_m8 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== TRAIN BEZIER M5 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m5.npz  --out artifacts/train_bezier_m5  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== TRAIN BEZIER M4 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m4.npz  --out artifacts/train_bezier_m4  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== TRAIN BSPLINE M5 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m5.npz --out artifacts/train_bspline_m5 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== TRAIN BSPLINE M4 ====="
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m4.npz --out artifacts/train_bspline_m4 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25

echo "===== DONE ====="
