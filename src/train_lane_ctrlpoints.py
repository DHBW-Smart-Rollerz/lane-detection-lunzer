from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training.

    Initializes Python, NumPy and PyTorch random generators
    to ensure deterministic experiment behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset
# ----------------------------
class LaneCtrlDataset(Dataset):
    """
    Loads the following data.
      - image_path: (N,) strings
      - lane_present: (N,3) bool
      - ctrl_points: (N,3,m,2) float32 (normalized, may exceed [0,1] if ctrl points are out-of-frame)
    """

    def __init__(
        self,
        npz_path: str,
        img_size: Tuple[int, int],
        *,
        augment: bool = False,
        ctrl_margin: float = 0.25,
        clamp_gt: bool = True,
        verbose_stats: bool = True,
    ):
        """
        Initialize lane control point dataset.

        Loads image paths, lane existence labels and normalized
        control points from the provided NPZ dataset file.
        """
        z = np.load(npz_path, allow_pickle=True)
        self.image_path = z["image_path"].astype(str)
        self.lane_present = z["lane_present"].astype(np.bool_)
        self.ctrl_points = z["ctrl_points"].astype(np.float32)

        # infer m
        if (
            self.ctrl_points.ndim != 4
            or self.ctrl_points.shape[1] != 3
            or self.ctrl_points.shape[3] != 2
        ):
            raise RuntimeError(
                f"ctrl_points has unexpected shape: {self.ctrl_points.shape}"
            )
        self.m = int(self.ctrl_points.shape[2])

        self.w, self.h = int(img_size[0]), int(img_size[1])
        self.augment = bool(augment)

        self.ctrl_margin = float(ctrl_margin)
        self.ctrl_min = -self.ctrl_margin
        self.ctrl_max = 1.0 + self.ctrl_margin

        # optional: clamp GT into model range (recommended so loss is well-defined)
        if clamp_gt:
            cp = self.ctrl_points
            finite = np.isfinite(cp)
            cp2 = cp.copy()
            cp2[finite] = np.clip(cp2[finite], self.ctrl_min, self.ctrl_max)
            self.ctrl_points = cp2

        if verbose_stats:
            cp = self.ctrl_points
            finite = np.isfinite(cp)
            if finite.any():
                mn = float(cp[finite].min())
                mx = float(cp[finite].max())
                print(
                    f"[LaneCtrlDataset] ctrl_points finite range after clamp_gt={clamp_gt}: "
                    f"min={mn:.3f}  max={mx:.3f}  (model range [{self.ctrl_min:.2f},{self.ctrl_max:.2f}])"
                )

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return int(len(self.image_path))

    def _read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _resize(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def _maybe_augment(self, img: np.ndarray) -> np.ndarray:
        # No geometric aug (would require transforming ctrl points).
        if not self.augment:
            return img
        if random.random() < 0.5:
            alpha = 0.8 + 0.4 * random.random()  # contrast
            beta = int(-10 + 20 * random.random())  # brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx: int):
        """
        Load and preprocess a dataset sample.

        Returns the resized image tensor, lane existence labels
        and corresponding control point targets.
        """
        path = self.image_path[idx]
        img = self._read_image(path)
        img = self._resize(img)
        img = self._maybe_augment(img)

        x = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0

        lane_present = torch.from_numpy(
            self.lane_present[idx].astype(np.float32)
        )  # (3,)
        ctrl = torch.from_numpy(self.ctrl_points[idx])  # (3,m,2)

        # replace NaNs with 0 (mask will ignore)
        ctrl = torch.nan_to_num(ctrl, nan=0.0)

        return x, lane_present, ctrl


# ----------------------------
# Model
# ----------------------------
class LaneNet(nn.Module):
    """
    MobileNetV3 backbone + two heads.
      - existence logits: (B,3)
      - ctrl points: (B,3,m,2) in normalized coords, but allow out-of-frame via scaled sigmoid:
            range = [-margin, 1+margin]
    """

    def __init__(self, m: int, *, ctrl_margin: float = 0.25, pretrained: bool = True):
        """
        Initialize lane detection network.

        Creates a MobileNetV3 backbone with two heads:
        one for lane existence classification and one
        for predicting control points.
        """
        super().__init__()
        self.m = int(m)
        self.ctrl_margin = float(ctrl_margin)

        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feat_dim = backbone.classifier[0].in_features  # usually 960

        self.head_exist = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 3),
        )

        self.head_ctrl = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 3 * self.m * 2),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Returns lane existence logits and predicted
        control point coordinates.
        """
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)

        exist_logits = self.head_exist(f)  # (B,3)

        ctrl_raw = self.head_ctrl(f).view(-1, 3, self.m, 2)  # (B,3,m,2)

        # scaled sigmoid to allow out-of-frame ctrl points
        s = torch.sigmoid(ctrl_raw)
        lo = -self.ctrl_margin
        hi = 1.0 + self.ctrl_margin
        ctrl_pred = s * (hi - lo) + lo  # -> [-m, 1+m]

        return exist_logits, ctrl_pred


# ----------------------------
# Loss
# ----------------------------
def masked_smooth_l1(
    pred: torch.Tensor, gt: torch.Tensor, mask_lane: torch.Tensor, beta: float = 0.02
) -> torch.Tensor:
    """
    Compute masked Smooth L1 regression loss.

    Applies regression loss only to lanes that are present
    in the ground truth using a lane presence mask.
    """
    loss = F.smooth_l1_loss(pred, gt, reduction="none", beta=beta)  # (B,3,m,2)
    mask = mask_lane[:, :, None, None]
    loss = loss * mask
    denom = mask.sum() * pred.shape[2] * pred.shape[3]
    denom = torch.clamp(denom, min=1.0)
    return loss.sum() / denom


# ----------------------------
# Train / Eval
# ----------------------------
@dataclass
class TrainConfig:
    """
    Configuration container for model training.

    Stores all hyperparameters and paths required
    to run a training experiment.
    """

    npz: str
    out_dir: str
    img_w: int = 640
    img_h: int = 384
    batch: int = 16
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.15
    seed: int = 42
    lambda_reg: float = 5.0
    pretrained: bool = True
    num_workers: int = 4
    ctrl_margin: float = 0.25
    clamp_gt: bool = True


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, lambda_reg: float
) -> dict:
    """
    Evaluate model performance on a validation dataset.

    Computes classification accuracy and regression loss
    metrics for lane existence and control point prediction.
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    total = 0
    sum_loss = 0.0
    sum_bce = 0.0
    sum_reg = 0.0

    correct = 0
    count_exist = 0

    for x, lane_present, ctrl in loader:
        x = x.to(device)
        lane_present = lane_present.to(device)
        ctrl = ctrl.to(device)

        exist_logits, ctrl_pred = model(x)

        loss_bce = bce(exist_logits, lane_present)
        loss_reg = masked_smooth_l1(ctrl_pred, ctrl, lane_present)
        loss = loss_bce + lambda_reg * loss_reg

        bs = x.shape[0]
        total += bs
        sum_loss += float(loss.item()) * bs
        sum_bce += float(loss_bce.item()) * bs
        sum_reg += float(loss_reg.item()) * bs

        probs = torch.sigmoid(exist_logits)
        pred_exist = (probs > 0.5).float()
        correct += int((pred_exist == lane_present).sum().item())
        count_exist += int(lane_present.numel())

    return {
        "loss": sum_loss / max(total, 1),
        "bce": sum_bce / max(total, 1),
        "reg": sum_reg / max(total, 1),
        "exist_acc": correct / max(count_exist, 1),
    }


def train(cfg: TrainConfig) -> None:
    """
    Train the lane detection model.

    Runs the training loop including optimization,
    validation evaluation and checkpoint saving.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_path = out_dir / "training_log.txt"
    log_file = open(log_path, "w")
    log_file.write("=== TRAIN CONFIG ===\n")
    log_file.write(str(cfg.__dict__) + "\n\n")
    log_file.flush()
    ##

    ds = LaneCtrlDataset(
        cfg.npz,
        img_size=(cfg.img_w, cfg.img_h),
        augment=True,
        ctrl_margin=cfg.ctrl_margin,
        clamp_gt=cfg.clamp_gt,
        verbose_stats=True,
    )

    n_total = len(ds)
    n_val = int(round(cfg.val_frac * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # store indices for reproducible val-only visualization later
    train_indices = list(getattr(train_ds, "indices", []))
    val_indices = list(getattr(val_ds, "indices", []))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = LaneNet(m=ds.m, ctrl_margin=cfg.ctrl_margin, pretrained=cfg.pretrained).to(
        device
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    best_val = float("inf")

    def _save_ckpt(path: Path, *, best: bool):
        torch.save(
            {
                "model_state": model.state_dict(),
                "m": ds.m,
                "img_w": cfg.img_w,
                "img_h": cfg.img_h,
                "npz": cfg.npz,
                "cfg": cfg.__dict__,
                "ctrl_margin": cfg.ctrl_margin,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "best": bool(best),
            },
            path,
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for x, lane_present, ctrl in train_loader:
            x = x.to(device)
            lane_present = lane_present.to(device)
            ctrl = ctrl.to(device)

            exist_logits, ctrl_pred = model(x)

            loss_bce = bce(exist_logits, lane_present)
            loss_reg = masked_smooth_l1(ctrl_pred, ctrl, lane_present)
            loss = loss_bce + cfg.lambda_reg * loss_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            bs = x.shape[0]
            running += float(loss.item()) * bs
            n_seen += bs

        train_loss = running / max(n_seen, 1)
        val_stats = evaluate(model, val_loader, device, cfg.lambda_reg)

        log_line = (
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_stats['loss']:.4f}  "
            f"val_bce={val_stats['bce']:.4f}  "
            f"val_reg={val_stats['reg']:.4f}  "
            f"exist_acc={val_stats['exist_acc']:.3f}"
        )

        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            ckpt_path = out_dir / "best.pt"
            _save_ckpt(ckpt_path, best=True)
            best_line = f"  ✓ saved best checkpoint: {ckpt_path}"
            print(best_line)
            log_file.write(best_line + "\n")
            log_file.flush()

    _save_ckpt(out_dir / "last.pt", best=False)
    print(f"[done] saved last checkpoint to: {out_dir / 'last.pt'}")
    log_file.close()


def main() -> None:
    """
    Entry point for training script.

    Parses command line arguments, builds the training
    configuration and launches the training process.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--img-h", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-reg", type=float, default=5.0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--ctrl-margin",
        type=float,
        default=0.25,
        help="Allows ctrl outside frame: range [-m,1+m]",
    )
    ap.add_argument(
        "--no-clamp-gt",
        action="store_true",
        help="Disable clamping GT ctrl points into model range",
    )
    args = ap.parse_args()

    cfg = TrainConfig(
        npz=args.npz,
        out_dir=args.out,
        img_w=args.img_w,
        img_h=args.img_h,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        val_frac=args.val_frac,
        seed=args.seed,
        lambda_reg=args.lambda_reg,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
        ctrl_margin=float(args.ctrl_margin),
        clamp_gt=not args.no_clamp_gt,
    )

    train(cfg)


if __name__ == "__main__":
    main()

# Verwendung zum Training der 4 Netze
"""
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m6.npz  --out artifacts/train_bezier_m6  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bezier_m8.npz  --out artifacts/train_bezier_m8  --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m6.npz --out artifacts/train_bspline_m6 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
python src/train_lane_ctrlpoints.py --npz artifacts/gt_bspline_m8.npz --out artifacts/train_bspline_m8 --epochs 50 --batch 16 --img-w 640 --img-h 384 --ctrl-margin 0.25
"""
