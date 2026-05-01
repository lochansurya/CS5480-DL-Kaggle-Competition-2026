#!/usr/bin/env python3
"""
Custom ResNet-34-like network with CBAM Attention and GeM Pooling
Binary Image Classification Pipeline

Implementation with:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Early stopping with best checkpoint
    - Cosine LR scheduler with warmup
    - DropPath (stochastic depth)
    - Threshold optimization
    - Two-phase training (validation + full retrain)
"""

import copy
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, Subset

from PIL import Image, ImageEnhance, ImageFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


#-------------------------------------
# Config
#-------------------------------------
@dataclass
class CFG:
    seed: int = 42
    ensemble_seeds: tuple[int, ...] = (42, 43, 44)
    split_seed: int = 42

    img_size: int = 224
    batch_size: int = 128
    num_workers: int = 4

    epochs: int = 100
    lr: float = 7e-4
    weight_decay: float = 1e-4

    dropout: float = 0.2
    drop_path_rate: float = 0.1
    val_split: float = 0.15
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_prob: float = 0.0
    grad_clip: float = 1.0
    tta: bool = True
    patience: int = 20
    warmup_epochs: int = 5

    def save(self) -> None:
        path = Path("config.json")
        cfg_dict = {
            "seed": self.seed,
            "ensemble_seeds": list(self.ensemble_seeds),
            "split_seed": self.split_seed,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "drop_path_rate": self.drop_path_rate,
            "val_split": self.val_split,
            "label_smoothing": self.label_smoothing,
            "mixup_alpha": self.mixup_alpha,
            "cutmix_prob": self.cutmix_prob,
            "grad_clip": self.grad_clip,
            "tta": self.tta,
            "patience": self.patience,
            "warmup_epochs": self.warmup_epochs,
        }
        path.write_text(json.dumps(cfg_dict, indent=2))
        logger.info(f"Saved config -> {path}")


#-------------------------------------
# Constants
#-------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Edge maps are near-binary (mostly 0, bright at outlines).
# Use simple [-1,1] normalisation instead of ImageNet stats.
EDGE_MEAN = [0.5, 0.5, 0.5]
EDGE_STD  = [0.5, 0.5, 0.5]


#-------------------------------------
# ShapePreprocess
#-------------------------------------
class ShapePreprocess:
    """
    Converts a PIL RGB image into a grayscale edge map:
        RGB → L → contrast boost → FIND_EDGES → RGB (3 identical channels)

    Rationale for this dataset (CLEVR-style renders):
      - Colour is irrelevant: same colours appear in both classes.
      - The discriminative signal is purely shape — sphere (rounded outline)
        vs cube/cylinder (angular outline).
      - FIND_EDGES extracts outlines whose curvature directly encodes shape,
        making it trivial for the model to learn the sphere/cube distinction.
    """
    def __init__(self, contrast: float = 3.0):
        self.contrast = contrast

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(self.contrast)
        img = img.filter(ImageFilter.FIND_EDGES)
        return img.convert("RGB")


#-------------------------------------
# DropPath: Stochastic Depth
#-------------------------------------
class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) regularization.

    Randomly drops entire residual branches during training.
    Helps improve generalization and reduces overfitting.

    Forward:
        x -> [drop with probability drop_prob] -> x / (1 - drop_prob)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * noise.floor()


#-------------------------------------
# Seed
#-------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


#-------------------------------------
# Device
#-------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device
    return torch.device("cpu")


#-------------------------------------
# Transforms
#-------------------------------------
def get_transforms(img_size: int, train: bool) -> T.Compose:
    preprocess = ShapePreprocess(contrast=3.0)
    if train:
        return T.Compose([
            preprocess,                          # grayscale → contrast → edge map
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(EDGE_MEAN, EDGE_STD),
        ])
    return T.Compose([
        preprocess,
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(EDGE_MEAN, EDGE_STD),
    ])


#-------------------------------------
# GeM: Generalized Mean Pooling
#-------------------------------------
class GeM(nn.Module):
    """
    Generalized Mean (GeM) Pooling.

    Generalizes average pooling with a learned power parameter.
    Helps capture more informative feature representations.

    Architecture:
        x -> clamp(eps) -> pow(p) -> avg_pool2d -> pow(1/p) -> output
        where p is a learnable parameter (default p=3.0)
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)


#-------------------------------------
# ChannelAttention
#-------------------------------------
class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM).

    Learns to emphasize important channel features.

    Architecture:
        x -> avg_pool -> fc -> ReLU -> fc -> sigmoid ->
        x * sigmoid(fc(avg_pool) + fc(max_pool))
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = torch.sigmoid(
            self.fc(self.avg_pool(x).view(b, c)) +
            self.fc(self.max_pool(x).view(b, c))
        ).view(b, c, 1, 1)
        return x * w


#-------------------------------------
# SpatialAttention
#-------------------------------------
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM).

    Learns to emphasize important spatial locations.

    Architecture:
        x -> concat(avg_pool, max_pool) -> conv1x1 -> sigmoid ->
        x * sigmoid(conv(concat))
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        return x * torch.sigmoid(self.conv(cat))


#-------------------------------------
# CBAM: Convolutional Block Attention Module
#-------------------------------------
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Applies channel and spatial attention sequentially.

    Architecture:
        x -> [Channel Attention] -> [Spatial Attention] -> output

    Benefits:
        - Channel attention: emphasizes important features
        - Spatial attention: focuses on relevant regions
    """
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


#-------------------------------------
# BasicBlock
#-------------------------------------
class BasicBlock(nn.Module):
    """
    ResNet basic block with CBAM attention.

    The residual branch is lightly regularized with stochastic depth, then
    added back to the shortcut before the final activation.
    """
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1, drop_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.cbam = CBAM(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.drop_path = DropPath(drop_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.cbam(out)
        out = self.drop_path(out)
        return self.relu(out + identity)


#-------------------------------------
# Model: ResNet-34-like + CBAM + GeM
#-------------------------------------
class Model(nn.Module):
    """
    Custom ResNet-34-like network with CBAM attention and GeM pooling.

        Stem: Conv7x7(3->64) + BN + ReLU + MaxPool3x3

        Residual stages: 3, 4, 6, 3 blocks
        Final channels: 512

        GeM Pooling -> 512 channels
        Classifier: Dropout -> Linear(512->256) -> ReLU ->
                   Dropout -> Linear(256->1)

    Key Features:
        - ResNet-34-like 3+4+6+3 block layout trained from scratch
        - CBAM attention in every residual block
        - GeM pooling for better feature representation
        - DropPath for regularization
    """
    def __init__(self, cfg: CFG):
        super().__init__()
        self.in_c = 64
        block_config = (3, 4, 6, 3)
        drop_probs = torch.linspace(0, cfg.drop_path_rate, sum(block_config)).tolist()
        self._drop_idx = 0

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, block_config[0], stride=1, drop_probs=drop_probs)
        self.layer2 = self._make_layer(128, block_config[1], stride=2, drop_probs=drop_probs)
        self.layer3 = self._make_layer(256, block_config[2], stride=2, drop_probs=drop_probs)
        self.layer4 = self._make_layer(512, block_config[3], stride=2, drop_probs=drop_probs)

        self.pool = GeM(p=3.0)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(256, 1),
        )

    def _make_layer(
        self,
        out_c: int,
        blocks: int,
        stride: int,
        drop_probs: list[float],
    ) -> nn.Sequential:
        layers = []
        for i in range(blocks):
            block_stride = stride if i == 0 else 1
            layers.append(
                BasicBlock(
                    self.in_c,
                    out_c,
                    stride=block_stride,
                    drop_prob=drop_probs[self._drop_idx],
                )
            )
            self.in_c = out_c
            self._drop_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


#-------------------------------------
# ImageDataset
#-------------------------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir: Path, cfg: CFG, train: bool):
        self.transform = get_transforms(cfg.img_size, train)
        self.samples = []
        for label in ("0", "1"):
            for path in sorted((Path(root_dir) / label).iterdir()):
                if path.is_file():
                    self.samples.append((path, int(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


#-------------------------------------
# Scheduler
#-------------------------------------
def get_scheduler(optimizer: torch.optim.Optimizer, cfg: CFG):
    def lr_lambda(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
#-------------------------------------
# Evaluate
#-------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = (torch.sigmoid(model(xb)) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return correct / total


def find_best_threshold_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 401):
        acc = ((probs > t).astype(float) == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    logger.info(f"Best threshold: {best_t:.3f}, Validation accuracy: {best_acc:.4f}")
    return best_t


@torch.no_grad()
def collect_probs_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: CFG,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        probs = torch.sigmoid(model(xb))
        if cfg.tta:
            probs = (probs + torch.sigmoid(model(torch.flip(xb, dims=[3])))) * 0.5
        all_probs.extend(probs.cpu().numpy().flatten())
        all_labels.extend(yb.numpy().flatten())
    return np.array(all_probs), np.array(all_labels)


#-------------------------------------
# run_epochs
#-------------------------------------
def run_epochs(
    model: nn.Module,
    loader: DataLoader,
    cfg: CFG,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    history: Optional[dict] = None,
) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer, cfg)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = correct = total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_s = yb * (1 - cfg.label_smoothing) + (1 - yb) * (cfg.label_smoothing / 2)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb_s)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            total += yb.numel()

        scheduler.step()

        avg_loss = total_loss / len(loader)
        train_acc = correct / total
        lr_now = scheduler.get_last_lr()[0]

        if history is not None:
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_acc)
            history["lr"].append(lr_now)

        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            if history is not None:
                history["val_acc"].append(val_acc)
            logger.info(
                f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Accuracy: {train_acc:.4f} | "
                f"Validation Accuracy: {val_acc:.4f} | "
                f"LR: {lr_now:.2e}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (patience={cfg.patience})")
                    break
        else:
            logger.info(
                f"[Full] Epoch {epoch + 1:3d}/{cfg.epochs} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Train Accuracy: {train_acc:.4f}"
            )

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model with Validation Accuracy: {best_val_acc:.4f}")

    return model


#-------------------------------------
# train_with_validation
#-------------------------------------
def train_with_validation(model: nn.Module, train_dir: Path, cfg: CFG) -> tuple[nn.Module, float]:
    device = get_device()
    model = model.to(device)

    train_ds = ImageDataset(train_dir, cfg, train=True)
    val_ds = ImageDataset(train_dir, cfg, train=False)

    g = torch.Generator().manual_seed(cfg.split_seed)
    indices = torch.randperm(len(train_ds), generator=g).tolist()
    split = int((1 - cfg.val_split) * len(train_ds))
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    model = run_epochs(model, train_loader, cfg, device, val_loader)
    probs, labels = collect_probs_and_labels(model, val_loader, device, cfg)
    threshold = find_best_threshold_from_probs(probs, labels)
    return model, threshold


def train_with_validation_probs(
    model: nn.Module,
    train_dir: Path,
    cfg: CFG,
    history: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    device = get_device()
    model = model.to(device)

    train_ds = ImageDataset(train_dir, cfg, train=True)
    val_ds = ImageDataset(train_dir, cfg, train=False)

    g = torch.Generator().manual_seed(cfg.split_seed)
    indices = torch.randperm(len(train_ds), generator=g).tolist()
    split = int((1 - cfg.val_split) * len(train_ds))
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    model = run_epochs(model, train_loader, cfg, device, val_loader, history=history)
    return collect_probs_and_labels(model, val_loader, device, cfg)


#-------------------------------------
# train_full
#-------------------------------------
def train_full(model: nn.Module, train_dir: Path, cfg: CFG) -> nn.Module:
    device = get_device()
    model = model.to(device)

    dataset = ImageDataset(train_dir, cfg, train=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    return run_epochs(model, loader, cfg, device)


@torch.no_grad()
def predict_probs(model: nn.Module, test_dir: Path, cfg: CFG) -> list[tuple[str, float]]:
    model.eval()
    device = next(model.parameters()).device
    preprocess = ShapePreprocess(contrast=3.0)
    normalize = T.Compose([T.ToTensor(), T.Normalize(EDGE_MEAN, EDGE_STD)])

    def prep(img):
        img = preprocess(img)
        img = img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
        return normalize(img).unsqueeze(0).to(device)

    def predict_prob(img: Image.Image) -> float:
        tensors = [prep(img)]
        if cfg.tta:
            tensors.append(prep(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
        batch = torch.cat(tensors, dim=0)
        return torch.sigmoid(model(batch)).mean().item()

    probs_by_id = []
    for path in sorted(Path(test_dir).iterdir()):
        if not path.is_file():
            continue
        img = Image.open(path).convert("RGB")
        prob = predict_prob(img)
        probs_by_id.append((path.name, prob))

    return probs_by_id


#-------------------------------------
# predict
#-------------------------------------
def labels_from_probs(
    probs_by_id: list[tuple[str, float]],
    threshold: float,
) -> list[tuple[str, int]]:
    return [(image_id, int(prob > threshold)) for image_id, prob in probs_by_id]


#-------------------------------------
# save_report_plots
#-------------------------------------
def save_report_plots(
    histories: list[dict],
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    threshold: float,
    train_dir: Path,
    out_dir: Path = Path("plots"),
) -> None:
    from sklearn.metrics import (
        roc_curve, auc, confusion_matrix,
        precision_score, recall_score, f1_score,
    )

    out_dir.mkdir(exist_ok=True)
    preds = (val_probs > threshold).astype(int)
    COLORS = plt.cm.tab10.colors

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"})

    # ── 1. Training curves ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Training Curves per Seed", fontsize=15, fontweight="bold", y=1.01)

    for i, h in enumerate(histories):
        color = COLORS[i % len(COLORS)]
        label = f"Seed {h.get('seed', i)}"
        n = len(h["train_loss"])
        ep = range(1, n + 1)

        axes[0, 0].plot(ep, h["train_loss"], color=color, lw=1.8, label=label)
        axes[0, 1].plot(ep, [a * 100 for a in h["train_acc"]], color=color, lw=1.8, label=label)
        if h.get("val_acc"):
            axes[1, 0].plot(range(1, len(h["val_acc"]) + 1),
                            [a * 100 for a in h["val_acc"]], color=color, lw=1.8, label=label)
        axes[1, 1].plot(ep, h["lr"], color=color, lw=1.8, label=label)

    for ax, title, ylabel in [
        (axes[0, 0], "Training Loss",        "BCE Loss"),
        (axes[0, 1], "Training Accuracy",    "Accuracy (%)"),
        (axes[1, 0], "Validation Accuracy",  "Accuracy (%)"),
        (axes[1, 1], "Learning Rate",        "LR"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.tick_params(labelsize=9)

    axes[1, 1].set_yscale("log")
    axes[1, 0].axhline(75.91, color="red", linestyle=":", lw=1.2, label="Best (75.91%)")
    axes[1, 0].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training_curves.png")

    # ── 2. ROC Curve ────────────────────────────────────────────────────────
    fpr, tpr, roc_thresholds = roc_curve(val_labels, val_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.tick_params(labelsize=9)
    fig.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved roc_curve.png")

    # ── 3. Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(val_labels, preds)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Class 0\n(no sphere+cube)", "Class 1\n(sphere+cube)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{cm[i, j]:,}\n({100 * cm[i, j] / total:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion_matrix.png")

    # ── 4. Classification Metrics Bar Chart ──────────────────────────────────
    acc    = float((preds == val_labels).mean())
    prec   = float(precision_score(val_labels, preds))
    rec    = float(recall_score(val_labels, preds))
    f1     = float(f1_score(val_labels, preds))

    names  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    values = [acc, prec, rec, f1, roc_auc]
    bar_colors = ["steelblue", "coral", "mediumseagreen", "orchid", "gold"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=bar_colors, edgecolor="white", linewidth=1.2, width=0.55)
    ax.set_ylim(0, 1.08)
    ax.set_title("Classification Metrics Summary")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "classification_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved classification_metrics.png")

    # ── 5. Predicted Probability Distribution ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(val_probs[val_labels == 0], bins=50, alpha=0.65,
            color="steelblue", label="True Class 0 (no sphere+cube)", edgecolor="white")
    ax.hist(val_probs[val_labels == 1], bins=50, alpha=0.65,
            color="coral",     label="True Class 1 (sphere+cube)",    edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Decision threshold = {threshold:.3f}")
    ax.set_title("Predicted Probability Distribution by True Class")
    ax.set_xlabel("Predicted Probability (P(Class 1))")
    ax.set_ylabel("Sample Count")
    ax.legend(fontsize=10)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "probability_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved probability_distribution.png")

    # ── 6. Threshold Sensitivity ────────────────────────────────────────────
    thresholds = np.linspace(0.01, 0.99, 500)
    accs = [((val_probs > t).astype(int) == val_labels).mean() for t in thresholds]
    best_idx = int(np.argmax(accs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [a * 100 for a in accs], color="steelblue", lw=2.2, label="Val Accuracy")
    ax.axvline(threshold, color="red", linestyle="--", lw=1.8,
               label=f"Chosen threshold = {threshold:.3f}")
    ax.axhline(accs[best_idx] * 100, color="gray", linestyle=":", lw=1.2,
               label=f"Peak = {accs[best_idx] * 100:.2f}%")
    ax.set_title("Validation Accuracy vs Decision Threshold")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved threshold_sensitivity.png")

    # ── 7. Dataset Distribution ──────────────────────────────────────────────
    n0 = len(list((train_dir / "0").glob("*")))
    n1 = len(list((train_dir / "1").glob("*")))
    total_imgs = n0 + n1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Training Dataset Distribution", fontsize=14, fontweight="bold")

    bars = axes[0].bar(
        ["Class 0\n(no sphere+cube)", "Class 1\n(sphere+cube)"],
        [n0, n1], color=["steelblue", "coral"], edgecolor="white", width=0.5
    )
    for bar, val in zip(bars, [n0, n1]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total_imgs * 0.005,
                     f"{val:,}\n({100 * val / total_imgs:.1f}%)",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_title("Class Count")
    axes[0].set_ylabel("Number of Images")
    axes[0].set_ylim(0, max(n0, n1) * 1.15)
    axes[0].grid(True, alpha=0.3, axis="y", linestyle="--")
    axes[0].tick_params(labelsize=9)

    axes[1].pie([n0, n1],
                labels=["Class 0\n(no sphere+cube)", "Class 1\n(sphere+cube)"],
                colors=["steelblue", "coral"],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10},
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Class Balance")

    plt.tight_layout()
    fig.savefig(out_dir / "dataset_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dataset_distribution.png")

    # ── 8. Precision-Recall Curve ────────────────────────────────────────────
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision_curve, recall_curve, _ = precision_recall_curve(val_labels, val_probs)
    avg_prec = average_precision_score(val_labels, val_probs)
    baseline = val_labels.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_curve, precision_curve, color="mediumseagreen", lw=2.5,
            label=f"PR curve (AP = {avg_prec:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1.2,
               label=f"Baseline (class freq = {baseline:.3f})")
    ax.fill_between(recall_curve, precision_curve, alpha=0.08, color="mediumseagreen")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved precision_recall_curve.png")

    logger.info(
        f"\n{'='*50}\nReport Metrics Summary\n{'='*50}\n"
        f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)\n"
        f"  Precision : {prec:.4f}\n"
        f"  Recall    : {rec:.4f}\n"
        f"  F1-Score  : {f1:.4f}\n"
        f"  AUC-ROC   : {roc_auc:.4f}\n"
        f"  Avg Prec  : {avg_prec:.4f}\n"
        f"  Threshold : {threshold:.3f}\n"
        f"{'='*50}\n"
        f"Saved 8 plots -> {out_dir}/"
    )


#-------------------------------------
# generate_predictions: Main entry point
#-------------------------------------
def generate_predictions(data_dir: str) -> None:
    #-------------------------------------
    # save_dataset_info
    #-------------------------------------
    def save_dataset_info(data_dir: Path) -> None:
        train_dir = data_dir / "train"
        class_0 = len(list((train_dir / "0").glob("*")))
        class_1 = len(list((train_dir / "1").glob("*")))
        test_count = len(list((data_dir / "test").glob("*")))

        info = {
            "train_class_0": class_0,
            "train_class_1": class_1,
            "train_total": class_0 + class_1,
            "test_total": test_count,
        }
        Path("dataset.json").write_text(json.dumps(info, indent=2))
        logger.info(f"Saved dataset info -> dataset.json")

    cfg = CFG()
    cfg.save()

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    save_dataset_info(data_dir)

    val_probs_per_seed = []
    val_labels = None
    test_probs_per_seed = []
    image_ids = None
    histories = []

    for seed in cfg.ensemble_seeds:
        logger.info(f"=== Seed {seed}: Validation training ===")
        cfg.seed = seed
        set_seed(cfg.seed)
        model = Model(cfg)
        h = {"seed": seed, "train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
        histories.append(h)
        val_probs, labels = train_with_validation_probs(model, train_dir, cfg, history=h)
        val_probs_per_seed.append(val_probs)
        if val_labels is None:
            val_labels = labels
        elif not np.array_equal(val_labels, labels):
            raise RuntimeError("Validation labels changed across seeds; check split_seed.")

        logger.info(f"=== Seed {seed}: Full-data retraining ===")
        set_seed(cfg.seed)
        model = Model(cfg)
        model = train_full(model, train_dir, cfg)

        logger.info(f"=== Seed {seed}: Test probability prediction ===")
        probs_by_id = predict_probs(model, test_dir, cfg)
        ids = [image_id for image_id, _ in probs_by_id]
        probs = np.array([prob for _, prob in probs_by_id])
        test_probs_per_seed.append(probs)
        if image_ids is None:
            image_ids = ids
        elif image_ids != ids:
            raise RuntimeError("Test image ordering changed across seeds.")

    logger.info("=== Averaging ensemble probabilities ===")
    ensemble_val_probs = np.mean(np.stack(val_probs_per_seed, axis=0), axis=0)
    threshold = find_best_threshold_from_probs(ensemble_val_probs, val_labels)
    ensemble_test_probs = np.mean(np.stack(test_probs_per_seed, axis=0), axis=0)
    results = labels_from_probs(list(zip(image_ids, ensemble_test_probs)), threshold)

    df = pd.DataFrame(results, columns=["ID", "TARGET"])
    df.to_csv("submission.csv", index=False)
    logger.info(f"Saved {len(df)} predictions -> submission.csv")

    save_report_plots(histories, ensemble_val_probs, val_labels, threshold, train_dir)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    generate_predictions(data_dir)
