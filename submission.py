#!/usr/bin/env python3
"""
DenseNet-121 with CBAM Attention and GeM Pooling
Binary Image Classification Pipeline

Production-grade implementation with:
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, Subset

from PIL import Image

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

    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4

    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-4

    dropout: float = 0.3
    drop_path_rate: float = 0.0  # Disabled for DenseNet (dense connections provide regularization)
    val_split: float = 0.15
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    tta: bool = False
    patience: int = 15
    warmup_epochs: int = 5

    def save(self) -> None:
        path = Path("config.json")
        cfg_dict = {
            "seed": self.seed,
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
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.15),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
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
# DenseLayer
#-------------------------------------
class DenseLayer(nn.Module):
    """
    Single layer in DenseNet.

    BN -> ReLU -> Conv3x3 -> DropPath

    Each layer receives feature maps from all preceding layers.
    This dense connectivity improves gradient flow.
    """
    def __init__(self, in_c: int, growth_rate: int, drop_prob: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, growth_rate, 3, stride=1, padding=1, bias=False),
        )
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.layers(x))


#-------------------------------------
# DenseNetBlock
#-------------------------------------
class DenseNetBlock(nn.Module):
    """
    Dense block with CBAM attention.

    DenseNet Architecture:
        x0 ----> concatenated output
        x1 ----> concatenated output
        ...
        xN ----> concatenated output

    After concatenation, applies CBAM for attention.
    """
    def __init__(self, in_c: int, growth_rate: int, num_layers: int, drop_prob: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_c + i * growth_rate, growth_rate, drop_prob)
            for i in range(num_layers)
        ])
        self.cbam = CBAM(in_c + num_layers * growth_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return self.cbam(x)


#-------------------------------------
# Transition
#-------------------------------------
class Transition(nn.Module):
    """
    Transition layer between dense blocks.

    Reduces number of channels and spatial dimensions.

    Architecture:
        x -> BN -> ReLU -> Conv1x1 -> AvgPool2x2 -> output
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


#-------------------------------------
# Model: DenseNet-121 + CBAM + GeM
#-------------------------------------
class Model(nn.Module):
    """
    DenseNet-121 with CBAM attention and GeM pooling.

    Architecture (DenseNet-121 style):

        Stem: Conv7x7(3->64) + BN + ReLU + MaxPool3x3

        Dense Block 1: 6 layers, growth=32
        Output: 64 + (6 x 32) = 256 channels

        Transition 1: 256 -> 128, 2x downsampling
        Dense Block 2: 12 layers, growth=32
        Output: 128 + (12 x 32) = 512 channels

        Transition 2: 512 -> 256, 2x downsampling
        Dense Block 3: 24 layers, growth=32
        Output: 256 + (24 x 32) = 1024 channels

        Transition 3: 1024 -> 512, 2x downsampling
        Dense Block 4: 16 layers, growth=32
        Output: 512 + (16 x 32) = 1024 channels

        GeM Pooling -> 1024 channels
        Classifier: Dropout -> Linear(1024->256) -> ReLU ->
                   Dropout -> Linear(256->1)

    Key Features:
        - Dense connections: all layers connected to each other
        - CBAM attention after each dense block
        - GeM pooling for better feature representation
        - DropPath for regularization
    """
    GROWTH_RATE = 32

    def __init__(self, cfg: CFG):
        super().__init__()
        block_config = (6, 12, 24, 16)
        total_layers = sum(block_config)
        dp = torch.linspace(0, cfg.drop_path_rate, total_layers).tolist()
        idx = 0

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        num_features = 64
        self.dense1 = DenseNetBlock(num_features, self.GROWTH_RATE, block_config[0], cfg.drop_path_rate)
        num_features = num_features + block_config[0] * self.GROWTH_RATE
        self.trans1 = Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.dense2 = DenseNetBlock(num_features, self.GROWTH_RATE, block_config[1], cfg.drop_path_rate)
        num_features = num_features + block_config[1] * self.GROWTH_RATE
        self.trans2 = Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.dense3 = DenseNetBlock(num_features, self.GROWTH_RATE, block_config[2], cfg.drop_path_rate)
        num_features = num_features + block_config[2] * self.GROWTH_RATE
        self.trans3 = Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.dense4 = DenseNetBlock(num_features, self.GROWTH_RATE, block_config[3], cfg.drop_path_rate)
        num_features = num_features + block_config[3] * self.GROWTH_RATE

        self.bn_final = nn.BatchNorm2d(num_features)
        self.pool = GeM()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        x = self.bn_final(x)
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


@torch.no_grad()
#-------------------------------------
# Find Best Threshold
#-------------------------------------
def find_best_threshold(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_probs, all_labels = [], []
    for xb, yb in loader:
        probs = torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(yb.numpy().flatten())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 401):
        acc = ((probs > t).astype(float) == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    logger.info(f"Best threshold: {best_t:.3f}, Validation accuracy: {best_acc:.4f}")
    return best_t


#-------------------------------------
# run_epochs
#-------------------------------------
def run_epochs(
    model: nn.Module,
    loader: DataLoader,
    cfg: CFG,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
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

        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            lr_now = scheduler.get_last_lr()[0]
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

    g = torch.Generator().manual_seed(cfg.seed)
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
    threshold = find_best_threshold(model, val_loader, device)
    return model, threshold


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
#-------------------------------------
# predict
#-------------------------------------
def predict(model: nn.Module, test_dir: Path, cfg: CFG, threshold: float) -> list[tuple[str, int]]:
    model.eval()
    device = next(model.parameters()).device
    normalize = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def prep(img):
        img = img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
        return normalize(img).unsqueeze(0).to(device)

    results = []
    for path in sorted(Path(test_dir).iterdir()):
        if not path.is_file():
            continue
        img = Image.open(path).convert("RGB")
        prob = torch.sigmoid(model(prep(img))).item()
        results.append((path.name, int(prob > threshold)))

    return results


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
    save_dataset_info(data_dir)

    set_seed(cfg.seed)

    logger.info("=== Phase 1: Training with validation ===")
    model = Model(cfg)
    model, threshold = train_with_validation(model, train_dir, cfg)

    logger.info("=== Phase 2: Retraining on full dataset ===")
    model = Model(cfg)
    model = train_full(model, train_dir, cfg)

    logger.info("=== Phase 3: Predicting ===")
    results = predict(model, test_dir, cfg, threshold)

    df = pd.DataFrame(results, columns=["ID", "TARGET"])
    df.to_csv("submission.csv", index=False)
    logger.info(f"Saved {len(df)} predictions -> submission.csv")


if __name__ == "__main__":
    generate_predictions(input().strip())