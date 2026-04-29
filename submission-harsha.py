#!/usr/bin/env python3
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Subset


#-----------------
# config
#-----------------
@dataclass
class CFG:
    seed: int = 42

    img_size: int = 224    
    batch_size: int = 128
    num_workers: int = 4

    epochs: int = 80
    lr: float = 7e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    dropout: float = 0.3
    val_split: float = 0.15
    label_smoothing: float = 0.0   # CHANGED
    grad_clip: float = 1.0
    tta: bool = False
    patience: int = 15             # CHANGED


#-----------------
# seed
#-----------------
def set_seed(*, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#-----------------
# transforms
#-----------------
def get_transforms(cfg: CFG, train: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),  # CHANGED
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.15),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((cfg.img_size, cfg.img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


#-----------------
# GeM pooling
#-----------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, 1)
        ).pow(1.0 / self.p)


#-----------------
# CBAM
#-----------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.fc(self.avg_pool(x).view(b, c))
        m = self.fc(self.max_pool(x).view(b, c))
        w = torch.sigmoid(a + m).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


#-----------------
# model
#-----------------
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.cbam  = CBAM(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return self.relu(out + self.shortcut(x))


def _make_layer(in_c, out_c, num_blocks, stride):
    layers = [BasicBlock(in_c, out_c, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_c, out_c))
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, *, cfg: CFG):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(64,  64,  3, stride=1)
        self.layer2 = _make_layer(64,  128, 4, stride=2)
        self.layer3 = _make_layer(128, 256, 6, stride=2)
        self.layer4 = _make_layer(256, 512, 3, stride=2)

        self.pool = GeM()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


#-----------------
# dataset
#-----------------
class ImageDataset(Dataset):
    def __init__(self, *, root_dir: Path, cfg: CFG, train: bool):
        self.samples = []
        self.transform = get_transforms(cfg, train)

        for label in ["0", "1"]:
            class_dir = Path(root_dir) / label
            for path in sorted(class_dir.iterdir()):
                if path.is_file():
                    self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


#-----------------
# scheduler
#-----------------
def get_scheduler(optimizer, cfg):
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))  # CHANGED
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


#-----------------
# evaluate + threshold
#-----------------
def evaluate(*, model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = (torch.sigmoid(model(xb)) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total


def find_best_threshold(*, model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = torch.sigmoid(model(xb)).cpu().numpy().flatten()
            all_probs.extend(p)
            all_labels.extend(yb.numpy().flatten())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 401):  # CHANGED
        acc = ((probs > t).astype(float) == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    print(f"Best threshold: {best_t:.3f}  Val acc: {best_acc:.4f}")
    return best_t


#-----------------
# training loop (unchanged)
#-----------------
def _run_epochs(*, model, loader, cfg, device, show_val=False, val_loader=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    smooth = cfg.label_smoothing

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_s = yb * (1 - smooth) + (1 - yb) * (smooth / 2)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
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

        if show_val and val_loader is not None:
            val_acc = evaluate(model=model, loader=val_loader, device=device)
            print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss/len(loader):.4f} | "
                  f"Train Acc: {correct/total:.4f} | Val Acc: {val_acc:.4f}")
        else:
            print(f"[Full] Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss/len(loader):.4f}")


#-----------------
# train_with_validation
#-----------------
def train_with_validation(*, model, train_dir: Path, cfg: CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = ImageDataset(root_dir=train_dir, cfg=cfg, train=True)
    val_ds   = ImageDataset(root_dir=train_dir, cfg=cfg, train=False)

    g = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(train_ds), generator=g).tolist()
    split = int((1 - cfg.val_split) * len(train_ds))
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader   = DataLoader(Subset(val_ds, val_idx), batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)

    _run_epochs(model=model, loader=train_loader, cfg=cfg, device=device,
                show_val=True, val_loader=val_loader)

    threshold = find_best_threshold(model=model, loader=val_loader, device=device)
    return model, threshold


#-----------------
# train_full
#-----------------
def train_full(*, model, train_dir: Path, cfg: CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = ImageDataset(root_dir=train_dir, cfg=cfg, train=True)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=cfg.num_workers > 0)

    _run_epochs(model=model, loader=loader, cfg=cfg, device=device)
    return model


#-----------------
# predict
#-----------------
def predict(*, model, test_dir: Path, cfg: CFG, threshold: float):
    model.eval()
    device = next(model.parameters()).device
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    normalize = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def prep(img):
        img = img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
        return normalize(img).unsqueeze(0).to(device)

    results = []
    for path in sorted(Path(test_dir).iterdir()):
        if not path.is_file():
            continue

        img = Image.open(path).convert("RGB")

        with torch.no_grad():
            prob = torch.sigmoid(model(prep(img))).item()

        results.append((path.name, int(prob > threshold)))

    return results


#-----------------
# pipeline
#-----------------
def generate_predictions(data_dir: str):
    cfg = CFG()
    set_seed(seed=cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    data_dir  = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    print("=== Training with validation ===")
    model = Model(cfg=cfg)
    model, threshold = train_with_validation(model=model, train_dir=train_dir, cfg=cfg)

    print("=== Retraining on full dataset ===")
    model = Model(cfg=cfg)
    model = train_full(model=model, train_dir=train_dir, cfg=cfg)

    print("=== Predicting ===")
    results = predict(model=model, test_dir=test_dir, cfg=cfg, threshold=threshold)

    df = pd.DataFrame(results, columns=["ID", "TARGET"])
    df.to_csv("submission-harsha.csv", index=False)


#-----------------
# entry
#-----------------
if __name__ == "__main__":
    generate_predictions(input().strip())
