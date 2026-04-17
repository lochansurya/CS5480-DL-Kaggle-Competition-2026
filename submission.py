#!/usr/bin/env python3
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
    batch_size: int = 64
    num_workers: int = 2

    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4

    dropout: float = 0.3
    val_split: float = 0.15
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    tta: bool = True


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
            T.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(20),
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    else:
        return T.Compose([
            T.Resize((cfg.img_size, cfg.img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


#-----------------
# model (ResNet-style with BasicBlocks)
#-----------------
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class Model(nn.Module):
    def __init__(self, *, cfg: CFG):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64),  BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2),  BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 1),
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
# evaluate
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


#-----------------
# train loop (shared)
#-----------------
def _run_epochs(*, model, loader, cfg, device, show_val=False, val_loader=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        epochs=cfg.epochs, steps_per_epoch=len(loader),
        pct_start=0.1, div_factor=25, final_div_factor=1e4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    label = 1 - cfg.label_smoothing

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_smooth = yb * label + (1 - yb) * (cfg.label_smoothing / 2)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb_smooth)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            total += yb.numel()

        if show_val and val_loader is not None:
            val_acc = evaluate(model=model, loader=val_loader, device=device)
            print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss/len(loader):.4f} | Train: {correct/total:.4f} | Val: {val_acc:.4f}")
        else:
            print(f"[Full] Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss/len(loader):.4f}")


#-----------------
# train with validation
#-----------------
def train_with_validation(*, model, train_dir: Path, cfg: CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # build two separate datasets so train/val get independent transforms
    train_ds = ImageDataset(root_dir=train_dir, cfg=cfg, train=True)
    val_ds   = ImageDataset(root_dir=train_dir, cfg=cfg, train=False)

    g = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(train_ds), generator=g).tolist()
    split = int((1 - cfg.val_split) * len(train_ds))
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    _run_epochs(model=model, loader=train_loader, cfg=cfg, device=device,
                show_val=True, val_loader=val_loader)
    return model


#-----------------
# train full dataset
#-----------------
def train_full(*, model, train_dir: Path, cfg: CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = ImageDataset(root_dir=train_dir, cfg=cfg, train=True)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers, pin_memory=True)

    _run_epochs(model=model, loader=loader, cfg=cfg, device=device)
    return model


#-----------------
# predict (with TTA)
#-----------------
def predict(*, model, test_dir: Path, cfg: CFG):
    model.eval()
    device = next(model.parameters()).device
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    normalize = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def prep(img, hflip=False, vflip=False):
        img = img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
        if hflip:
            img = TF.hflip(img)
        if vflip:
            img = TF.vflip(img)
        return normalize(img).unsqueeze(0).to(device)

    results = []
    for path in sorted(Path(test_dir).iterdir()):
        if not path.is_file():
            continue

        img = Image.open(path).convert("RGB")

        with torch.no_grad():
            if cfg.tta:
                views = [
                    prep(img),
                    prep(img, hflip=True),
                    prep(img, vflip=True),
                    prep(img, hflip=True, vflip=True),
                ]
                prob = sum(torch.sigmoid(model(v)).item() for v in views) / len(views)
            else:
                prob = torch.sigmoid(model(prep(img))).item()

        results.append((path.name, int(prob > 0.5)))

    return results


#-----------------
# pipeline
#-----------------
def generate_predictions(data_dir: str):
    cfg = CFG()
    set_seed(seed=cfg.seed)

    data_dir  = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    assert train_dir.exists(), "Missing train directory"
    assert test_dir.exists(),  "Missing test directory"

    print("=== Training with validation ===")
    model = Model(cfg=cfg)
    model = train_with_validation(model=model, train_dir=train_dir, cfg=cfg)

    print("=== Retraining on full dataset ===")
    model = Model(cfg=cfg)
    model = train_full(model=model, train_dir=train_dir, cfg=cfg)

    print("=== Predicting ===")
    results = predict(model=model, test_dir=test_dir, cfg=cfg)

    df = pd.DataFrame(results, columns=["ID", "TARGET"])
    df.to_csv("submission.csv", index=False, lineterminator="\n", encoding="utf-8")


#-----------------
# entry
#-----------------
if __name__ == "__main__":
    data_dir = input().strip()
    generate_predictions(data_dir)
