#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split


#-----------------
# config
#-----------------
@dataclass
class CFG:
    seed: int = 42

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4

    scheduler: str = "cosine"
    min_lr: float = 1e-6

    dropout: float = 0.3
    hidden_dim: int = 256

    val_split: float = 0.2


#-----------------
# seed
#-----------------
def set_seed(*, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#-----------------
# model
#-----------------
class Model(nn.Module):
    def __init__(self, *, cfg: CFG):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


#-----------------
# image loader
#-----------------
def load_image(*, path: Path, cfg: CFG):
    img = Image.open(path).convert("RGB")
    img = img.resize((cfg.img_size, cfg.img_size))

    img = np.array(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


#-----------------
# dataset
#-----------------
class ImageDataset(Dataset):
    def __init__(self, *, root_dir: Path, cfg: CFG):
        self.samples = []
        self.cfg = cfg

        for label in ["0", "1"]:
            class_dir = Path(root_dir) / label
            for path in class_dir.iterdir():
                if path.is_file():
                    self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = load_image(path=path, cfg=self.cfg)
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
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()

            correct += (preds == yb).sum().item()
            total += yb.numel()

    return correct / total


#-----------------
# train with validation
#-----------------
def train_with_validation(*, model, train_dir: Path, cfg: CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = ImageDataset(root_dir=train_dir, cfg=cfg)

    train_size = int((1 - cfg.val_split) * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.min_lr
        )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == yb).sum().item()
            train_total += yb.numel()

        if scheduler:
            scheduler.step()

        val_acc = evaluate(model=model, loader=val_loader, device=device)
        train_acc = train_correct / train_total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return model


#-----------------
# train full dataset
#-----------------
def train_full(*, model, train_dir: Path, cfg: CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = ImageDataset(root_dir=train_dir, cfg=cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(cfg.epochs):
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Full Train] Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f}")

    return model


#-----------------
# predict (internal strict API)
#-----------------
def predict(*, model, test_dir: Path, cfg: CFG):
    model.eval()
    device = next(model.parameters()).device

    results = []
    files = sorted(Path(test_dir).iterdir())

    for path in files:
        if not path.is_file():
            continue

        x = load_image(path=path, cfg=cfg).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit).item()
            label = 1 if prob > 0.5 else 0

        results.append((path.name, label))

    return results


#-----------------
# pipeline (Kaggle API - REQUIRED SIGNATURE)
#-----------------
def generate_predictions(data_dir: str):
    cfg = CFG()
    set_seed(seed=cfg.seed)

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    assert train_dir.exists(), "Missing train directory"
    assert test_dir.exists(), "Missing test directory"

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
