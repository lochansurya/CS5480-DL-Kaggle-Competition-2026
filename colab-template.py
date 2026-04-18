# =========================
# 1) Mount Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 2) Imports
# =========================
import math, random, os
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

# =========================
# 3) Config
# =========================
@dataclass
class CFG:
    seed: int = 42
    img_size: int = 224
    batch_size: int = 128   # safer for Colab GPU
    num_workers: int = 2
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    dropout: float = 0.3
    val_split: float = 0.15
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    tta: bool = True

# =========================
# 4) Seed
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 5) Transforms
# =========================
def get_transforms(cfg, train):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return T.Compose([
            T.RandomResizedCrop(cfg.img_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3,0.3,0.3,0.1),
            T.ToTensor(),
            T.Normalize(mean,std),
        ])
    else:
        return T.Compose([
            T.Resize((cfg.img_size, cfg.img_size)),
            T.ToTensor(),
            T.Normalize(mean,std),
        ])

# =========================
# 6) Dataset
# =========================
class ImageDataset(Dataset):
    def __init__(self, root_dir, cfg, train):
        self.samples = []
        self.transform = get_transforms(cfg, train)

        for label in ["0","1"]:
            class_dir = Path(root_dir) / label
            for p in class_dir.iterdir():
                if p.is_file():
                    self.samples.append((p, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y

# =========================
# 7) Model (simplified)
# =========================
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(256,1)
        )

    def forward(self,x):
        return self.net(x)

# =========================
# 8) Train
# =========================
def train(model, loader, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()

    for epoch in range(cfg.epochs):
        total_loss = 0
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

# =========================
# 9) Predict
# =========================
def predict(model, test_dir, cfg, device):
    model.eval()
    results = []

    transform = get_transforms(cfg, train=False)

    for p in sorted(Path(test_dir).iterdir()):
        if not p.is_file(): continue

        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        results.append((p.name, int(prob>0.5)))

    return results

# =========================
# 10) MAIN PIPELINE
# =========================
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 🔥 YOUR DATA PATH HERE
    data_dir = Path("/content/drive/MyDrive/data")

    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    train_ds = ImageDataset(train_dir, cfg, train=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    model = Model(cfg).to(device)

    print("=== Training ===")
    train(model, train_loader, cfg, device)

    print("=== Predicting ===")
    results = predict(model, test_dir, cfg, device)

    df = pd.DataFrame(results, columns=["ID","TARGET"])
    df.to_csv("/content/submission.csv", index=False)

    print("Saved to /content/submission.csv")

# =========================
# 11) Run
# =========================
main()