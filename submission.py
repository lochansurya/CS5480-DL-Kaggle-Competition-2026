#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split


#-----------------
# seed
#-----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#-----------------
# model
#-----------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))  # robust to input size
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


#-----------------
# image loader
#-----------------
def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))  # updated size
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


#-----------------
# dataset
#-----------------
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label in ["0", "1"]:
            class_dir = Path(root_dir) / label
            for path in class_dir.iterdir():
                if path.is_file():
                    self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = load_image(path)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y


#-----------------
# evaluate
#-----------------
def evaluate(model, loader, device):
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
def train_with_validation(model, train_dir, epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = ImageDataset(train_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Val Acc: {val_acc:.4f}")

    return model


#-----------------
# train full dataset
#-----------------
def train_full(model, train_dir, epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = ImageDataset(train_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

    return model


#-----------------
# predict
#-----------------
def predict(model, test_dir):
    model.eval()
    device = next(model.parameters()).device

    results = []
    files = sorted(Path(test_dir).iterdir())

    for path in files:
        if not path.is_file():
            continue

        x = load_image(path).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit).item()
            label = 1 if prob > 0.5 else 0

        results.append((path.name, label))

    return results


#-----------------
# main
#-----------------
def generate_predictions(data_dir):
    set_seed(42)

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    assert train_dir.exists(), "Missing train directory"
    assert test_dir.exists(), "Missing test directory"

    # Phase 1: validation training
    print("=== Training with validation ===")
    model = Model()
    model = train_with_validation(model, train_dir)

    # Phase 2: retrain on full data
    print("=== Retraining on full dataset ===")
    model = Model()
    model = train_full(model, train_dir)

    # Predict
    results = predict(model, test_dir)

    df = pd.DataFrame(results, columns=["ID", "TARGET"])
    df.to_csv("submission.csv", index=False, lineterminator="\n", encoding="utf-8")


#-----------------
# entry
#-----------------
if __name__ == "__main__":
    data_dir = input().strip()
    generate_predictions(data_dir)
