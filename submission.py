#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path


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
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


#-----------------
# data
#-----------------
def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


#-----------------
# dataset
#-----------------
def load_dataset(train_dir: Path):
    X, y = [], []

    for label in ["0", "1"]:
        class_dir = train_dir / label
        for path in class_dir.iterdir():
            if path.is_file():
                X.append(load_image(path))
                y.append(int(label))

    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X, y


#-----------------
# train
#-----------------
def train(model, train_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    X, y = load_dataset(train_dir)
    X, y = X.to(device), y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    epochs = 5
    batch_size = 64

    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))

        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y[idx]

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model


#-----------------
# predict
#-----------------
def predict(model, test_dir: Path):
    model.eval()
    results = []

    device = next(model.parameters()).device
    files = sorted(test_dir.iterdir())

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

    model = Model()
    model = train(model, train_dir)

    results = predict(model, test_dir)

    df = pd.DataFrame(results, columns=["ID", "TARGET"])
    df.to_csv("submission.csv", index=False, lineterminator="\n", encoding="utf-8")


#-----------------
# entry
#-----------------
if __name__ == "__main__":
    data_dir = input().strip()
    generate_predictions(data_dir)

