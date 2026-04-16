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
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 1)
        )

    def forward(self, x):
        return self.net(x)


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
# train
#-----------------
def train(model, train_dir: Path):
    model.eval()
    return model


#-----------------
# predict
#-----------------
def predict(model, test_dir: Path):
    model.eval()
    results = []

    files = sorted(test_dir.iterdir())

    for path in files:
        if not path.is_file():
            continue

        x = load_image(path).unsqueeze(0)

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
    df.to_csv("submission.csv", index=False, line_terminator="\n", encoding="utf-8")


#-----------------
# entry
#-----------------
if __name__ == "__main__":
    data_dir = input().strip()
    generate_predictions(data_dir)

