#!/usr/bin/env python3
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


#-----------------
# config
#-----------------
@dataclass
class CFG:
    seed: int = 42
    img_size: int = 224
    num_workers: int = 4

    # VAE
    latent_dim: int = 256
    vae_batch_size: int = 32
    vae_epochs: int = 40
    vae_lr: float = 1e-3
    vae_beta: float = 0.5

    # Classifier
    clf_batch_size: int = 512
    clf_epochs: int = 80
    clf_lr: float = 1e-3
    clf_weight_decay: float = 1e-4
    clf_dropout: float = 0.3
    clf_warmup_epochs: int = 5
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    val_split: float = 0.15
    patience: int = 10


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
def get_transforms(img_size: int, train: bool):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


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
        return x * torch.sigmoid(a + m).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


#-----------------
# ResNet-34 + CBAM encoder backbone
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


class ResNetEncoder(nn.Module):
    """ResNet-34 + CBAM backbone, global avg pool, 512-d feature."""
    def __init__(self):
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
        self.pool   = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)  # (B, 512)


#-----------------
# VAE
#-----------------
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 7 to 14
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 14 to 28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1),  # 28 to 56
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1),  # 56 to 112
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 32,   3, 4, stride=2, padding=1),  # 112 to 224
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 512, 7, 7))


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder   = ResNetEncoder()
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.decoder   = VAEDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


#-----------------
# Classifier on latent z
#-----------------
class Classifier(nn.Module):
    def __init__(self, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, z):
        return self.net(z)


#-----------------
# datasets
#-----------------
class UnlabeledDataset(Dataset):
    def __init__(self, *, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class LatentDataset(Dataset):
    def __init__(self, z, labels):
        self.z = z
        self.labels = labels

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.labels[idx]


#-----------------
# VAE training
#-----------------
def _vae_loss(recon, x, mu, logvar, beta):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x_01 = (x * std + mean).clamp(0, 1)
    recon_loss = nn.functional.mse_loss(recon, x_01)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


def train_vae(*, vae, all_paths, cfg: CFG, device):
    dataset = UnlabeledDataset(paths=all_paths,
                               transform=get_transforms(cfg.img_size, train=True))
    loader  = DataLoader(dataset, batch_size=cfg.vae_batch_size, shuffle=True,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=cfg.num_workers > 0)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.vae_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(cfg.vae_epochs):
        vae.train()
        total = 0.0
        for xb in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                recon, mu, logvar = vae(xb)
                loss = _vae_loss(recon, xb, mu, logvar, cfg.vae_beta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        print(f"[VAE] Epoch {epoch+1}/{cfg.vae_epochs} | Loss: {total/len(loader):.4f}")


@torch.no_grad()
def extract_latents(*, vae, paths, cfg: CFG, device):
    dataset = UnlabeledDataset(paths=paths,
                               transform=get_transforms(cfg.img_size, train=False))
    loader  = DataLoader(dataset, batch_size=cfg.vae_batch_size * 2, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)
    vae.eval()
    mus = []
    for xb in loader:
        xb = xb.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            h  = vae.encoder(xb)
            mu = vae.fc_mu(h)
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0)


#-----------------
# classifier training
#-----------------
def _clf_scheduler(optimizer, cfg: CFG):
    def lr_lambda(epoch):
        if epoch < cfg.clf_warmup_epochs:
            return (epoch + 1) / cfg.clf_warmup_epochs
        progress = (epoch - cfg.clf_warmup_epochs) / max(cfg.clf_epochs - cfg.clf_warmup_epochs, 1)
        return max(0.001, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_classifier(*, clf, z, labels, cfg: CFG, device, use_val: bool):
    """Returns (best_val_acc, epochs_run). best_val_acc is None when use_val=False."""
    n = len(z)
    g = torch.Generator().manual_seed(cfg.seed)
    idx = torch.randperm(n, generator=g).tolist()

    if use_val:
        split = int((1 - cfg.val_split) * n)
        tr_loader  = DataLoader(LatentDataset(z[idx[:split]], labels[idx[:split]]),
                                batch_size=cfg.clf_batch_size, shuffle=True)
        val_loader = DataLoader(LatentDataset(z[idx[split:]], labels[idx[split:]]),
                                batch_size=cfg.clf_batch_size, shuffle=False)
    else:
        tr_loader  = DataLoader(LatentDataset(z, labels),
                                batch_size=cfg.clf_batch_size, shuffle=True)
        val_loader = None

    optimizer = torch.optim.AdamW(clf.parameters(), lr=cfg.clf_lr,
                                  weight_decay=cfg.clf_weight_decay)
    scheduler = _clf_scheduler(optimizer, cfg)
    smooth = cfg.label_smoothing

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    epochs_run   = cfg.clf_epochs

    for epoch in range(cfg.clf_epochs):
        clf.train()
        correct, total = 0, 0
        for zb, yb in tr_loader:
            zb, yb = zb.to(device), yb.to(device)
            yb_s = yb * (1 - smooth) + (1 - yb) * (smooth / 2)
            optimizer.zero_grad()
            logits = clf(zb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), cfg.grad_clip)
            optimizer.step()
            correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            total   += yb.numel()
        scheduler.step()

        if val_loader is not None:
            clf.eval()
            vc, vt = 0, 0
            with torch.no_grad():
                for zb, yb in val_loader:
                    zb, yb = zb.to(device), yb.to(device)
                    vc += ((torch.sigmoid(clf(zb)) > 0.5).float() == yb).sum().item()
                    vt += yb.numel()
            val_acc = vc / vt
            print(f"[CLF] Epoch {epoch+1}/{cfg.clf_epochs} | "
                  f"Train: {correct/total:.4f} | Val: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    epochs_run = epoch + 1
                    print(f"Early stopping at epoch {epochs_run} "
                          f"(best val acc: {best_val_acc:.4f})")
                    break
        else:
            print(f"[CLF Full] Epoch {epoch+1}/{epochs_run} | Train: {correct/total:.4f}")

    if best_state is not None:
        clf.load_state_dict(best_state)
        print(f"Restored best classifier (val acc: {best_val_acc:.4f})")

    return best_val_acc if use_val else None, epochs_run


def _find_threshold(*, clf, z_val, labels_val, device):
    clf.eval()
    with torch.no_grad():
        probs = torch.sigmoid(clf(z_val.to(device))).cpu().numpy().flatten()
    labels = labels_val.numpy().flatten()
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        acc = ((probs > t).astype(float) == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    print(f"Best threshold: {best_t:.3f}  Val acc: {best_acc:.4f}  "
          f"Pred-1 ratio: {(probs > best_t).mean():.3f}")
    return best_t


#-----------------
# pipeline
#-----------------
def generate_predictions(data_dir: str):
    cfg = CFG()
    set_seed(seed=cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    data_dir  = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    train_paths, train_labels = [], []
    for label in ["0", "1"]:
        for p in sorted((train_dir / label).iterdir()):
            if p.is_file():
                train_paths.append(p)
                train_labels.append(int(label))
    test_paths = [p for p in sorted(test_dir.iterdir()) if p.is_file()]

    # Phase 1: Train VAE on all images (train + test, no labels needed)
    print("=== Phase 1: Training VAE on all images ===")
    vae = VAE(cfg.latent_dim).to(device)
    train_vae(vae=vae, all_paths=train_paths + test_paths, cfg=cfg, device=device)

    # Phase 2: Extract latent vectors
    print("=== Phase 2: Extracting latent vectors ===")
    z_train = extract_latents(vae=vae, paths=train_paths, cfg=cfg, device=device)
    z_test  = extract_latents(vae=vae, paths=test_paths,  cfg=cfg, device=device)
    labels  = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    # Phase 3: Train classifier with validation + early stopping
    print("=== Phase 3: Training classifier (with validation) ===")
    clf = Classifier(cfg.latent_dim, cfg.clf_dropout).to(device)
    _, epochs_run = train_classifier(clf=clf, z=z_train, labels=labels,
                                     cfg=cfg, device=device, use_val=True)

    # find best threshold on the same val split
    g = torch.Generator().manual_seed(cfg.seed)
    idx   = torch.randperm(len(z_train), generator=g).tolist()
    split = int((1 - cfg.val_split) * len(z_train))
    val_idx = idx[split:]
    threshold = _find_threshold(clf=clf, z_val=z_train[val_idx],
                                labels_val=labels[val_idx], device=device)

    # Phase 4: Retrain classifier on full train set for same number of epochs
    print(f"=== Phase 4: Retraining classifier on full data ({epochs_run} epochs) ===")
    cfg_full = CFG()
    cfg_full.clf_epochs = epochs_run
    clf_full = Classifier(cfg.latent_dim, cfg.clf_dropout).to(device)
    train_classifier(clf=clf_full, z=z_train, labels=labels,
                     cfg=cfg_full, device=device, use_val=False)

    # Phase 5: Predict
    print("=== Phase 5: Predicting ===")
    clf_full.eval()
    with torch.no_grad():
        probs = torch.sigmoid(clf_full(z_test.to(device))).cpu().numpy().flatten()
    preds = (probs > threshold).astype(int)

    pred1 = int(preds.sum())
    print(f"Predictions: {pred1} class-1 ({pred1/len(preds):.3f}), "
          f"{len(preds)-pred1} class-0  threshold={threshold:.3f}")

    df = pd.DataFrame([(p.name, int(pred)) for p, pred in zip(test_paths, preds)],
                      columns=["ID", "TARGET"])
    df.to_csv("submission.csv", index=False)


#-----------------
# entry
#-----------------
if __name__ == "__main__":
    generate_predictions(input().strip())
