# Change Log: submission.py

### Current Best Public Score: **76.88%** (2026-05-01) — previous best 75.91%

---

## 2026-05-01 — Single Seed for Fast Turnaround

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `ensemble_seeds` | `(42, 43, 44)` | `(42,)` | Deadline pressure — ~⅓ runtime, single val+retrain cycle |

---

## 2026-05-01 — Edge-Map Preprocessing (`ShapePreprocess`)

### Motivation
Visual inspection of the dataset confirmed the classification rule:
- **Class 1** = scene has ≥1 sphere AND ≥1 cube
- **Class 0** = everything else

The discriminative signal is **purely shape** (sphere curvature vs. angular cube/cylinder outlines). All prior experiments used RGB images where the model had to learn this from shading — slower to converge and vulnerable to colour spurious correlations.

### New `ShapePreprocess` class
Applied deterministically to every image (train + val + test):
```
RGB → Grayscale → Contrast ×3.0 → FIND_EDGES → RGB (3 identical channels)
```
Converts each scene into a **shape outline map**: sphere appears as a smooth elliptical contour, cube as a sharp rectangular outline — directly encoding the class signal.

### Changes to `submission.py`
| Component | Before | After |
|-----------|--------|-------|
| `PIL` import | `Image` | `Image, ImageEnhance, ImageFilter` |
| `MEAN / STD` | ImageNet `[0.485…]` | `EDGE_MEAN/STD = [0.5, 0.5, 0.5]` (suits near-binary edge maps) |
| `ShapePreprocess` | absent | **new class** (contrast + `FIND_EDGES`) |
| `get_transforms` train | `Resize + HFlip + RandomGrayscale + ColorJitter` | `ShapePreprocess + Resize + HFlip` |
| `get_transforms` val/test | `Resize` | `ShapePreprocess + Resize` |
| `predict_probs` normalisation | raw ImageNet stats | `ShapePreprocess + EDGE_MEAN/STD` |

### No changes to model architecture, CFG, or training loop.

---

## 2026-05-01 — Fix Label-Corrupting Crop + Shape-Aware Augmentation

### Root Cause Identified
Dataset is CLEVR-style rendered scenes. Class 1 = scene has sphere + cube; Class 0 = everything else.
`RandomResizedCrop` could cut the only sphere or cube out of frame, silently mislabeling the crop.
This was the primary cause of the ~76% accuracy ceiling across all experiments.

### Augmentation Changes
| Transform | Before | After | Reason |
|-----------|--------|-------|--------|
| `RandomResizedCrop(scale=0.8)` | present | **removed** | Crops can exclude key objects, corrupting labels |
| `Resize(img_size)` | val-only | **train + val** | Keep all 4 objects always in view |
| `RandomGrayscale(p=0.3)` | absent | **added** | Color is irrelevant to class; forces shape-based learning |
| `ColorJitter hue` | `0.1` | **`0.0`** | Hue change is pure noise for this task |

### No architecture or hyperparameter changes.

---

## 2026-05-01 — Light Augmentation Rollback + TTA Enable

### Hyperparameter Changes
| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `tta` | `False` | `True` | Enables hflip TTA at val + test time |
| `dropout` | `0.3` | `0.2` | Lighter regularization for synthetic data |
| `epochs` | `80` | `100` | Slower convergence without heavy aug |
| `patience` | `15` | `20` | More headroom for best checkpoint |

### Augmentation Changes
| Transform | Before | After |
|-----------|--------|-------|
| `RandomResizedCrop scale` | `(0.7, 1.0)` | `(0.8, 1.0)` |
| `ColorJitter` | `(0.4, 0.4, 0.4, 0.15)` | `(0.2, 0.2, 0.2, 0.1)` |
| `GaussianBlur(p=0.2)` | present | **removed** |

### No structural changes — same Model, ImageDataset, generate_predictions logic.

---

## 2026-04-30 00:35:43 - Major Update for >80% Target

### Hyperparameter Changes
| Parameter | Before | After | Rationale |
|-----------|--------|-------|---------|
| epochs | 120 | 150 | More training to converge with stronger regularization |
| weight_decay | 5e-4 | 1e-3 | Stronger L2 penalty |
| dropout | 0.4 | 0.5 | Force diverse neuron usage |
| drop_path_rate | 0.2 | 0.3 | Stronger stochastic depth |
| label_smoothing | 0.0 | 0.1 | Prevent overconfidence |
| mixup_alpha | 0.4 | 0.8 | Harder interpolated samples |
| optimizer | sgd | adamw | Better convergence |
| patience | 20 | 25 | Allow more epochs without improvement |

### Architecture Changes
- **Pooling**: GeM → ConcatPool2d (Adaptive Concatenated Pooling)
  - Concatenates Global Average + Global Max Pooling
  - Output: 512 → 1024 features
- **Classifier**: Expanded to handle 1024 input features

### New Features Added
- CutMix augmentation (50% probability when using mixup)
- MixUp augmentation (α=0.8)

---

## 2026-04-23 21:59:03 - ResNet-18 Implementation

### Changes
- Replaced custom 16-block architecture with ResNet-18 (2,2,2,2 blocks = 8 total)
- Simplified classifier (removed intermediate layer)
- Added TTA by default
- Added docstrings to all functions and classes

### Score: 75.9%

---

## 2026-04-23 19:38:18 - Block Comments & Config Save

### Changes
- Added block style comments (#-----...----#) to all sections
- Added CFG.save() method to save config.json
- Added save_dataset_info() to save dataset.json

---

## Original Implementation

### Architecture
- Custom ResNet-34-like (3+4+6+3 = 16 blocks)
- CBAM attention after each block
- GeM pooling
- DropPath (stochastic depth)

### Features
- TTA (test-time augmentation)
- Cosine LR scheduler with warmup
- Mixed precision training (AMP)
- Early stopping

---

## NOTES

### Why the gap (99% val vs 75.9% test)?
- Model memorizes training data instead of learning generalizable features
- No pretrained ImageNet weights (competition rule)
- Insufficient regularization for from-scratch training

### What helps:
- Stronger regularization (dropout, weight decay, drop_path)
- CutMix/MixUp augmentation
- Concatenated pooling (captures more feature information)
- Longer training with patience