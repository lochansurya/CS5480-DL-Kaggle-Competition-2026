# Plan: Edge-Map Preprocessing — Targeting 80%

## Constraints
- ❌ No pretrained models allowed (train from scratch)
- ✓ Image size: 224 x 224 pixels (fixed)

## Current Status
- **All-Time Best Score**: **77.53%** (ShapePreprocess Edge-Map + TTA, 2026-05-01) ⭐ NEW BEST
- **Previous Best**: 76.88% (Resnet-34 Ensemble + Train-100, 2026-05-01)
- **Target**: 80%
- **Model**: Custom ResNet-34-like (3+4+6+3 blocks) + CBAM + GeM
- **Next run**: Edge-map preprocessing (`ShapePreprocess`) — not yet submitted

## Root Cause Analysis: The Augmentation Trap

### Why the "Anti-Overfit" Strategy Failed
The recent attempt to heavily regularize the model (MixUp 0.8, CutMix 0.5, Label Smoothing 0.1) actively hurt test performance. The hard ceiling around ~75.5% - 75.9% across multiple architectures is not a capacity limit; it is a **signal extraction problem**.

1. **Destructive Augmentations:** In rendered synthetic images, the crucial differences between Class 0 and Class 1 are crisp, high-frequency geometric details or specific shading patterns. Blurring (GaussianBlur), cutting boxes (RandomErasing), and blending (MixUp) destroyed these exact discriminative features.
2. **Label Smoothing Penalty:** By forcing targets to 0.9 and 0.1, the model was penalized for being confident about clear, distinct synthetic boundaries.
3. **GeM Dominance:** Concatenated pooling introduced too much noise. Generalized Mean (GeM) Pooling has historically proven to be the superior feature aggregator for this specific dataset.

---

## Changes to Apply (The Rollback)

We are reverting to a clean, lightly regularized pipeline optimized for crisp synthetic data.

### 1. Hyperparameter Rollback

| Parameter | "Heavy" Value (Failed) | New Target (Rollback) | Rationale |
|-----------|------------------------|-----------------------|-----------|
| **epochs** | 150 | **100** | Faster convergence required since heavy augmentations are removed. |
| **dropout** | 0.5 | **0.2** | Light regularization to preserve the clean geometric signal. |
| **drop_path_rate** | 0.3 | **0.1** | Reverted to historically best setting from the 75.91% run. |
| **weight_decay** | 1e-3 | **1e-4** | Standard L2 penalty. |
| **mixup_alpha** | 0.8 | **0.0 (Disabled)** | Stop blending crisp geometric edges. |
| **cutmix_prob** | 0.5 | **0.0 (Disabled)** | Stop destroying local features with black boxes. |
| **label_smoothing**| 0.1 | **0.0 (Disabled)** | Allow the model to be 100% confident on synthetic data. |

### 2. Architectural Rollback
- **Restore GeM Pooling**: Revert from `ConcatPool2d` back to `GeM(p=3.0)`.
- **Restore Classifier**: Change the first linear layer input back from 1024 to **512** (to match the GeM output).

### 3. Data Pipeline Adjustments (Clean Renders)
- ❌ Remove `RandomErasing`
- ❌ Remove `GaussianBlur`
- 🔽 Reduce `RandomResizedCrop` scale minimum from `0.7` to **`0.8`** (keep the object mostly intact within the frame).
- 🔽 Reduce `ColorJitter` from `(0.4, 0.4, 0.4, 0.15)` to **`(0.2, 0.2, 0.2, 0.1)`** (lighter color variance).

### 4. Keep Existing
- **TTA (Test Time Augmentation)**: Crucial for squeezing out an extra ~1% by averaging predictions.
- **Optimizer & Scheduler**: AdamW + Cosine Annealing (Warmup: 5 epochs).
- **CBAM**: Continue using Convolutional Block Attention to focus on key structural areas.

---

## Expected Impact
- **Training Speed**: Convergence will be much faster. Training accuracy should hit the mid-90s rapidly.
- **Goal**: Reclaim the 75.91% baseline and push past 76.0% by combining the stable GeM pooling architecture with the optimized dynamic thresholding logic (calculating the optimal threshold on the validation set instead of assuming 0.5).

## Dataset Understanding (CRITICAL)

These are **CLEVR-style rendered scenes** — 4 small geometric objects (cubes, cylinders, spheres) on a plain gray background.

**Classification rule (confirmed by visual inspection):**
- **Class 1**: scene contains at least one **sphere** AND at least one **cube**
- **Class 0**: everything else (no sphere, or sphere with only cylinders but no cube)

**Why prior experiments plateaued at ~76%:**
1. `RandomResizedCrop` could cut the only sphere or cube out of frame → silent label corruption on every epoch
2. Color was never a discriminative feature — same colours appear in both classes — but the model wasted capacity trying to use it
3. `GaussianBlur` destroyed shape curvature (the only true signal)

---

## Current Approach — Edge-Map Preprocessing

### `ShapePreprocess` pipeline (applied to every image, train and test)
```
RGB → Grayscale (L) → Contrast ×3 → FIND_EDGES → RGB (3 identical channels)
```

**Why this works:**
- `FIND_EDGES` extracts shape outlines. A **sphere** produces a smooth **circular/elliptical contour**. A **cube** produces sharp **rectangular corners**. A **cylinder** produces rounded-top, flat-side outlines. The classification signal (sphere + cube present?) is now directly visible in the edge map — no shading, no colour, no background noise.
- Normalisation changed from ImageNet stats to `EDGE_MEAN/STD = [0.5, 0.5, 0.5]` to match the near-binary edge image distribution.

### Active augmentations (train only)
| Transform | Kept? | Reason |
|-----------|-------|--------|
| `Resize(224)` | ✅ | All objects stay in frame |
| `RandomHorizontalFlip` | ✅ | Edge outlines are flip-symmetric |
| `ColorJitter` | ❌ | Irrelevant — input is grayscale edge map |
| `RandomGrayscale` | ❌ | Already grayscale — redundant |
| `RandomResizedCrop` | ❌ | Would remove key objects, corrupting labels |
| `GaussianBlur` | ❌ | Smears edges — destroys the signal |

### Hyperparameters (current)
| Parameter | Value | Note |
|-----------|-------|------|
| `epochs` | 100 | With early stopping (patience=20) |
| `dropout` | 0.2 | Light — edge maps are already regularised |
| `drop_path_rate` | 0.1 | Proven best in prior runs |
| `lr` | 7e-4 | AdamW + cosine warmup |
| `tta` | True | HFlip TTA at val + test |
| `ensemble_seeds` | (42,) | Single seed — faster turnaround |

## Strategy for Final Day (deadline 17:00 IST)
1. ✅ Edge-map preprocessing implemented in `submission.py`.
2. Run `./run.sh data true` — trains 3 seeds (val + full retrain each), auto-submits.
3. Expected finish: ~10:30 AM IST, leaving buffer for a retry if needed.
