# Change Log: submission.py

## Date: 2026-04-30 00:35:43

---

### Version History

### Current Best Score: 79.45% (leaderboard)

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