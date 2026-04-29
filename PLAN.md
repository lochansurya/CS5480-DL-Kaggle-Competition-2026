# Plan: Improve Submission Score from 75.9% to >80%

## Constraints (per README.md)
- ❌ No pretrained models allowed (train from scratch)
- ✓ Image size: 224 x 224 pixels (fixed)

## Current Status
- **Current Score**: 75.9%
- **Val Accuracy**: ~99% (severe overfitting)
- **Model**: Custom ResNet-34-like (3+4+6+3 blocks)

## Root Cause Analysis: Why the Gap?

### The Problem: "Memorization" vs "Generalization"
The model achieves ~99% validation accuracy but only 75.9% test accuracy. This 23% gap indicates:

1. **Learns training set noise** - Model memorizes specific training images rather than learning generalizable features
2. **Background features** - May be learning backgrounds that differ between train/test distributions
3. **Insufficient regularization** - Not enough forcing function to learn robust features

### Why Current Regularization Fails:
| Issue | Current Value | Problem |
|-------|------------|---------|
| Dropout | 0.3 | Too weak for 16-block network |
| DropPath | 0.1 | Too conservative |
| Weight Decay | 1e-4 | Allows large weights |
| MixUp | 0.4 | Not aggressive enough |

---

## Changes Applied

### 1. Hyperparameter Tuning (The "Anti-Overfit" Suite)

| Parameter | Old Value | New Value | Why This Helps |
|-----------|----------|----------|---------------|
| **dropout** | 0.3 | 0.5 | Forces classifier to use ALL neurons, not just a few |
| **drop_path_rate** | 0.1 | 0.3 | Prevents deep layers from co-adapting |
| **weight_decay** | 1e-4 | 1e-3 | Keeps weights small and general |
| **mixup_alpha** | 0.4 | 0.8 | Harder samples → broader decision boundaries |
| **label_smoothing** | 0.0 | 0.1 | Softens 0/1 targets → less overconfident |
| **batch_size** | 128 | 64 | More gradient noise = implicit regularization |
| **epochs** | 120 | 150 | Longer to converge with strong regularization |

### 2. Architectural Adjustments

**Why Concatenated Pooling?**
- GeM only captures mean intensity (average pooling)
- Max pooling captures presence of features
- Concatenating both captures MORE information:
  - "Is this feature present?" (Max)
  - "How much of the feature?" (Average)
- Expands 512 → 1024 features = more signal to classifier

**Why Expand Classifier?**
- More input features (1024) need larger first layer
- Previous 512→256 was bottleneck

### 3. Training Logic

**Why Keep Cosine Scheduler?**
- OneCycleLR can be unstable with small batch sizes
- Cosine is tested and stable
- Still finds good minima

### Keep Existing
- **TTA**: Already working, adds ~1-2%
- **CBAM**: Attention helps focus on relevant pixels

## Expected Impact

**Why Validation Accuracy Will Drop:**
- Stronger regularization = harder to achieve 99%
- Expected: ~90-93% validation accuracy
- This is actually GOOD - means less memorization

**Why Test Score Should Improve:**
- Model forced to learn generalizable features
- Can't memorize with dropout + weight decay
- Expected: ~81-83% test score

## Implementation Status
- ✅ dropout: 0.5 applied
- ✅ drop_path_rate: 0.3 applied
- ✅ weight_decay: 1e-3 applied
- ✅ mixup_alpha: 0.8 applied
- ✅ label_smoothing: 0.1 applied
- ✅ batch_size: 64 applied
- ✅ epochs: 150 applied
- ✅ Concatenated Pooling applied
- ✅ Classifier expanded to 1024 features
- ⚠️ OneCycleLR - kept cosine scheduler (works better in practice)