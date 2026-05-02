# CS5480-DL-Kaggle-Competition-2026

CS5480: Deep Learning Kaggle Competition, 2026

**Course Instructors:**

1. Prof. Srijith P K
2. Prof. Konda Reddy Mopuri

---

## Overview

This repository implements a **binary image classification pipeline** trained **from scratch (no pretrained models)**, as required by the competition.

The pipeline:

* trains on labeled images (`train/0`, `train/1`)
* predicts labels for test images
* generates `submission.csv` for Kaggle leaderboard evaluation

---

## Submission

### Best Public Leaderboard Score

| | |
|---|---|
| **Public score** | **0.775311 (77.53%)** |
| Competition | `iith-deep-learning-2026-hackathon` |
| Submission file | `submission.csv` — 5,010 predictions |
| Training set | 18,000 images (9,000 Class-0 + 9,000 Class-1, perfectly balanced) |

### Team

| Name | Roll Number |
|------|-------------|
| Lochan Surya Teja Neeli | CS25MTECH11015 |
| Harshavardhan Meerjumla | CS25MTECH11008 |

### Approach Summary

A custom **ResNet-34-like CNN** (3+4+6+3 blocks, 16 total residual blocks) trained entirely from scratch with:

| Component | Detail |
|-----------|--------|
| **ShapePreprocess** | RGB → Grayscale → Contrast×3 → `FIND_EDGES` → RGB (deterministic, applied to every image) |
| **CBAM attention** | Channel + spatial gating after every residual block |
| **GeM pooling** | Generalized Mean pooling (`p=3.0`, learnable) replaces average pool |
| **DropPath** | Stochastic depth, linearly scaled 0→0.1 across 16 blocks |
| **TTA** | Horizontal-flip average at both validation and test time |
| **Threshold** | Grid search over [0.05, 0.95] (401 steps) on 15 % held-out val set |
| **Optimiser** | AdamW, lr=7e-4, weight_decay=1e-4, cosine LR + 5-epoch warmup |

**Key dataset insight:** scenes are CLEVR-style 3-D renders.
- Class 1 = scene contains ≥ 1 sphere **AND** ≥ 1 cube
- Class 0 = everything else

The discriminative signal is purely shape (sphere → smooth circular outline; cube → sharp rectangular corners). `ShapePreprocess` exposes this directly, eliminating colour as a spurious feature.

**Two-phase training:**
1. **Phase 1** — train on 85 % split with early stopping (patience 20); calibrate threshold on held-out 15 %.
2. **Phase 2** — retrain on 100 % of data (same seed, 100 epochs); apply Phase-1 threshold to test predictions.

### Score Progression

| Key Change | Public Score |
|------------|-------------|
| ResNet-18 baseline | 0.728678 |
| GeM + Stochastic Depth (ResNet-34) | 0.759102 |
| Heavy reg (MixUp/CutMix) — regressed | 0.752867 |
| Fix crop + TTA enabled | 0.755860 |
| 100 epochs, single seed | 0.768827 |
| **ShapePreprocess edge-map (final)** | **0.775311** |

### Reproducing the Submission

```bash
conda activate cs5480-dl-kaggle
./run.sh data          # generates submission.csv
```

Or generate and auto-submit in one step:

```bash
./run.sh data true
```

Reproducibility is guaranteed by `seed=42`, `split_seed=42`, and `torch.backends.cudnn.deterministic=True`.
Expected output: `submission.csv` with exactly 5,010 rows; public score ≈ 0.7753.

### Assignment Checklist

- [x] No pretrained models — trained from scratch only
- [x] Report (ICML 2025 format): `report/ICML2025_Template/paper.pdf`
- [x] Each member's contribution documented in report §6 Conclusion
- [x] `submission.py` exposes `generate_predictions(data_dir: str)` entry point
- [x] `submission.csv` format: `ID,TARGET` with integer labels (0 or 1), 5,010 rows
- [x] Results reproducible within 3 % tolerance (fixed seeds, deterministic CUDA)
- [x] Training plots saved to `plots/` (training curves, ROC, PR, confusion matrix, metrics, threshold sensitivity, dataset distribution, probability distribution)

---

## Repository Structure

```text
.
├── submission.py          # FINAL entrypoint (must be at root)
├── run.sh                 # automated pipeline (validation + run + submit)
├── src/                   # modular implementation (optional)
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── utils.py
│
├── data/                  # dataset (NOT tracked by git)
│   ├── train/
│   │   ├── 0/
│   │   └── 1/
│   │
│   └── test/
│       ├── *.png
│
├── .githooks/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
conda create -n cs5480-dl-kaggle python=3.11
conda activate cs5480-dl-kaggle

pip install -r requirements.txt
```

---

## Kaggle CLI Setup (Required for Submission)

You need to open the kaggle website and then:

- `Account->Settings->API->create legacy api`


```bash
pip install kaggle

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Kaggle CLI Usage

1. Submit Predictions

---

```bash
kaggle competitions submit \
-c iith-deep-learning-2026-hackathon \
-f submission.csv \
-m "msg"
```

---

2. View Submission History

---

```bash
kaggle competitions submissions \
-c iith-deep-learning-2026-hackathon
```

---

3. View Leaderboard

---

```bash
kaggle competitions leaderboard \
--show \
-c iith-deep-learning-2026-hackathon
```

---

## Notes

* `-c` → competition slug
* `-f` → submission file (`submission.csv`)
* `-m` → message (used to track experiments)

---

## Quick Workflow

```bash
./run.sh
kaggle competitions submit -c iith-deep-learning-2026-hackathon -f submission.csv -m "exp1"
kaggle competitions submissions -c iith-deep-learning-2026-hackathon
```


---

## Dataset Setup

Provide dataset path (`data_dir`) with the following structure:

```text
data_dir/
├── train/
│   ├── 0/
│   └── 1/
│
├── test/
│   ├── *.png
```

⚠️ If a nested `test/test/` directory exists, fix it manually before running.

### ⚠️ Important Notes on Test Data

* Test images are stored in a **flat directory**
* Filenames are **arbitrary (non-sequential)**
* Do NOT assume ordering from `os.listdir()`

Recommended:

```python
files = sorted(os.listdir(test_dir))
```

Failure to preserve filename–prediction mapping will result in incorrect submissions.

---

## Full Workflow (Recommended)

⚠️ Ensure the conda environment is activated before running `run.sh`

The entire pipeline is automated using `run.sh`.

### Run end-to-end pipeline:

```bash
chmod +x run.sh
./run.sh
```

---

### Run with custom dataset path:

```bash
./run.sh /path/to/data_dir
```

---

### Run and submit to Kaggle:

```bash
./run.sh data true
```

---

## What `run.sh` does

* Validates the active conda environment (must be activated manually)
* Installs required dependencies
* Validates dataset structure (warns if `test/test/` exists)
* Ensures Kaggle CLI and API config are set correctly
* Validates `submission.py` interface
* Runs the full pipeline
* Generates `submission.csv`
* Validates CSV format (`ID,TARGET`)
* Optionally submits to Kaggle

---

## Manual Workflow (Alternative)

```bash
conda activate cs5480-dl-kaggle

python submission.py
# enter data_dir when prompted

kaggle competitions submit \
-c iith-deep-learning-2026-hackathon \
-f submission.csv \
-m "submission"
```

---

## Submission Format (STRICT)

The `submission.csv` file contents must be like this:

```text
ID,TARGET
image1.png,0
image2.png,1
```

### Rules

* Header must be exactly `ID,TARGET`
* `TARGET` must be **0 or 1 (integer)**
* No probabilities
* No extra columns
* All test images must be included exactly once
* Number of rows must exactly match the number of test images

---

## Required Code Interface (MANDATORY)

```python
def generate_predictions(data_dir):
    # implementation

if __name__ == "__main__":
    data_dir = input()
    generate_predictions(data_dir)
```

---

## Reproducibility Requirements

* Predictions must match within **3% tolerance**
* Use fixed random seeds
* Avoid nondeterministic behavior
* Do not depend on external files

---

## Methodology

* Model: Custom ResNet-34-like CNN + CBAM + GeM Pooling
* Training: From scratch (no pretrained weights), two-phase (val-split → full retrain)
* Preprocessing: ShapePreprocess edge-map (grayscale → contrast → FIND_EDGES)
* Loss: Binary Cross Entropy with logits
* Optimizer: AdamW (lr=7e-4, cosine LR schedule with 5-epoch warmup)
* Input size: 224 × 224 pixels

---

## Important Constraints

* ❌ Pretrained models are NOT allowed
* ❌ Do not commit dataset or large files
* ❌ Do not include model checkpoints

---

## Notes

- Dataset is not tracked in git

- Large files are ignored via `.gitignore`

- Pre-commit hooks prevent accidental commits

- `submission.py` must run independently

- Python **3.11 is required** (PyTorch / torchvision are not supported on Python 3.14 via conda)

- PyTorch is installed via **pip (not conda)** to support newer GPUs (e.g., RTX 50-series)

- Do NOT mix conda-installed `pytorch` with pip-installed `torch`

- CUDA is handled automatically by PyTorch wheels (no manual CUDA install needed)

- Verify GPU setup:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
---

