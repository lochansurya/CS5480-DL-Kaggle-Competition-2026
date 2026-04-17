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

* Model: Custom CNN
* Training: From scratch (no pretrained weights)
* Loss: Binary Cross Entropy with logits
* Optimizer: Adam
* Input size: 128×128

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

