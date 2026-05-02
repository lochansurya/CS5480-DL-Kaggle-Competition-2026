# Grader Instructions

## Team

| Name | Roll Number |
|------|-------------|
| Lochan Surya Teja Neeli | CS25MTECH11015 |
| Harshavardhan Meerjumla | CS25MTECH11011 |

**Best public leaderboard score: 0.775311 (77.53%)**

---

## Requirements

- Python 3.11
- NVIDIA GPU (CPU fallback works but is slow)

```bash
pip install -r requirements.txt
```

> PyTorch must be installed via **pip**, not conda, to support newer GPUs.

---

## Dataset Layout

Place the competition data so it matches this structure:

```
data/
├── train/
│   ├── 0/      # Class 0 images
│   └── 1/      # Class 1 images
└── test/       # Unlabelled test images (flat directory)
```

---

## Running

```bash
python submission.py data
```

Or, if the data directory is elsewhere:

```bash
python submission.py /path/to/data
```

This will:
1. Train Phase 1 (85% split, early stopping, threshold calibration)
2. Train Phase 2 (full 18k images, 100 epochs)
3. Write `submission.csv` (5,010 rows, `ID,TARGET` format)
4. Save training plots to `plots/`


---

## Entry Point

```python
def generate_predictions(data_dir: str) -> None:
    ...

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    generate_predictions(data_dir)
```

---

## Reproducibility

Fixed seeds (`seed=42`, `split_seed=42`) and `torch.backends.cudnn.deterministic=True`
guarantee identical predictions across runs. Expected public score: **≈ 0.7753**.

---

## Output

| File | Description |
|------|-------------|
| `submission.csv` | 5,010 predictions in `ID,TARGET` format |
| `plots/` | Training curves, ROC, PR, confusion matrix, metrics, threshold sensitivity |
| `config.json` | Hyperparameters used |
