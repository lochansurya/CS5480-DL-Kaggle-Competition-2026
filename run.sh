#!/usr/bin/env bash

# =========================
# CS5480 PIPELINE (ROBUST)
# =========================

set -euo pipefail
trap 'echo "❌ Error at line $LINENO"; exit 1' ERR

ENV_NAME="cs5480-dl-kaggle"

# -------------------------
# 0. Flags
# -------------------------
SHOW_QUOTA="false"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage:"
    echo "  ./run.sh [data_dir] [auto_submit]"
    echo ""
    echo "Options:"
    echo "  --help        Show this help message"
    echo "  --quota       Show remaining Kaggle submissions today"
    echo ""
    echo "Examples:"
    echo "  ./run.sh"
    echo "  ./run.sh data"
    echo "  ./run.sh data true"
    echo "  ./run.sh --quota"
    exit 0
fi

if [[ "${1:-}" == "--quota" ]]; then
    SHOW_QUOTA="true"
    DATA_DIR="data"
else
    DATA_DIR=${1:-"data"}
fi

AUTO_SUBMIT=${2:-"false"}

echo "========== START =========="

# -------------------------
# 1. Check Conda Env
# -------------------------
echo "[1] Checking environment..."

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    echo "❌ Expected env: $ENV_NAME"
    echo "👉 Run: conda activate $ENV_NAME"
    exit 1
fi

echo "✅ Environment OK: $CONDA_DEFAULT_ENV"

# -------------------------
# 2. Check Python
# -------------------------
echo "[2] Checking Python..."

python - <<EOF
import sys
assert sys.version_info[:2] == (3,11), f"Python must be 3.11, got {sys.version}"
EOF

echo "✅ Python version OK"

# -------------------------
# 3. Install Dependencies (only if missing)
# -------------------------
echo "[3] Checking dependencies..."

if [[ ! -f ".deps_installed" ]]; then
    pip install -r requirements.txt
    touch .deps_installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed (skipped)"
fi

# -------------------------
# 4. Dataset Validation
# -------------------------
echo "[4] Validating dataset..."

if [[ ! -d "$DATA_DIR/train/0" || ! -d "$DATA_DIR/train/1" ]]; then
    echo "❌ Missing train/0 or train/1"
    exit 1
fi

if [[ -d "$DATA_DIR/test/test" ]]; then
    echo "⚠️ Nested test/test detected"
    echo "👉 Fix manually:"
    echo "   mv $DATA_DIR/test/test/* $DATA_DIR/test/"
    echo "   rm -r $DATA_DIR/test/test"
fi

if [[ ! -d "$DATA_DIR/test" ]]; then
    echo "❌ Missing test directory"
    exit 1
fi

TEST_COUNT=$(find "$DATA_DIR/test" -type f | wc -l)

if [[ "$TEST_COUNT" -lt 1 ]]; then
    echo "❌ No test images found"
    exit 1
fi

echo "✅ Dataset structure OK ($TEST_COUNT test images)"

# -------------------------
# 5. Check submission.py
# -------------------------
echo "[5] Validating submission.py..."

if [[ ! -f "submission.py" ]]; then
    echo "❌ submission.py missing"
    exit 1
fi

grep -q "def generate_predictions(data_dir)" submission.py \
    || { echo "❌ Missing required function"; exit 1; }

echo "✅ submission.py valid"

# -------------------------
# 6. Check Kaggle CLI
# -------------------------
echo "[6] Checking Kaggle CLI..."

if ! command -v kaggle >/dev/null 2>&1; then
    echo "⚠️ Kaggle CLI not found → installing..."
    pip install kaggle
fi

KAGGLE_CONFIG="$HOME/.kaggle/kaggle.json"

if [[ ! -f "$KAGGLE_CONFIG" ]]; then
    echo "❌ Missing ~/.kaggle/kaggle.json"
    exit 1
fi

chmod 600 "$KAGGLE_CONFIG"

echo "✅ Kaggle setup OK"

# -------------------------
# X. Submission Quota
# -------------------------
if [[ "$SHOW_QUOTA" == "true" ]]; then
    echo "[Quota] Checking submissions today..."

    TODAY=$(date +%Y-%m-%d)

    COUNT=$(kaggle competitions submissions \
        -c iith-deep-learning-2026-hackathon \
        | grep "$TODAY" | wc -l || true)

    LIMIT=5
    LEFT=$((LIMIT - COUNT))

    if [[ "$LEFT" -lt 0 ]]; then
        LEFT=0
    fi

    echo "Submissions today: $COUNT / $LIMIT"
    echo "Remaining: $LEFT"

    exit 0
fi

# -------------------------
# 7. Run Pipeline
# -------------------------
echo "[7] Running submission pipeline..."

echo "$DATA_DIR" | python submission.py

echo "✅ Pipeline executed"

# -------------------------
# 8. Validate Output
# -------------------------
echo "[8] Validating submission.csv..."

if [[ ! -f "submission.csv" ]]; then
    echo "❌ submission.csv not generated"
    exit 1
fi

HEADER=$(head -n 1 submission.csv)

if [[ "$HEADER" != "ID,TARGET" ]]; then
    echo "❌ Wrong header: $HEADER"
    exit 1
fi

ROW_COUNT=$(($(wc -l < submission.csv) - 1))

if [[ "$ROW_COUNT" -ne "$TEST_COUNT" ]]; then
    echo "❌ Row mismatch: expected $TEST_COUNT got $ROW_COUNT"
    exit 1
fi

echo "✅ submission.csv valid ($ROW_COUNT rows)"

# -------------------------
# 9. Optional Submit
# -------------------------
if [[ "$AUTO_SUBMIT" == "true" ]]; then
    echo "[9] Submitting to Kaggle..."

    kaggle competitions submit \
        -c iith-deep-learning-2026-hackathon \
        -f submission.csv \
        -m "auto submission"

    echo "✅ Submission complete"
fi

# -------------------------
# DONE
# -------------------------
echo "========== DONE =========="

