#!/usr/bin/env bash
#
# Download dataset for BANASPATI Multimodal RAG project.
# Uses gdown to pull files from Google Drive.
#
# Usage:
#   bash scripts/download_data.sh          # download + extract
#   bash scripts/download_data.sh --skip   # skip download if data/ already exists

set -euo pipefail

DATA_DIR="data"
DOWNLOAD_DIR="${DATA_DIR}/downloads"

# Google Drive file IDs (from Soal.md)
DATABASE_ZIP_ID="1DboLrRCgom60v85ygac0DqmA7yVRmbIs"
QUESTIONS_CSV_ID="1ctqCBx8A4QU63MGJ0D5bTM2bDfC5Ckx4"

DATABASE_ZIP="${DOWNLOAD_DIR}/database-modul5-ai.zip"
QUESTIONS_CSV="${DATA_DIR}/banaspati_eval_questions.csv"

# ── check skip flag ──────────────────────────────────────────────
if [[ "${1:-}" == "--skip" ]] && [[ -d "${DATA_DIR}/database" ]] && [[ -f "${QUESTIONS_CSV}" ]]; then
    echo "Dataset already present in ${DATA_DIR}/. Skipping download."
    exit 0
fi

# ── ensure gdown is installed ───────────────────────────────────
if ! command -v gdown &>/dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# ── create directories ──────────────────────────────────────────
mkdir -p "${DOWNLOAD_DIR}"

# ── download database zip ───────────────────────────────────────
if [[ ! -f "${DATABASE_ZIP}" ]]; then
    echo "Downloading database-modul5-ai.zip..."
    gdown "https://drive.google.com/uc?id=${DATABASE_ZIP_ID}" -O "${DATABASE_ZIP}"
else
    echo "database-modul5-ai.zip already downloaded."
fi

# ── download eval questions ─────────────────────────────────────
if [[ ! -f "${QUESTIONS_CSV}" ]]; then
    echo "Downloading banaspati_eval_questions.csv..."
    gdown "https://drive.google.com/uc?id=${QUESTIONS_CSV_ID}" -O "${QUESTIONS_CSV}"
else
    echo "banaspati_eval_questions.csv already downloaded."
fi

# ── extract zip ─────────────────────────────────────────────────
if [[ -f "${DATABASE_ZIP}" ]] && [[ ! -d "${DATA_DIR}/database" ]]; then
    echo "Extracting database..."
    unzip -q -o "${DATABASE_ZIP}" -d "${DATA_DIR}"
    # Some zips nest a folder; flatten if needed
    EXTRACTED="${DATA_DIR}/database-modul5-ai"
    if [[ -d "${EXTRACTED}" ]]; then
        mv "${EXTRACTED}"/* "${DATA_DIR}/database/" 2>/dev/null || true
        rmdir "${EXTRACTED}" 2>/dev/null || true
    fi
    echo "Extracted to ${DATA_DIR}/database/"
else
    echo "Database already extracted."
fi

echo ""
echo "Done. Dataset is ready:"
echo "  Documents  : ${DATA_DIR}/database/"
echo "  Eval CSV   : ${QUESTIONS_CSV}"
echo ""
echo "Next step: python scripts/data_pipeline.py prepare --input data/database --output artifacts"
