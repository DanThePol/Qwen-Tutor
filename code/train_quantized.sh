#!/usr/bin/env bash
# train_quantized.sh - Train quantized MCQA model with QLoRA (W4A16)
set -euo pipefail

# -------------------- PROJECT ROOT AUTO-DETECT --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
echo "[INFO] Project root detected as: $PROJECT_ROOT"

# -------------------- PYTHON VERSION CHECK --------------------
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.10"
if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
  echo "[ERROR] Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
  exit 1
fi

# -------------------- ENVIRONMENT SETUP --------------------
cd "${SCRIPT_DIR}/train_quantized"

echo "[INFO] Installing Python requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# -------------------- HF TOKEN SETUP --------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN not set. Please export your Hugging Face token as HF_TOKEN."
  read -s -p "Enter Hugging Face token: " HF_TOKEN
  export HF_TOKEN
  echo -e "\n[INFO] HF_TOKEN successfully set."
else
  echo "[INFO] HF_TOKEN already set in environment."
fi

# -------------------- TRAINING --------------------
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOGFILE="train_quantized_mcqa_qlora_W4A16_${TIMESTAMP}.log"

echo "[INFO] Starting QLoRA MCQA quantization pipeline..."
python3 train_quantized_mcqa_qlora_W4A16.py 2>&1 | tee "${LOGFILE}"

if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
  echo "[INFO] ✅ Training complete. Log saved at: ${LOGFILE}"
else
  echo "[ERROR] ❌ Training script failed! Check the log at: ${LOGFILE}" >&2
  exit 1
fi
