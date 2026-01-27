#!/bin/bash
set -e

echo "[INFO] Welcome to the MCQA Training Pipeline"

# --- Ask whether to generate or reuse datasets ---
read -p "Do you want to generate the dataset? (yes/no): " GENERATE_DATA

if [[ "$GENERATE_DATA" == "no" || "$GENERATE_DATA" == "n" ]]; then
  echo "[INFO] Using preprocessed datasets."
  export HF_USER="mgatti"

  # Ask only what's needed for training and model upload
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[WARN] HF_TOKEN not set."
    read -s -p "Enter Hugging Face token: " HF_TOKEN
    echo
    export HF_TOKEN
  fi

  if [[ -z "${WANDB_NAME:-}" ]]; then
    read -p "Enter W&B run name: " WANDB_NAME
    export WANDB_NAME
  fi

  if [[ -z "${OUTPUT_DIR:-}" ]]; then
    read -p "Enter output directory path: " OUTPUT_DIR
    export OUTPUT_DIR
  fi

  if [[ -z "${HF_MODEL_REPO:-}" ]]; then
    read -p "Enter HF model repo name (e.g. username/model_name): " HF_MODEL_REPO
    export HF_MODEL_REPO
  fi

else
  echo "[INFO] You chose to generate datasets from scratch."

  # --- Hugging Face credentials ---
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[WARN] HF_TOKEN not set."
    read -s -p "Enter Hugging Face token: " HF_TOKEN
    echo
    export HF_TOKEN
  fi

  if [[ -z "${HF_USER:-}" ]]; then
    read -p "Enter your Hugging Face username: " HF_USER
    export HF_USER
  fi

  # --- GPT API credentials ---
  if [[ -z "${GPT_API_BASE:-}" ]]; then
    read -p "Enter GPT API base URL: " GPT_API_BASE
    export GPT_API_BASE
  fi

  if [[ -z "${GPT_API_KEY:-}" ]]; then
    read -s -p "Enter GPT API key: " GPT_API_KEY
    echo
    export GPT_API_KEY
  fi

  # --- Weights & Biases ---
  if [[ -z "${WANDB_PROJECT:-}" ]]; then
    read -p "Enter W&B project name: " WANDB_PROJECT
    export WANDB_PROJECT
  fi

  if [[ -z "${WANDB_NAME:-}" ]]; then
    read -p "Enter W&B run name: " WANDB_NAME
    export WANDB_NAME
  fi

  # --- Output and Repo Info ---
  if [[ -z "${OUTPUT_DIR:-}" ]]; then
    read -p "Enter output directory path: " OUTPUT_DIR
    export OUTPUT_DIR
  fi

  if [[ -z "${HF_MODEL_REPO:-}" ]]; then
    read -p "Enter HF model repo name (e.g. username/model_name): " HF_MODEL_REPO
    export HF_MODEL_REPO
  fi
fi



echo "[INFO] Installing requirements..."
pip install -r train_mcqa/requirements.txt

# --- Step 1: Prepare datasets ---
if [[ "$GENERATE_DATA" != "no" && "$GENERATE_DATA" != "n" ]]; then
  echo "[INFO] Generating and uploading datasets to Hugging Face Hub..."
  python3 train_mcqa/prepare_sciq.py
  python3 train_mcqa/prepare_aqua.py
  python3 train_mcqa/prepare_medmcqa.py
  python3 train_mcqa/prepare_arc.py
  python3 train_mcqa/prepare_openbook.py
  python3 train_mcqa/prepare_all_datasets.py
else
  echo "[INFO] Skipping dataset preparation. Reusing datasets from: $HF_USER"
fi

# --- Step 2: Train model ---
echo "[INFO] Starting training..."
python3 train_mcqa/train.py \
    --output_dir "$OUTPUT_DIR" \
    --hub_repo_id "$HF_MODEL_REPO"
