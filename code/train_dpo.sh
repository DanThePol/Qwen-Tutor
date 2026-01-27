#!/bin/bash

# Simple DPO Training Script
# EDIT THE LINE BELOW TO SET YOUR HUGGING FACE REPO:
# Leave empty ("") to save locally only, or set to "username/model-name" to upload
HF_REPO=""  # Example: "username/my-dpo-model"

# Fixed configuration (do NOT modify)
BASE_MODEL="Qwen/Qwen3-0.6B-Base"
DATASET="albertfares/MNLP_M3_dpo_dataset"
OUTPUT_DIR="./dpo_model"

echo "📦 Installing requirements..."
pip install -r train_dpo/requirements.txt

echo "Training DPO model..."
echo "Base Model: $BASE_MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
if [ -n "$HF_REPO" ]; then
    echo "Will upload to: $HF_REPO"
fi

cd train_dpo
python fdpo_training.py --base_model "$BASE_MODEL" --dataset "$DATASET" --output_dir "$OUTPUT_DIR" --hf_repo "$HF_REPO"
