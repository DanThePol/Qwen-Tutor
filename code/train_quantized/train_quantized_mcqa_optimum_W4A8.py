#!/usr/bin/env python3
"""
train_quantized_mcqa_optimum_W4A8.py

Quantizes the MCQA model with Optimum-Quanto (W4A8: 4-bit weights, 8-bit activations).
This script documents the process used to generate the released model
'abdou-u/MNLP_M3_w4a8_quantized_mcqa_model' on Hugging Face Hub.

- Model: mgatti/MNLP_M3_mcqa_model
- Quantization: W4A8 (Optimum-Quanto official API)
- Output: w4a8_quantized_mcqa_final/ (local folder)

Author: Ahmed Abdelmalek | CS-552 M3 | 2025
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.quanto.models import QuantizedModelForCausalLM

# ====================== CONFIG ======================
MODEL_NAME = "mgatti/MNLP_M3_mcqa_model"
LOCAL_SAVE_PATH = "./w4a8_quantized_mcqa_final"
# ====================================================

def main():
    print(f"[INFO] Quantizing {MODEL_NAME} to W4A8 (4-bit weights, 8-bit activations) using Optimum-Quanto...")

    # Load original model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Perform quantization
    quantized_model = QuantizedModelForCausalLM.quantize(
        model,
        weights="qint4",     # 4-bit quantization for weights
        activations="qint8"  # 8-bit quantization for activations
    )

    # Save locally
    os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)
    quantized_model.save_pretrained(LOCAL_SAVE_PATH)
    tokenizer.save_pretrained(LOCAL_SAVE_PATH)

    print(f"[INFO] Quantized model and tokenizer saved to {LOCAL_SAVE_PATH}.")
    print("✅ W4A8 quantization and export complete. Model ready for upload/evaluation!")

if __name__ == "__main__":
    main()
