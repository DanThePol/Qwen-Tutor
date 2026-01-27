# 🔧 Quantized Training Scripts for MCQA – CS-552 M3

This folder contains the full training code used to produce **quantized versions** of the MCQA model (`mgatti/MNLP_M3_mcqa_model`) for the **MNLP-M3 Quantized Evaluation**. Two quantization strategies are included:

---

## 📁 Folder Structure

```
train_quantized/
├── train_quantized_mcqa_qlora_W4A16.py     # QLoRA-based W4A16 quantization (4-bit weights, 16-bit activations)
├── train_quantized_mcqa_optimum_W4A8.py    # Optimum-Quanto W4A8 quantization (4-bit weights, 8-bit activations)
├── train_quantized.sh                      # Shell script to reproduce training (for QLoRA) (!!! This file is outside the train_quantized folder as instructed !!!)
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```

---

## 🧠 Quantization Strategies

### 1. `train_quantized_mcqa_qlora_W4A16.py`

This script fine-tunes the base MCQA model with **QLoRA adapters** using 4-bit quantization (`nf4`) and 16-bit activation precision (`bfloat16`). 

**Details:**
- Uses Hugging Face `transformers`, `peft`, and `bitsandbytes`.
- Loads the dataset.
- Applies LoRA adapters (`r=16`, `alpha=32`) on `q_proj` and `v_proj`.
- Trains for 1 epoch with `batch_size=8`, `grad_accum=4`.
- Saves the merged final model to `./qlora_quantized_mcqa_w4a16`.

**Output HF Model (submission):**
`abdou-u/MNLP_M3_quantized_model` ✅

---

### 2. `train_quantized_mcqa_optimum_W4A8.py`

This script uses **Optimum-Quanto's official API** to quantize the base model **without training**:
- **W4A8 = 4-bit weights + 8-bit activations**
- No LoRA or fine-tuning required.

**Output HF Model:**
`abdou-u/MNLP_M3_w4a8_quantized_mcqa_model` ✅

---

## 💻 Usage

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

For QLoRA training, ensure:
- NVIDIA GPU (A100 preferred)
- `bitsandbytes`, `transformers`, `peft`, `datasets`

For Optimum-Quanto:
```bash
pip install "optimum[quanto]"
```

---

## 🚀 Run Training (QLoRA)

To reproduce the QLoRA training and save the quantized model:

```bash
bash train_quantized.sh
```

This script:
- Loads HF token securely
- Triggers `train_quantized_mcqa_qlora_W4A16.py`
- Saves the trained and merged model to disk

---

## 🔐 Notes on Authentication

You must have a Hugging Face token to load models/datasets. You can generate one from https://huggingface.co/settings/tokens.

---

## 📦 Requirements

See `requirements.txt` for needed packages.

---

## 📁 Outputs

Each script saves:
- Quantized model (merged)
- Tokenizer files
- Metadata JSON with quantization config

---

## 📜 Licensing & Credits

- MCQA model: `mgatti/MNLP_M3_mcqa_model`
- Dataset: `abdou-u/MNLP_M3_quantized_dataset`
- Developed by Ahmed Abdelmalek (EPFL, 2025) for the CS-552 M3 project.
