# 🧠 MCQA Fine-Tuning Pipeline – `train_mcqa`

This folder contains the complete pipeline to prepare datasets and fine-tune a large language model (**Qwen3-0.6B-Base**) for **Multiple-Choice Question Answering (MCQA)** using Hugging Face and GPT-based rationale generation.

The output model can be pushed to the Hugging Face Hub and used for downstream MCQA tasks. Rationales are generated and cleaned using a GPT API, and training is tracked via Weights & Biases (W&B).

---

## 📁 Folder Structure

```
train_mcqa/
├── prepare_sciq.py          # Prepares SciQ dataset with cleaned rationales
├── prepare_aqua.py          # Prepares AQUA-RAT dataset using GPT-based rationale cleaning
├── prepare_medmcqa.py       # Processes MedMCQA dataset
├── prepare_arc.py           # Formats Arc-Challenge and adds GPT rationales
├── prepare_openbook.py      # Processes OpenBookQA with GPT-based explanations
├── prepare_all_datasets.py  # Merges all cleaned datasets into one
├── train.py                 # Fine-tunes Qwen3-0.6B on the merged dataset
├── train_mcqa.sh           # Full pipeline script: prepare + train + upload
├── requirements.txt         # Dependencies
└── README.md               # This file
```

---

## 🧪 Dataset Preparation

Each `prepare_*.py` script loads a specific MCQA dataset, cleans it (optionally using GPT for rationale generation), and pushes the final version to the Hugging Face Hub.

- **Supported datasets**: **SciQ, AQUA-RAT, MedMCQA, Arc-Challenge, OpenBookQA**
- Final dataset is merged and split into `train` and `validation` sets using `prepare_all_datasets.py`.

---

## 🏋️‍♀️ Model Training

The `train.py` script fine-tunes the **Qwen3-0.6B-Base** model using the Hugging Face `transformers` library and W&B for experiment tracking.

### Hyperparameters
- **Base Model**: `Qwen/Qwen3-0.6B-Base`
- **Max Sequence Length**: 384 tokens
- **Batch Size**: 8 (per device)
- **Gradient Accumulation Steps**: 8 (effective batch size: 64)
- **Learning Rate**: 4e-6
- **Weight Decay**: 0.10
- **Max Epochs**: 5
- **Dropout**: 0.15
- **Label Smoothing**: 0.05
- **LR Scheduler**: Cosine with 8% warmup
- **Optimizer**: AdamW 8-bit
- **Early Stopping**: 3 patience, 5e-4 threshold
- **Precision**: bfloat16
- **Gradient Clipping**: 1.0 max norm

### Dataset Mixing Probabilities
- **SciQ**: 20%
- **AQUA-RAT**: 25%
- **MedMCQA**: 25%
- **Arc-Challenge**: 15%
- **OpenBookQA**: 15%

### Features
- Handles tokenization, config, early stopping, and Hub integration
- Uploads both model and tokenizer to the Hugging Face Model Hub
- Gradient checkpointing for memory efficiency
- Custom padding collator for variable-length sequences

---

## 💻 Usage

### Prerequisites

Install dependencies:
```bash
pip install -r train_mcqa/requirements.txt
```

**For dataset generation** (if you choose to generate the dataset):
You'll need the GPT wrapper installed. Create a folder `_artifacts`, add the wrapper inside and run such command (adapt it with the name of your wrapper):
```bash
pip install _artifacts/gpt_wrapper-0.2.0-py3-none-any.whl
```

### Run Full Pipeline

```bash
bash train_mcqa/train_mcqa.sh
```

The script will first ask: **"Do you want to generate the dataset? (yes/no)"**

#### If you answer **"yes"** (Generate the dataset):
⚠️ Warning: Dataset generation can take several hours due to GPT API calls for rationale generation.

You'll be prompted for:
- 🤗 **Hugging Face credentials** (username and token)
- 🤖 **GPT API credentials** (for rationale generation)
- 📊 **Weights & Biases info** (project name, run name)
- 📁 **Output directory** (where to save the model)
- 🏷️ **Repository name** (for Hugging Face Hub upload)

The script will:
- Generate and clean all datasets using GPT-based rationales when needed
- Merge datasets and create train/validation splits
- Fine-tune the model and push to Hugging Face Hub

#### If you answer **"no"** (Use existing dataset):
The pipeline will use the pre-built dataset: **`mgatti/MNLP_M3_mcqa_dataset`**

You'll only be prompted for:
- 📊 **Weights & Biases info** (project name, run name)
- 📁 **Output directory** (where to save the model)
- 🤗 **Hugging Face token and repository name** (for model upload)

The script will:
- Load the existing cleaned dataset
- Fine-tune the model directly
- Push the final model to Hugging Face Hub

---

## 🔐 Notes on Authentication

This pipeline requires a valid Hugging Face token to push datasets and models to the Hub. You can create one at:

👉 https://huggingface.co/settings/tokens

If not set, the `train_mcqa.sh` script will securely prompt you for it.

---

## Requirements

Dependencies are listed in `requirements.txt`, including:
- `transformers`
- `datasets`
- `huggingface_hub`
- `wandb`
- `tqdm`
- `matplotlib`
- `numpy`
- `gpt_wrapper` (you must provide and install it)

---

## 📁 Outputs

The pipeline will generate:
- Cleaned Hugging Face datasets
- Final merged dataset (train and validation splits)
- Fine-tuned model (locally saved and uploaded to Hugging Face)
- W&B logs for training monitoring

