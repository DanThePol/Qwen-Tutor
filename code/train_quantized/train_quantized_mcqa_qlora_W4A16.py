#!/usr/bin/env python3
"""
train_quantized_mcqa_qlora_W4A16.py

Quantizes the MCQA model with QLoRA (W4A16), fine-tunes with a randomized 15% subset of MCQA stabilization data, and exports the final model.

Ahmed Abdelmalek | CS-552 M3 | 2025
"""

import os
import json
from datetime import datetime
import warnings

# =============== USER CONFIG SECTION ===============
HF_TOKEN = os.environ.get("HF_TOKEN", None) # set as env variable or leave None to prompt
MODEL_NAME = "mgatti/MNLP_M3_mcqa_model"
DATASET_NAME = "abdou-u/MNLP_M3_quantized_dataset"
OUTPUT_DIR = "./qlora_quantized_mcqa_w4a16"
NUM_EPOCHS = 1
BATCH_SIZE = 8
GRAD_ACCUM = 4
LEARNING_RATE = 2e-5
SEED = 1206
# ===================================================

# ---- Authenticate to Hugging Face ----
from huggingface_hub import login
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    import getpass
    login(token=getpass.getpass("Hugging Face token: "))

# ---- Silence non-critical warnings ----
warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")

# ---- Reproducibility Setup ----
import random
import numpy as np
import torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- Load Dataset and Sample Random 15% ----
from datasets import load_dataset

print(f"[INFO] Loading dataset: {DATASET_NAME}")
raw_ds = load_dataset(DATASET_NAME, split="train")

# ---- Create output directory ----
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Import Transformers/PEFT/Datasets ----
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print(f"[INFO] Loading model: {MODEL_NAME}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
print("[INFO] Model and tokenizer loaded with W4A16 quantization.")


# ---- Prompt/Completion Construction ----
def format_choices(choices):
    """Format choices as 'A. ...' style."""
    letters = [chr(65 + i) for i in range(len(choices))]
    return "\n".join([f"{l}. {c}" for l, c in zip(letters, choices)])

def build_prompt(q, choices):
    """Builds the prompt for a multiple-choice QA example."""
    return f"Question: {q}\nChoices:\n{format_choices(choices)}\nAnswer:"

def build_completion(answer_letter, choices, rationale):
    """Formats the model completion (answer + explanation)."""
    if rationale is None:
        rationale = ""
    answer_idx = ord(answer_letter.upper()) - 65
    if 0 <= answer_idx < len(choices):
        return f"{answer_letter}. {choices[answer_idx]}\nExplanation: {rationale}"
    else:
        return f"{answer_letter}\nExplanation: {rationale}"

def preprocess(example):
    """Tokenizes and masks prompts for causal LM training."""
    prompt = build_prompt(example["question"], example["choices"])
    completion = build_completion(example["answer"], example["choices"], example.get("rationale", ""))
    full_text = prompt + "\n" + completion
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"].squeeze()
    attention_mask = tokenized["attention_mask"].squeeze()
    labels = input_ids.clone()
    prompt_len = len(tokenizer(prompt + "\n")["input_ids"])
    labels[:prompt_len] = -100  # Mask loss on prompt portion
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print("[INFO] Preprocessing and tokenizing dataset...")
tokenized_ds = raw_ds.map(preprocess, batched=False)
print("[INFO] Tokenization complete.")
print("[INFO] Example prompt:\n" + build_prompt(raw_ds[0]['question'], raw_ds[0]['choices']))

# ---- Attach LoRA Adapters ----
print("[INFO] Attaching LoRA adapters...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ---- Define Training Loop ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=20,
    save_steps=100,
    report_to="none"
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator
)

print("[INFO] Starting QLoRA fine-tuning...")
trainer.train()
print("[INFO] Training complete.")

# ---- Merge & Save Final Quantized Model ----
print("[INFO] Merging LoRA adapters and exporting final model...")
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---- Save Metadata ----
print("[INFO] Saving metadata...")
lora_config_metadata = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
metadata = {
    "model_name": MODEL_NAME,
    "dataset": DATASET_NAME,
    "subset_seed": SEED,
    "quantization": "QLoRA W4A16",
    "lora_config": lora_config_metadata,
    "date": datetime.now().isoformat(),
    "output_dir": OUTPUT_DIR
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ QLoRA quantization + fine-tuning complete. Model ready for evaluation.")
