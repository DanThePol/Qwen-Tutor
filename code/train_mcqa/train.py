
from pathlib import Path
import argparse, random, re
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict, interleave_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import wandb
import os 
from huggingface_hub import login

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]

login(token=HF_TOKEN)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--hub_repo_id", type=str, required=False,
                    help="Hugging Face Hub repo ID (e.g. username/model-name)")
args = parser.parse_args()

HUB_REPO_ID = args.hub_repo_id

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
MAX_LEN         = 384
BATCH_SIZE      = 8           
GRAD_ACC        = 8            
LR              = 4e-6
WEIGHT_DECAY    = 0.10
EPOCHS_MAX      = 5
SEED            = 42
DROPOUT         = 0.15
LABEL_SMOOTH    = 0.05

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = Path(args.output_dir)

data = load_dataset(f"{HF_USER}/MNLP_M3_mcqa_dataset")

# Filter by dataset name using the `dataset` field
datasets_to_split = ["sciq", "aqua_rat", "medmcqa", "ai2_arc_challenge", "openbookqa"]
split_datasets = {
    name: data["train"].filter(lambda x: x["dataset"] == name)
    for name in datasets_to_split
}


probs = [0.20, 0.25, 0.25, 0.15, 0.15]
train_mix = interleave_datasets(
    [split_datasets[name] for name in datasets_to_split],
    probabilities=probs,
    seed=42,
    stopping_strategy="all_exhausted"
)

ds = DatasetDict({
    "train": train_mix,
    "validation": data["validation"]  
})


# Prompt formatter and tokeniser 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_mcqa(example):
    LETTERS = [" A", " B", " C", " D"]
    choices = "".join(f"{l}. {c}\n" for l, c in zip(LETTERS, example["choices"]))
    prompt = (
        "The following are multiple-choice questions (with answers) about "
        "advanced STEM knowledge.\n\n"
        f"Question: {example['question'].strip()}\n{choices}\nAnswer:"
    )
    idx = ord(example["answer"].strip().upper()) - ord("A")
    completion = (
        f"{LETTERS[idx]}. {example['choices'][idx]}\n"
        f"Explanation: {example['rationale'].strip()}"
    )
    return {"prompt": prompt, "completion": completion}

def tok(ex):
    prompt_ids = tokenizer(ex["prompt"])["input_ids"]
    avail      = MAX_LEN - len(prompt_ids)
    comp_ids   = tokenizer(ex["completion"] + tokenizer.eos_token)["input_ids"][:avail]
    ids        = prompt_ids + comp_ids
    labels     = [-100]*len(prompt_ids) + comp_ids
    ids, labels = ids[:MAX_LEN], labels[:MAX_LEN]
    return {
        "input_ids": ids,
        "labels":    labels,
        "attention_mask": [1]*len(ids)
    }

print("Tokenising …")
columns_before = ds['train'].column_names
ds = ds.map(format_mcqa).map(tok, remove_columns=columns_before, num_proc=8)

# Data collator 
class PadCollator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id
    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        for b in batch:
            pad = m - len(b["input_ids"])
            b["input_ids"].extend([self.pad]*pad)
            b["labels"].extend([-100]*pad)
            b["attention_mask"].extend([0]*pad)
        return {k: torch.tensor([b[k] for b in batch]) for k in batch[0]}

collator = PadCollator(tokenizer)

# Model 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, device_map="auto"
)
model.config.dropout = DROPOUT

# TrainingArguments 
steps_per_epoch = len(ds["train"]) // (BATCH_SIZE * GRAD_ACC)

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACC,
    bf16                        = True,
    learning_rate               = LR,
    weight_decay                = WEIGHT_DECAY,
    lr_scheduler_type           = "cosine",
    warmup_steps                = int(0.08 * steps_per_epoch * EPOCHS_MAX),
    num_train_epochs            = EPOCHS_MAX,
    eval_strategy               = "steps",
    eval_steps                  = steps_per_epoch // 2,   
    save_strategy               = "steps",
    save_steps                  = steps_per_epoch // 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    label_smoothing_factor      = LABEL_SMOOTH,
    logging_steps               = 25,
    gradient_checkpointing      = True,
    max_grad_norm               = 1.0,
    seed                        = SEED,
    report_to                   = ["wandb"],
    run_name                    = os.getenv("WANDB_NAME", "run_name_fallback"),
    save_total_limit            = 1,
    optim                       = "adamw_8bit",
    dataloader_num_workers      = 8,
)

# W&B 
wandb.init(
    project=os.getenv("WANDB_PROJECT", "default_project"),
    name=os.getenv("WANDB_NAME", "run_name_fallback"),
    config={
        "batch_size"   : BATCH_SIZE,
        "grad_acc_steps": GRAD_ACC,
        "lr"           : LR,
        "epochs_max"   : EPOCHS_MAX,
        "dropout"      : DROPOUT,
        "label_smooth" : LABEL_SMOOTH,
        "weight_decay" : WEIGHT_DECAY,
        "max_len"      : MAX_LEN,
        "seed"         : SEED,
    }
)

# Trainer + early stopping 
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = ds["train"],
    eval_dataset    = ds["validation"],
    tokenizer       = tokenizer,
    data_collator   = collator,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience  = 3,      
        early_stopping_threshold = 5e-4
    )]
)

# Train 
print("Starting training …")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Model saved to {OUTPUT_DIR}")


if HUB_REPO_ID:
    print(f"Pushing model and tokenizer to Hugging Face Hub: {HUB_REPO_ID}")
    model.push_to_hub(HUB_REPO_ID)
    tokenizer.push_to_hub(HUB_REPO_ID)
