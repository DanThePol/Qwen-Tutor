from datasets import load_dataset, concatenate_datasets, DatasetDict
import os
from huggingface_hub import login


HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]

login(token=HF_TOKEN)

ds_sciq = load_dataset(f"{HF_USER}/sciq_with_formatted_cleaned")
ds_aqua = load_dataset(f"{HF_USER}/aqua_cleaned")
ds_medmcqa = load_dataset(f"{HF_USER}/medmcqa_cleaned")
ds_arc = load_dataset(f"{HF_USER}/ai2_arc_challenge_with_rationales")
ds_openbook = load_dataset(f"{HF_USER}/openbookqa_with_rationales")

def add_dataset_field(example):
    example["dataset"] = "aqua_rat"
    return example

ds_aqua = DatasetDict({
    split: ds_aqua[split].map(add_dataset_field)
    for split in ds_aqua
})

val_joint = concatenate_datasets([
    ds_sciq["validation"],
    ds_aqua["validation"],
    ds_medmcqa["validation"],
    ds_arc["validation"],
    ds_openbook["validation"],
])

train_mix = concatenate_datasets(
    [
        ds_sciq["train"],
        ds_aqua["train"],
        ds_medmcqa["train"],
        ds_arc["train"],
        ds_openbook["train"],
    ]
)

ds = DatasetDict({
    "train": train_mix, 
    "validation": val_joint
})

ds_cleaned = DatasetDict({
    split: ds[split].remove_columns(["question_id", "subject_name"]).shuffle(seed=42)
    for split in ds
})

ds_cleaned.push_to_hub(
   f"{HF_USER}/MNLP_M3_mcqa_dataset",
    private=False,
    max_shard_size="500MB"
)