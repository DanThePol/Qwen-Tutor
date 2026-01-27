from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import random
import re
import os
from huggingface_hub import login


HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]


login(token=HF_TOKEN)

sciq = load_dataset("allenai/sciq")

def convert_sciq_example(example, idx):

    question_id = f"sciq_{idx}"
    question = example["question"]
    correct = example["correct_answer"]
    options = [example["distractor1"], example["distractor2"], example["distractor3"], correct]
    random.shuffle(options)
    answer = chr(65 + options.index(correct))  
    return {
        "question_id": question_id,
        "question": question,
        "choices": options,
        "answer": answer,
        "rationale": example.get("support", ""),
        "dataset": "sciq"
    }

formatted_sciq = DatasetDict({
    split: sciq[split].map(
        lambda ex, idx: convert_sciq_example(ex, idx),
        with_indices=True,
        remove_columns=sciq[split].column_names
    )
    for split in sciq
})



def has_rationale(example):
    return bool(example.get("rationale", "").strip())

with_rationale = DatasetDict()
without_rationale = DatasetDict()

for split, ds in formatted_sciq.items():
    with_rationale[split] = ds.filter(has_rationale)
    without_rationale[split] = ds.filter(lambda x: not has_rationale(x))

for split in formatted_sciq:
    print(f"{split.upper()}: With rationale = {len(with_rationale[split])}, Without = {len(without_rationale[split])}")



def print_answer_distribution(dataset, name):
    counts = {k: 0 for k in "ABCD"}
    for a in dataset["answer"]:
        if a in counts:
            counts[a] += 1
    print(f"\n{name} – Answer Distribution:")
    for letter in "ABCD":
        print(f"  {letter}: {counts[letter]}")

for split in with_rationale:
    print_answer_distribution(with_rationale[split], f"{split} (with rationale)")
    print_answer_distribution(without_rationale[split], f"{split} (without rationale)")
    


# Filter: Remove Rationales Containing URLs or Figure References
def contains_url(text):
    return bool(re.search(r"(https?\s*:\s*//|www\s*\.)", text, re.IGNORECASE))

def contains_figure_reference(text):
    return bool(re.search(r"\b(see|as\s+shown\s+in|refer\s+to)\s+Figure\b", text, re.IGNORECASE))

def filter_rationales(dataset):
    kept = []
    dropped_indices = []
    for idx, example in enumerate(dataset):
        rationale = example.get("rationale", "")
        if contains_url(rationale) or contains_figure_reference(rationale):
            dropped_indices.append(idx)
        else:
            kept.append(example)
    print(f"Kept {len(kept)} / {len(dataset)} (dropped {len(dropped_indices)})")
    return kept, dropped_indices

filtered_train, _ = filter_rationales(with_rationale["train"])
filtered_val, _   = filter_rationales(with_rationale["validation"])
filtered_test, _  = filter_rationales(with_rationale["test"])

final_sciq = DatasetDict({
    "train": Dataset.from_list(filtered_train),
    "validation": Dataset.from_list(filtered_val),
    "test": Dataset.from_list(filtered_test),
})

# Shuffle and take ~500 extra examples from train
extra_val = final_sciq["train"].shuffle(seed=42).select(range(150))
new_train = final_sciq["train"].select(range(150, len(final_sciq["train"])))
new_validation = concatenate_datasets([final_sciq["validation"], extra_val])

new_final_sciq = DatasetDict({
    "train": new_train,
    "validation": new_validation,
    "test": final_sciq["test"],
})

new_final_sciq.push_to_hub(
    f"{HF_USER}/sciq_with_formatted_cleaned",
    private=False,
    max_shard_size="500MB"
)