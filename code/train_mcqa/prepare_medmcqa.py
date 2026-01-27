import re, random, hashlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, concatenate_datasets, disable_caching
from transformers import AutoTokenizer
import os
from huggingface_hub import login


HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]

login(token=HF_TOKEN)

disable_caching()

# Model Config 
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MAX_LEN = 512
VALIDATION_SIZE = 4000

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

CHOICE_LETTERS = [" A", " B", " C", " D"]
VALID_SUBJECTS = {
    **{s: 0.10 for s in ["Anatomy", "Physiology", "Biochemistry", "Pathology", "Microbiology", "Pharmacology"]},
    "Medicine": 0.07, "Surgery": 0.04, "Pediatrics": 0.04,
    "Gynaecology & Obstetrics": 0.03, "Psychiatry": 0.03, "Skin": 0.03,
    "Ophthalmology": 0.02, "Orthopaedics": 0.02, "Anaesthesia": 0.02,
    "Radiology": 0.02, "ENT": 0.01, "Social & Preventive Medicine": 0.05,
}

# Helpers 
def convert_medmcqa(example, idx, dataname="medmcqa"):
    if example.get('choice_type') != 'single': return None
    try:
        cop = int(example.get('cop', 0))
        if not (0 <= cop <= 3): return None
    except Exception: return None

    options = [example.get(k, '') for k in ['opa', 'opb', 'opc', 'opd']]
    correct_option = options[cop]
    shuffled = options[:]
    random.shuffle(shuffled)
    correct_idx = shuffled.index(correct_option)
    return {
        "question_id": f"{dataname}_{idx}",
        "question": example['question'],
        "choices": shuffled,
        "answer": chr(65 + correct_idx),
        "rationale": example['exp'],
        "subject_name": example['subject_name'],
        "dataset": dataname
    }

def is_ok(example):
    return (
        example["answer"] in {"A", "B", "C", "D"} and
        example["question"] and
        len(example["choices"]) == 4 and
        all(example["choices"])
    )

def normalize(text): return re.sub(r"\s+", " ", text.strip())

def make_hash(ex):
    return hashlib.md5("¶".join([normalize(ex["question"])] + [normalize(c) for c in ex["choices"]]).encode()).hexdigest()

def has_rationale(example):
    return isinstance(example["rationale"], str) and example["rationale"].strip() != ""

def no_reference_rationale(example):
    rationale = example["rationale"].lower()
    patterns = [
        r"\bref(?:er)?\b", r"\bchapter\b\s*\d+", r"\bp(?:ages?|g)?\.?\s*\d+",
        r"\bvolume\b\s*\d+", r"\b(?:[1-9][0-9]?)(?:/?e|th edition)\b",
        r"http[s]?://", r"\bwww\.", r"\b(harrison|guyton|robbins|chandrasoma|kdt|bdc|snell|goodman|gilman|harper|uptodate|lippincott|ganong|grays|netter)\b"
    ]
    return not any(re.search(p, rationale) for p in patterns)

def remove_answer_mention(example):
    rationale = example["rationale"]
    if not rationale: return example

    patterns = [
        r"""^ans(?:wer)?\.?\s*(is)?\s*[:=]?\s*[\(\[\']?[a-dA-D][\)\]\']?(?:\.|:)?(?:\s*i\.?e\.?|,)?\s*""",
        r"""^[.,;:\-\s]+"""
    ]
    for pat in patterns:
        rationale = re.sub(pat, "", rationale, flags=re.IGNORECASE | re.VERBOSE).strip()
    example["rationale"] = rationale
    return example

def not_truncated(example):
    formatted = format_mcqa(example)
    plen = len(tokenizer(formatted["prompt"])["input_ids"])
    clen = len(tokenizer(formatted["completion"] + tokenizer.eos_token)["input_ids"])
    return (plen + clen) <= MAX_LEN

def format_mcqa(example):
    choices = "".join(f"{l}. {c}\n" for l, c in zip(CHOICE_LETTERS, example["choices"]))
    idx = ord(example["answer"].strip().upper()) - ord("A")
    return {
        "prompt": (
            "The following are multiple-choice questions (with answers) about advanced STEM knowledge.\n\n"
            f"Question: {example['question'].strip()}\n{choices}\nAnswer:"
        ),
        "completion": f"{CHOICE_LETTERS[idx]}. {example['choices'][idx]}\nExplanation: {example['rationale'].strip()}"
    }

def balance_by_subject(dataset, subject_probs):
    dataset = dataset.filter(lambda ex: ex["subject_name"] in subject_probs)
    weights = np.array([subject_probs[ex["subject_name"]] for ex in dataset])
    weights /= weights.sum()
    indices = np.random.choice(len(dataset), size=len(dataset), replace=True, p=weights)
    return dataset.select(indices)

# Main processing 
medmcqa_raw = load_dataset("openlifescienceai/medmcqa")
filtered = {
    split: medmcqa_raw[split].filter(
        lambda x: (
            x.get("choice_type") == "single" and
            x.get("cop") is not None and
            str(x.get("cop")).strip().isdigit() and
            0 <= int(str(x.get("cop")).strip()) <= 3
        )
    )
    for split in medmcqa_raw
}

formatted = DatasetDict({
    split: filtered[split].map(
        lambda ex, idx: convert_medmcqa(ex, idx),
        with_indices=True,
        remove_columns=filtered[split].column_names
    )
    for split in filtered
})

cleaned = {}
for split, data in formatted.items():
    if split == 'test': 
        continue 

    # Quality filter
    data = data.filter(is_ok, num_proc=8)

    # Deduplication
    seen = set()
    def dedup(ex):
        h = make_hash(ex)
        if h in seen: return False
        seen.add(h)
        return True
    data = data.filter(dedup, num_proc=8)

    # Remove missing/empty rationales
    data = data.filter(has_rationale, num_proc=8)
    print(f"removed rationale : {data}")

    # Remove reference-citing rationales
    data = data.filter(no_reference_rationale, num_proc=8)
    print(f"removed references : {data}")

    # Clean rationale
    data = data.map(remove_answer_mention)
    print(f"cleaned rationale : {data}")

    # Remove short rationales (bottom 10%)
    lengths = [len(ex["rationale"]) for ex in data if ex["rationale"]]
    cutoff = np.percentile(lengths, 10)
    data = data.filter(lambda ex: len(ex["rationale"]) > cutoff)

    cleaned[split] = data

ds = DatasetDict(cleaned)

# Resplit train/val 
new_split = ds["train"].train_test_split(test_size=VALIDATION_SIZE, seed=42)
ds["train"] = new_split["train"]
ds["validation"] = concatenate_datasets([ds["validation"], new_split["test"]])

# Filter out long samples 
ds["train"] = ds["train"].filter(not_truncated, num_proc=8)
ds["validation"] = ds["validation"].filter(not_truncated, num_proc=8)

# Balance train set by subject ─
ds["train"] = balance_by_subject(ds["train"], VALID_SUBJECTS)

print(f"\nFinal sizes:")
print(f" Train: {len(ds['train']):,}")
print(f" Validation: {len(ds['validation']):,}")

from collections import Counter

answer_counts = Counter(ex["answer"] for ex in ds["train"])

total = sum(answer_counts.values())
print("Distribution of correct answers in train set:")
for letter in sorted(answer_counts):
    count = answer_counts[letter]
    print(f"  {letter}: {count} ({100 * count / total:.2f}%)")

ds.push_to_hub(
    f"{HF_USER}/medmcqa_cleaned",
    private=False,
    max_shard_size="500MB"
)