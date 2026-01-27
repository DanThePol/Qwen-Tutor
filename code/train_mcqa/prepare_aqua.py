from datasets import load_dataset, DatasetDict, Dataset
import re, json, unicodedata
from typing import List
import gpt_wrapper
from gpt_wrapper.chat import Chat
from pathlib import Path
import time
from collections import defaultdict, Counter
import random
import os
from huggingface_hub import login


HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]

login(token=HF_TOKEN)


mathqa_raw_dataset = load_dataset("allenai/math_qa")
aqua_raw_dataset = load_dataset("deepmind/aqua_rat")


# Canonicalization Helpers

RE_NUM = re.compile(r"\d+([.,]\d+)?")
RE_SPACE = re.compile(r"\s+")

def canonicalize_text(text: str) -> str:
    text = text.lower()
    text = RE_NUM.sub("<num>", text)
    text = RE_SPACE.sub(" ", text).strip(" ?.")
    return text

mathqa_canonical_questions = {
    canonicalize_text(row["Problem"])
    for split in mathqa_raw_dataset.values()
    for row in split
}
print(f"Canonical MathQA questions: {len(mathqa_canonical_questions):,}")


# Remove AQuA Questions Already in MathQA
def is_not_duplicate_aqua_question(example):
    return canonicalize_text(example["question"]) not in mathqa_canonical_questions

aqua_deduplicated_dataset = DatasetDict({
    name: split.filter(is_not_duplicate_aqua_question, num_proc=4)
    for name, split in aqua_raw_dataset.items()
})

for name, split in aqua_deduplicated_dataset.items():
    print(f"{name:<10} → {len(split):6,} rows (was {len(aqua_raw_dataset[name]):6,})")


# Filter Valid AQuA Examples
def has_required_fields(example):
    question = example.get("question", "").strip()
    rationale = example.get("rationale", "").strip()
    correct = example.get("correct", "").strip()
    options = example.get("options", [])

    if isinstance(options, str):
        options = [opt.strip() for opt in options.split(",") if opt.strip()]

    return bool(question and rationale and correct) and isinstance(options, list) and len(options) >= 2

aqua_with_rationale = DatasetDict()
aqua_without_rationale = DatasetDict()

for split in aqua_deduplicated_dataset:
    split_data = aqua_deduplicated_dataset[split]
    aqua_with_rationale[split] = split_data.filter(has_required_fields)
    aqua_without_rationale[split] = split_data.filter(lambda ex: not has_required_fields(ex))

for split in aqua_with_rationale:
    print(f"{split.upper():<10}  Valid: {len(aqua_with_rationale[split])} | Invalid: {len(aqua_without_rationale[split])}")


# Normalize and Reformat Data
RE_LETTER_LABEL = re.compile(r"^(?:\([A-E]\)\*?|[A-E]\))+\s*", re.X)

def strip_letter_prefix(options: List[str]) -> List[str]:
    return [RE_LETTER_LABEL.sub("", opt).strip() for opt in options]

def append_answer_to_rationale(rationale: str, answer: str) -> str:
    if "\n" in rationale:
        rationale = rationale[:rationale.rfind("\n")].strip()
    return f"{rationale}\nThe answer is: {answer}"

def format_example(example, idx):
    return {
        "question_id": f"aqua_rat{idx}",
        "question": example["question"],
        "choices": strip_letter_prefix(example["options"]),
        "answer": example["correct"],
        "rationale": append_answer_to_rationale(example["rationale"], example["correct"]),
    }

aqua_formatted_dataset = DatasetDict()
for split in ["train", "validation", "test"]:
    if split in aqua_with_rationale:
        aqua_formatted_dataset[split] = aqua_with_rationale[split].map(
            lambda ex, idx: format_example(ex, idx),
            with_indices=True,
            remove_columns=aqua_with_rationale[split].column_names
        )



# Normalize Rationales (v1): Strip common answer patterns
def normalize_rationale_v1(text, label):
    label_escaped = re.escape(label)
    text = re.sub(rf"[\-=~>]*\s*the\s+correct\s+answer\s+is\s+{label_escaped}", "", text, flags=re.I)
    text = re.sub(rf"(answer|correct answer)\s*:\s*{label_escaped}", "", text, flags=re.I)
    text = re.sub(rf"the answer is\s*:\s*{label_escaped}", "", text, flags=re.I)
    return text.strip()

def clean_rationales_v1(dataset):
    cleaned, modified_indices = [], []
    for idx, ex in enumerate(dataset):
        original, label = ex["rationale"], ex["answer"]
        cleaned_text = normalize_rationale_v1(original, label)
        if original.strip() != cleaned_text.strip():
            modified_indices.append(idx)
        ex["rationale"] = cleaned_text
        cleaned.append(ex)
    print(f"Cleaned v1: {len(modified_indices)} / {len(dataset)} modified.")
    return cleaned, modified_indices

train_v1_cleaned, train_v1_modified = clean_rationales_v1(aqua_formatted_dataset['train'])
val_v1_cleaned, val_v1_modified = clean_rationales_v1(aqua_formatted_dataset['validation'])
test_v1_cleaned, test_v1_modified = clean_rationales_v1(aqua_formatted_dataset['test'])

# Save rationale comparison for analysis
comparison_data = [
    {
        "question_id": idx,
        "original": aqua_formatted_dataset['train'][idx]["rationale"],
        "cleaned": train_v1_cleaned[idx]["rationale"]
    }
    for idx in train_v1_modified
]
os.makedirs("aqua/gpt_comparison", exist_ok=True)
with open("aqua/gpt_comparison/aqua_train_rationales_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison_data, f, indent=2)
    

GPT_API_BASE = os.environ["GPT_API_BASE"]
GPT_API_KEY  = os.environ["GPT_API_KEY"]

gpt_wrapper.api_base = GPT_API_BASE
gpt_wrapper.api_key  = GPT_API_KEY

input_path = Path("aqua/gpt_comparison/aqua_train_rationales_comparison.json")
with input_path.open("r", encoding="utf-8") as f:
    rationale_data = json.load(f)


RESPONSE_KEY = "A"

DEFAULT_ARGS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}

def build_prompt(item):
    return f"""
You are given two versions of a problem explanation.

Original:
{item['original']}

Cleaned:
{item['cleaned']}

Task:
Determine if the cleaned version preserves all the logical and mathematical reasoning from the original version. 
Ignore formatting changes like whitespace, symbols like '--->', 'Ans', or line breaks.
Focus only on whether any meaningful steps, values, or inferences were removed or altered.

Respond with one of:
- "MATCH: Cleaned version faithfully preserves the original reasoning."
- "MISMATCH: Cleaned version removes or changes important reasoning."

If there’s a mismatch, explain briefly what was lost or changed.
"""

OUTPUT_PATH = Path("aqua/gpt_comparison/aqua_rat_rationale_verification_output.json")
BACKUP_PATH = OUTPUT_PATH.with_suffix(".bak")

if OUTPUT_PATH.exists():
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        output_data = {q["question_id"]: q for q in json.load(f)}
else:
    output_data = {}

for item in rationale_data:
    question_id = item["question_id"]

    if question_id in output_data:
        print(f"Skipping {question_id} (already processed)")
        continue

    print(f"Processing {question_id}...")

    chat = Chat.create(f"{question_id}_{RESPONSE_KEY}_verify")
    prompt = build_prompt(item)
    response = chat.ask(prompt, model_args=DEFAULT_ARGS)

    response_text = response.content.strip()
    chat_id = chat.to_dict().get("chat_id")

    update = {
        RESPONSE_KEY: response_text,
        f"{RESPONSE_KEY}_chat_id": chat_id
    }

    output_data[question_id] = {
        "question_id": question_id,
        "original": item["original"],
        "cleaned": item["cleaned"],
        "verdict": response_text,
        "chat_id": chat_id
    }
    time.sleep(1)
    
print(OUTPUT_PATH)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(list(output_data.values()), f, ensure_ascii=False, indent=2)

print(f"\nAll done. Results saved to {OUTPUT_PATH}")

    

# Remove Rationales Flagged as MISMATCH by GPT
VERDICT_FILE = Path("aqua/gpt_comparison/aqua_rat_rationale_verification_output.json").expanduser()
with open(VERDICT_FILE, "r", encoding="utf-8") as f:
    gpt_verdicts = json.load(f)

mismatch_question_ids = [item["question_id"] for item in gpt_verdicts if item.get("verdict") == "MISMATCH"]
print(f"GPT mismatches: {len(mismatch_question_ids)}")

train_v1_filtered = [
    ex for idx, ex in enumerate(train_v1_cleaned) if idx not in mismatch_question_ids
]

# Normalize Rationales (v2): Remove "Ans D" patterns
def normalize_rationale_v2(text, label):
    return re.sub(rf"Ans\s+{re.escape(label)}\.?\s*(?:\r?\n)?", "", text).strip()

def clean_rationales_v2(dataset):
    cleaned, modified = [], []
    for idx, ex in enumerate(dataset):
        original, label = ex["rationale"], ex["answer"]
        cleaned_text = normalize_rationale_v2(original, label)
        if original.strip() != cleaned_text.strip():
            modified.append(idx)
        ex["rationale"] = cleaned_text
        cleaned.append(ex)
    print(f"Cleaned v2: {len(modified)} / {len(dataset)} modified.")
    return cleaned, modified

train_v2_cleaned, _ = clean_rationales_v2(train_v1_filtered)
val_v2_cleaned, _   = clean_rationales_v2(val_v1_cleaned)
test_v2_cleaned, _  = clean_rationales_v2(test_v1_cleaned)


# Filter Out Records Containing URLs
def find_url_violations(datasets, pattern=r"https?://\S+"):
    url_re = re.compile(pattern)
    bad_indices = defaultdict(list)
    for split, data in datasets.items():
        for idx, item in enumerate(data):
            if any(isinstance(val, str) and url_re.search(val) for val in item.values()):
                bad_indices[split].append(idx)
    return dict(bad_indices)

def remove_indices(data, indices_to_remove):
    return [ex for idx, ex in enumerate(data) if idx not in set(indices_to_remove)]

interim_datasets = {
    "train": train_v2_cleaned,
    "validation": val_v2_cleaned,
    "test": test_v2_cleaned
}

url_violations = find_url_violations(interim_datasets)
print(" URL-violating records:")
for split, idxs in url_violations.items():
    print(f"{split}: {len(idxs)}")

train_no_urls = remove_indices(train_v2_cleaned, url_violations.get("train", []))
val_no_urls   = remove_indices(val_v2_cleaned,   url_violations.get("validation", []))
test_no_urls  = remove_indices(test_v2_cleaned,  url_violations.get("test", []))


clean_dataset = DatasetDict({
    "train": Dataset.from_list(train_no_urls),
    "validation": Dataset.from_list(val_no_urls),
    "test": Dataset.from_list(test_no_urls)
})


# Clean Unicode and Filter Invalid Rationales
ALLOWED_CHARS = {'⁄', '×', '⇒', '→', '…', '‘', '’', '“', '”', '−', '∴', '∵',
                 'π', 'θ', 'α', 'β', 'Δ', '\uf0de', '‹', '›'}

def is_char_allowed(c):
    return (
        ord(c) < 128 or c in ALLOWED_CHARS or c == '€' or
        unicodedata.category(c) in ('Sm', 'Ll') or 'GREEK' in unicodedata.name(c, '')
    )

def strip_rationale_prefix(text):
    return re.sub(r"^(solution|sol\.?|explanation|expl)\s*:?\s*\.?\s*", "", text, flags=re.IGNORECASE)

def normalize_dashes(text):
    return text.replace("–", "-").replace("—", "-")

def clean_rationale_text(text):
    return normalize_dashes(strip_rationale_prefix(text))

def is_clean_text(text):
    return all(is_char_allowed(c) for c in text)

def clean_and_flag(example):
    cleaned_text = clean_rationale_text(example["rationale"])
    return {
        **example,
        "rationale": cleaned_text,
        "was_filtered": not is_clean_text(cleaned_text)
    }

def is_valid_short_rationale(example, min_len=8):
    text = example["rationale"].strip().lower()
    if len(text) < min_len:
        return False
    if any(keyword in text for keyword in ["option", "correct"]):
        return False
    return not re.search(r"\b(ans(wer)?|option|choice|response|correct)\b.*\b[A-E]\b", text)

from datasets import concatenate_datasets

def clean_and_filter_dataset(dataset_dict, char_threshold=27):
    cleaned_dataset = DatasetDict()
    for split in dataset_dict:
        print(f"\nCleaning: {split}")
        raw = dataset_dict[split].map(clean_and_flag)
        retained = raw.filter(lambda ex: not ex["was_filtered"])
        short = retained.filter(lambda ex: len(ex["rationale"]) < char_threshold)
        long = retained.filter(lambda ex: len(ex["rationale"]) >= char_threshold)
        short_valid = short.filter(is_valid_short_rationale)
        final = concatenate_datasets([long, short_valid])
        print(f"✔ Retained: {len(final)} / {len(raw)}")
        cleaned_dataset[split] = final
    return cleaned_dataset

unicode_cleaned_dataset = clean_and_filter_dataset(clean_dataset)


# Keep Only Essential Fields
def retain_fields(dataset_dict, fields):
    trimmed = DatasetDict()
    for split in dataset_dict:
        to_remove = [col for col in dataset_dict[split].column_names if col not in fields]
        trimmed[split] = dataset_dict[split].remove_columns(to_remove)
    return trimmed

final_fields = ["question_id", "question", "choices", "answer", "rationale", "dataset"]
trimmed_dataset = retain_fields(unicode_cleaned_dataset, final_fields)


# Shuffle Choices & Remap Answers
def shuffle_choices_and_remap_answers(dataset_dict, seed=42):
    random.seed(seed)
    updated = DatasetDict()

    def idx_to_letter(i): return chr(ord('A') + i)

    for split in dataset_dict:
        def process_example(example):
            choices = example["choices"]
            correct_idx = ord(example["answer"]) - ord("A")
            if correct_idx >= len(choices):
                return example

            correct_choice = choices[correct_idx]
            distractors = [opt for i, opt in enumerate(choices) if i != correct_idx]
            if len(distractors) < 3:
                return example

            sampled_distractors = random.sample(distractors, 3)
            all_choices = [correct_choice] + sampled_distractors
            random.shuffle(all_choices)

            new_answer = idx_to_letter(all_choices.index(correct_choice))
            return {
                **example,
                "choices": all_choices,
                "answer": new_answer
            }

        updated[split] = dataset_dict[split].map(process_example)

    return updated

shuffled_dataset = shuffle_choices_and_remap_answers(trimmed_dataset)


# Answer Distribution Check
def print_answer_distribution(dataset_dict):
    for split in dataset_dict:
        labels = dataset_dict[split]["answer"]
        counts = Counter(labels)
        total = len(labels)
        print(f"\n{split.upper()} distribution:")
        for letter in ['A', 'B', 'C', 'D']:
            count = counts.get(letter, 0)
            print(f"  {letter}: {count:4d} ({count/total:.1%})")

print_answer_distribution(shuffled_dataset)


# Rebalance: Move Some Train to Validation
def rebalance_validation(dataset_dict, val_fraction=0.08, seed=42):
    train_shuffled = dataset_dict["train"].shuffle(seed=seed)
    n_val = int(len(train_shuffled) * val_fraction)
    val_split = train_shuffled.select(range(n_val))
    train_split = train_shuffled.select(range(n_val, len(train_shuffled)))
    new_validation = concatenate_datasets([dataset_dict["validation"], val_split])
    return train_split, new_validation

train_final, val_final = rebalance_validation(shuffled_dataset)

balanced_dataset = DatasetDict({
    "train": train_final,
    "validation": val_final,
    "test": shuffled_dataset["test"]
})

print_answer_distribution(balanced_dataset)


balanced_dataset.push_to_hub(
    f"{HF_USER}/aqua_cleaned",
    private=False,
    max_shard_size="500MB"
)


