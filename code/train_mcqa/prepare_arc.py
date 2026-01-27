import os
import json
import time
import random
from pathlib import Path
import re
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset
from gpt_wrapper.chat import Chat
import gpt_wrapper
import gpt_wrapper, gpt_wrapper.chat as gw
import os
from huggingface_hub import login


HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]

login(token=HF_TOKEN)


SPLITS = ["train", "validation"]
BATCH = 25
SAVE_DIR = Path("arc_challenge_rationales")
SAVE_DIR.mkdir(exist_ok=True)

# GPT wrapper setup 
GPT_API_BASE = os.environ["GPT_API_BASE"]
GPT_API_KEY  = os.environ["GPT_API_KEY"]

gpt_wrapper.api_base = GPT_API_BASE
gpt_wrapper.api_key  = GPT_API_KEY
chat = gw.Chat.create("ARC-Challenge rationale generator")

FEW_SHOT = """You are a science tutor. For each multiple-choice question you will
give ONE short sentence that *explains why the correct option is correct*.
• Do **not** reveal the correct letter.
• Do **not** mention the other options.
Keep it crisp, factual, ≤ 25 words.

Example:
Q: Why does metal feel colder than wood at the same temperature?
A) Metal is colder.  B) Wood stores more heat.  C) Metal conducts heat faster.  D) Wood conducts heat faster.
Answer: C
Explanation: Metal conducts heat away from your skin efficiently, so it removes body heat faster and feels colder.

Now I’ll give you a new question, choices and the correct letter. Return only the explanation sentence."""


#  Helpers 
def make_question_block(question, choices, answer_letter):
    formatted_choices = "\n".join(
        f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)
    )
    return (
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}\n"
        f"Answer: {answer_letter}\n"
        f"Explanation:"
    )


def generate_rationale(
    question_block: str,
    retries: int = 3,
    backoff: float = 2.0,
    temperature: float = 0.4,
    max_tokens: int = 150,
) :
    """
    Open a fresh Chat for each question.
    """
    for attempt in range(1, retries + 1):
        try:
            chat = Chat.create("Rationale Generator")
            chat.ask(FEW_SHOT)  
            msg = chat.ask(
                question_block,
                model_args={"temperature": temperature, "max_tokens": max_tokens},
            )
            return msg.content.strip()
        except Exception as exc:
            print(f"Retry {attempt}/{retries} after error: {exc}")
            time.sleep(backoff)
    return None  

def convert_data(example, idx, dataname):
    question_id = f"{dataname}_{idx}"
    question = example['question']
    choices = example['choices']['text']

    # Find the correct answer text
    correct_idx = example['choices']['label'].index(example['answerKey'])
    correct_choice = choices[correct_idx]

    # If more than 4 choices, sample 3 distractors + correct
    if len(choices) > 4:
        distractors = [c for i, c in enumerate(choices) if i != correct_idx]
        sampled_distractors = random.sample(distractors, 3)
        final_choices = [correct_choice] + sampled_distractors
        random.shuffle(final_choices)
    else:
        final_choices = choices

    # Find new answer letter 
    correct_letter = chr(65 + final_choices.index(correct_choice))

    return {
        "question_id": question_id,
        "question": question,
        "choices": final_choices,  
        "answer": correct_letter,  
        "dataset": dataname,
    }


# Load and format the data
raw_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")


formatted_ds = DatasetDict(
    {
        split: raw_ds[split].map(
            lambda s, i =split: convert_data(s, i, 'ai2_arc_challenge'),
            with_indices=True,
            remove_columns=raw_ds[split].column_names,
        )
        for split in ("train", "validation", "test")
    }
)



# Main loop to process each split and batch
for split in SPLITS:
    data = formatted_ds[split]
    n_batches = (len(data) + BATCH - 1) // BATCH

    print(f"\nProcessing '{split}' split – {len(data)} questions")

    for batch_idx in range(n_batches):
        out_path = SAVE_DIR / f"{split}_batch_{batch_idx:03}.json"
        if out_path.exists():
            print(f"Batch {batch_idx} already done; skipping")
            continue

        start, end = batch_idx * BATCH, min(
            (batch_idx + 1) * BATCH, len(data)
        )
        batch = data.select(range(start, end))

        processed = []
        for entry in tqdm(batch, desc=f"{split} batch {batch_idx:03}", leave=False):
            question_block = make_question_block(
                entry["question"], entry["choices"], entry["answer"]
            )
            rationale = generate_rationale(question_block)

            if rationale is None:
                print(f"Failed for {entry['question_id']}; keeping empty string")
                rationale = ""

            entry = dict(entry)
            entry["rationale"] = rationale
            processed.append(entry)
            

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(processed)} items → {out_path}")

print("All done!")


def clean_rationale_prefix(text):
    return re.sub(r"^(exp(l(anation)?)?\.?)\s*:\s*", "", text, flags=re.IGNORECASE)

def clean_rationale_entry(example):
    example["rationale"] = clean_rationale_prefix(example["rationale"])
    return example

def load_rationale_batches(save_dir: Path, split_name):
    all_records = []
    for batch_file in sorted(save_dir.glob(f"{split_name}_batch_*.json")):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            all_records.extend(batch_data)
    print(f"Loaded {len(all_records)} records for split '{split_name}'")
    return Dataset.from_list(all_records)

SAVE_DIR = Path("arc_challenge_rationales")
splits = ["train", "validation"]

final_dataset_dict = DatasetDict({
    split: load_rationale_batches(SAVE_DIR, split)
    for split in splits
})

for split in final_dataset_dict:
    final_dataset_dict[split] = final_dataset_dict[split].map(clean_rationale_entry)
    final_dataset_dict[split] = final_dataset_dict[split].map(lambda x: {**x, "dataset": "ai2_arc_challenge"})

for split, ds in final_dataset_dict.items():
    print(f"{split}: {len(ds)} samples")

final_dataset_dict.push_to_hub(
   f"{HF_USER}/ai2_arc_challenge_with_rationales",
    private=False,
    max_shard_size="500MB"
)
