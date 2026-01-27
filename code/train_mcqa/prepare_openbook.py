import os
import json
import time
import random
from pathlib import Path
import json

from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset
from gpt_wrapper.chat import Chat
import gpt_wrapper
import os
from huggingface_hub import login


HF_TOKEN = os.environ["HF_TOKEN"]
HF_USER = os.environ["HF_USER"]

login(token=HF_TOKEN)



# GPT wrapper setup 
GPT_API_BASE = os.environ["GPT_API_BASE"]
GPT_API_KEY  = os.environ["GPT_API_KEY"]

gpt_wrapper.api_base = GPT_API_BASE
gpt_wrapper.api_key  = GPT_API_KEY

SPLITS = ["train", "validation"]
BATCH_SIZE = 25
SAVE_DIR = Path("openbookqa_rationales")
SAVE_DIR.mkdir(exist_ok=True)

FEW_SHOT_PROMPT = """You are a science educator.  For each multiple-choice \
question you receive, write a clear, concise explanation (1–2 sentences) that \
justifies the correct answer.  Do **not** restate the answer option itself.

Example 1
Q: Which material is a good conductor of electricity?
A: Copper’s free electrons allow electrical current to flow easily.

Now answer the next question:
"""


#  Helpers 
def make_question_block(question, choices, answer_letter) :
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
):
    """
    Open a fresh Chat for each question.
    """
    for attempt in range(1, retries + 1):
        try:
            chat = Chat.create("Rationale Generator")
            chat.ask(FEW_SHOT_PROMPT)  
            msg = chat.ask(
                question_block,
                model_args={"temperature": temperature, "max_tokens": max_tokens},
            )
            return msg.content.strip()
        except Exception as exc:
            print(f"Retry {attempt}/{retries} after error: {exc}")
            time.sleep(backoff)
    return None  


def convert_openbook(sample, idx, split_name):

    question_id = f"{split_name}_{idx}"
    question = sample["question_stem"]
    choices = sample["choices"]["text"]

    correct_idx = sample["choices"]["label"].index(sample["answerKey"])
    correct_choice = choices[correct_idx]

    # Ensure exactly 4 choices
    if len(choices) > 4:
        distractors = [c for i, c in enumerate(choices) if i != correct_idx]
        choices = random.sample(distractors, 3) + [correct_choice]
        random.shuffle(choices)

    answer_letter = chr(65 + choices.index(correct_choice))

    return {
        "question_id": question_id,
        "question": question,
        "choices": choices,
        "answer": answer_letter,
    }


# Load and format the data
raw_ds = load_dataset("allenai/openbookqa")


formatted_ds = DatasetDict(
    {
        split: raw_ds[split].map(
            lambda s, i, _split=split: convert_openbook(s, i, _split),
            with_indices=True,
            remove_columns=raw_ds[split].column_names,
        )
        for split in ("train", "validation", "test")
    }
)


# Main loop to process each split and batch
for split in SPLITS:
    data = formatted_ds[split]
    n_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nProcessing '{split}' split – {len(data)} questions")

    for batch_idx in range(n_batches):
        out_path = SAVE_DIR / f"{split}_batch_{batch_idx:03}.json"
        if out_path.exists():
            print(f"Batch {batch_idx} already done; skipping")
            continue

        start, end = batch_idx * BATCH_SIZE, min(
            (batch_idx + 1) * BATCH_SIZE, len(data)
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

print("\nAll done!")


def load_rationale_batches(save_dir: Path, split_name):
    """Load all JSON batch files for a given split and return a merged Dataset."""
    all_records = []

    for batch_file in sorted(save_dir.glob(f"{split_name}_batch_*.json")):
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
            all_records.extend(batch_data)

    print(f"Loaded {len(all_records)} records for split '{split_name}'")
    return Dataset.from_list(all_records)

SAVE_DIR = Path("openbookqa_rationales")

splits = ["train", "validation"]

final_dataset_dict = DatasetDict({
    split: load_rationale_batches(SAVE_DIR, split)
    for split in splits
})

final_dataset_dict = final_dataset_dict.map(lambda x: {**x, "dataset": "openbookqa"})

for split, ds in final_dataset_dict.items():
    print(f"{split}: {len(ds)} samples")

final_dataset_dict.push_to_hub(
    f"{HF_USER}/openbookqa_with_rationales",
    private=True,
    max_shard_size="500MB"
)
