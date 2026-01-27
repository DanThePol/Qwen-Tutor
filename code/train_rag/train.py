from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from random import Random
import torch, os

# -------------------------------------------------------------------
# Patch get_default_device for very old torch builds
# -------------------------------------------------------------------
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda *_: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
HF_REPO  = "danthepol/MNLP_M3_document_encoder" 
OUTPUT_PATH = "./MNLP_M3_document_encoder/"
TRAINING_DS = "danthepol/m3-rag-training"
BATCH_SIZE   = 32
EPOCHS       = 3
MAX_EXAMPLES = None # set None to use full dataset
SEED = 42

# -------------------------------------------------------------------
# Load SciQ dataset
# -------------------------------------------------------------------
print("📦  Loading dataset …")
ds = load_dataset(TRAINING_DS, split="train")
if MAX_EXAMPLES:
    ds = ds.select(range(min(len(ds), MAX_EXAMPLES)))

# -------------------------------------------------------------------
# Build (question, passage) pairs
# -------------------------------------------------------------------
print("🔧  Building InputExamples …")
examples = [
    InputExample(texts=[row["question"], row["context"]])
    for row in ds
    if row.get("context", "").strip()
]

# Shuffle data in reproducible manner with seed
rng = Random(SEED)
rng.shuffle(examples)

# -------------------------------------------------------------------
# Model, dataloader, loss
# -------------------------------------------------------------------
print("🚀  Loading embedding model …")
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer(EMBED_MODEL, device=device)

dataloader = DataLoader(
    examples,
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=model.smart_batching_collate,
)
loss_fn = losses.MultipleNegativesRankingLoss(model)

# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------
print("🎓  Training bi-encoder …")
model.fit(
    train_objectives=[(dataloader, loss_fn)],
    epochs=EPOCHS,
    warmup_steps=100,
    show_progress_bar=True,
    output_path=OUTPUT_PATH,        # checkpoints & final save
)

print(f"Saved model and checkpoints into {OUTPUT_PATH}")