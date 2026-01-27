import faiss, torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

# ───────────────────────────────────────────────
# STEP 1: Embed paragraphs
# ───────────────────────────────────────────────

OUTFILE = "aquarat.csv"
MAX_TK_LEN = 512
n_clusters = 20000

# Loading corpus
print("Loading DS")
# Taking the cleaned and formatted version from MNLP teammate Martina
aquarat = load_dataset("mgatti/aqua_rat_cleaned_and_formatted_v2", split="train")
# question, rationale, question_id, choices, answer, length_diff, was_cleaned, dataset

# Use a GPU-enabled embedding model
print("Loading embedder and tokeniser")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use own tokeniser to check sequence lengths
tokeniser = AutoTokenizer.from_pretrained("danthepol/MNLP_M3_rag_model")
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

print(f"Extracting questions with associated contexts with total MAX_TK_LEN {MAX_TK_LEN} from aquarat")
# Extract info from aquarat
qst = "question"
ans = "answer"
qa = "question-answer"
paragraphs = {qst : [], ans : [], qa : []}
for x in tqdm(aquarat):
    clean = x["was_cleaned"]
    if not clean:
        continue
    q = x["question"].replace("\"", "'").replace("\n", " ")
    a = x["rationale"].replace("\"", "'").replace("\n", " ")
    fmt = f"Example of related question Q and answer A: Q: {q} A: {a}"
    
    token_length = len(tokeniser.tokenize(fmt))
    if token_length <= MAX_TK_LEN:
        paragraphs[qst].append(q)
        paragraphs[ans].append(a)
        paragraphs[qa].append(fmt)

n_examples = len(paragraphs[qst])
print(f"Found {n_examples} contexts")

# Compute embeddings in batches to avoid OOM
batch_size = 256
all_embeddings = []

print("Encoding batches of paragraphs")
for i in tqdm(range(0, n_examples, batch_size)):
    batch = paragraphs[qa][i:i+batch_size]
    emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(emb)

embeddings = np.vstack(all_embeddings).astype('float32')  # shape: (2280000, 768)
print(f"Embeddings shape: {embeddings.shape}")

# ───────────────────────────────────────────────
# STEP 2: FAISS KMeans (GPU) to get 20k clusters
# ───────────────────────────────────────────────

d = embeddings.shape[1]

print(f"Loading FAISS KMeans on {device} with {n_clusters} clusters and d = {d}")

res = faiss.StandardGpuResources()
kmeans = faiss.Kmeans(d, n_clusters, niter=25, verbose=True, gpu=True)
kmeans.train(embeddings)

# ───────────────────────────────────────────────
# STEP 3: Find nearest paragraph to each centroid
# ───────────────────────────────────────────────

index_flat = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index.add(embeddings)

# Find closest point to each centroid
_, closest_indices = gpu_index.search(kmeans.centroids, 1)
closest_indices = closest_indices.flatten()

# ───────────────────────────────────────────────
# STEP 4: Write result
# ───────────────────────────────────────────────

with open(OUTFILE, "w") as f:
    f.write("question,answer\n")
    for i in closest_indices:
        idx = int(i)
        q = paragraphs[qst][idx]
        a = paragraphs[ans][idx]
        f.write(f"\"{q}\",\"{a}\"\n")