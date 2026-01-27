# Running `train_rag.sh`
Please follow these steps to run the training script:
1. Login to Hugging Face using `huggingface-cli login`
2. Create and activate a fresh new virtual Python environment
3. Finally, to run the script, type `sh train_rag.sh` in the previous directory

# Dataset
The code for the creation of the embedder's training dataset and corpus is mostly contained within the `cleaned.ipynb` notebook. The two Python scripts, `classify_aquarat.py` and `classify_pubmed.py` select the 20,000 "most relevant" (question, answer) pairs from *AQUA-RAT* (the subset cleaned by Martina) and *PubMedQA* datasets respectively by using KMeans (with the base pre-trained embedder `BAAI/bge-base-en-v1.5` from Hugging Face), and write them to `aquarat.csv` and `pubmedqa.csv`. The KMeans Python scripts have already been run and the `.csv` files populated, but should you want to run the script to see for yourself you can. 

### Note:
`requirements.txt` assumes that the system you will run the RAG model on can install a faiss-gpu-cu12 distribution. If this isn't the case, change that line to faiss-cpu.