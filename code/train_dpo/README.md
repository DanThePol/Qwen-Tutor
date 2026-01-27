# DPO Training for MNLP M3

This folder contains everything needed to train a Direct Preference Optimization (DPO) model using the MNLP M3 dataset.

## 📁 Contents

- **`fdpo_training.py`** - Main training script with fDPO implementation
- **`requirements.txt`** - Python dependencies
- **`dataset.ipynb`** - Jupyter notebook for creating and uploading the training dataset
- **`README.md`** - This file

## 🚀 Quick Start

### 1. Set Your Hugging Face Repository (Optional)

Edit the bash script in the root directory (`../train_dpo.sh`) and set your HF repo:

```bash
# EDIT THE LINE BELOW TO SET YOUR HUGGING FACE REPO:
# Leave empty ("") to save locally only, or set to "username/model-name" to upload
HF_REPO="your-username/your-model-name"  # Example: "username/my-dpo-model"
```

**Important:** 
- Use straight quotes (`""`) not smart quotes (`""`) to avoid bash syntax errors
- If you want to upload to Hugging Face, make sure you're logged in first: `huggingface-cli login`

### 2. Run Training

From the project root directory:

```bash
chmod +x train_dpo.sh
./train_dpo.sh
```

That's it! The script will:
- Install dependencies from `train_dpo/requirements.txt`
- Train the model with fixed parameters
- Save locally to `./dpo_model`
- Upload to Hugging Face if you specified a repo

## 📊 Dataset Creation

The `dataset.ipynb` notebook creates the MNLP M3 dataset by:

1. **Loading source datasets:**
   - MetaMath DPO FewShot (200K samples → filtered)
   - EvolCodeAlpaca v1 DPO (80K samples → filtered)

2. **Processing and filtering:**
   - Extracts clean questions from MetaMath prompts
   - Applies token length filtering (max 800 tokens total, 400 prompt)
   - Creates train/validation/test splits (80/10/10%)

3. **Upload options:**
   - Set `hf_repo` in the notebook config to upload to HF
   - Leave empty to save locally only

**Note:** You do NOT need to run `dataset.ipynb` for training to work. The dataset `albertfares/MNLP_M3_dpo_dataset` already exists and will be automatically downloaded during training. This notebook is only provided for reference or if you want to create your own dataset variant.

## ⚙️ Training Configuration

The training uses these **fixed parameters** (do **NOT** modify):

```python
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
DATASET = "albertfares/MNLP_M3_dpo_dataset"
OUTPUT_DIR = "./dpo_model"

# Training hyperparameters
epochs = 1
learning_rate = 1e-5
beta = 0.3 (DPO parameter)
batch_size = 16
gradient_accumulation = 8
max_length = 800
filter_threshold = 0.1 (fDPO filtering)
```

## 🔧 Technical Details

### fDPO (Filtered DPO)
The training implements filtered DPO which:
- Applies dynamic filtering during training
- Uses a loss scaling factor based on `filter_threshold`
- Helps improve training stability on noisy preference data

### Model Architecture
- **Base Model:** Qwen3-0.6B-Base (600M parameters)
- **Training:** Uses both policy and reference models
- **Optimization:** AdamW with cosine learning rate schedule
- **Memory:** Gradient checkpointing enabled for efficiency

### Hardware Requirements
- **GPU:** Recommended (automatically detected)
  - Uses bfloat16 precision on GPU
  - ~4-6GB VRAM needed for 0.6B model
- **CPU:** Fallback supported
  - Uses float32 precision
  - Slower but works without GPU

## 📈 Logging and Monitoring

### Weights & Biases (Optional)
- Automatically enabled if `WANDB_API_KEY` is set
- Project name: `MNLP_M3_fDPO`
- Logs training metrics, loss curves, and hyperparameters
- Gracefully skips if not configured

### Training Output
The script provides detailed logging:
- Configuration summary
- Dataset statistics
- Training progress
- Save/upload status
- Sample data preview

## 📦 Dependencies

All dependencies are in `requirements.txt`:

```
torch>=2.1.0
transformers>=4.46.0
trl>=0.11.4
accelerate>=1.1.1
datasets>=2.14.0
wandb>=0.16.0
numpy>=1.24.0
seaborn>=0.11.0
matplotlib>=3.5.0
huggingface_hub>=0.19.0
```

## 🔄 Expected Workflow

1. **Prepare dataset** (optional - dataset already exists):
   ```bash
   # Edit dataset.ipynb to set your HF repo or leave empty for local
   jupyter notebook dataset.ipynb
   ```

2. **Configure upload** (optional):
   ```bash
   # Edit ../train_dpo.sh to set HF_REPO
   HF_REPO="username/my-model"
   ```

3. **Train model:**
   ```bash
   ./train_dpo.sh
   ```

4. **Use trained model:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   
   model = AutoModelForCausalLM.from_pretrained("./dpo_model")
   tokenizer = AutoTokenizer.from_pretrained("./dpo_model")
   ```

## 📊 Dataset Statistics

The MNLP M3 dataset contains:
- **Total samples:** ~25K after filtering
- **Math problems:** ~20K from MetaMath (reasoning tasks)
- **Code problems:** ~5K from EvolCodeAlpaca (programming tasks)
- **Splits:** 80% train, 10% validation, 10% test
- **Format:** Standard DPO format with prompt/chosen/rejected

## 🛠️ Troubleshooting

### Common Issues

**1. Dataset Corruption Error (Parquet/Arrow Files)**

If you encounter errors like:
- `pyarrow.lib.ArrowInvalid: Parquet file size is 0 bytes`
- `OSError: Couldn't read parquet files`
- Dataset loading fails with corruption errors

**Solution:**
1. **Create a new dataset** using `dataset.ipynb`:
   ```python
   # In dataset.ipynb, change the hf_repo to a new name:
   CONFIG = {
       # ...
       "hf_repo": "your-username/M3_dataset_v2",  # Use a NEW repo name
       # ...
   }
   ```

2. **Run the notebook** to create and upload the fresh dataset

3. **Update your training script** to use the new dataset:
   ```bash
   # In train_dpo.sh, change:
   DATASET="your-username/M3_dataset_v2"  # Use your new dataset
   ```

4. **Run training** with the new dataset:
   ```bash
   ./train_dpo.sh
   ```

This creates a fresh, uncorrupted dataset that should work properly for training.

**2. CUDA Out of Memory**
```bash
# Reduce batch size in fdpo_training.py
batch_size = 8  # or 4
gradient_accumulation = 16  # increase to maintain effective batch size
```

**3. Dataset Loading Fails**
```bash
# Check internet connection and HF access
huggingface-cli login
```

**4. Upload Fails**
```bash
# Make sure you're logged in to HF first
huggingface-cli login
# And have write access to the repo
```

**5. Bash Syntax Errors / Line Ending Issues**

If you get errors like:
- `bash: $'\r': command not found`
- `syntax error: unexpected end of file`
- Script fails to run on Linux/Mac

**Solution - Fix line endings:**
```bash
# Convert Windows line endings to Unix (run this in project root)
sed -i 's/\r$//' train_dpo.sh

# Then make executable and run
chmod +x train_dpo.sh
./train_dpo.sh
```


**6. Weights & Biases Errors**
```bash
# Login to wandb or skip logging
wandb login
# Or unset WANDB_API_KEY to disable
unset WANDB_API_KEY
```

### Performance Tips

- **GPU Training:** Always faster - script auto-detects
- **Batch Size:** Adjust based on your GPU memory
- **Mixed Precision:** Automatically enabled (bfloat16 on GPU)
- **Gradient Checkpointing:** Enabled to save memory

## 📝 Output Files

After training, `./dpo_model/` contains:
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer files
- `tokenizer_config.json` - Tokenizer configuration

## 🔗 Related Files

- **`../train_dpo.sh`** - Main training script (edit HF_REPO here)
- **`../requirements.txt`** - May exist at root level (not used)
- **Other training folders** - `../train_mcqa/`, `../train_quantized/`, etc.

## 📚 References

- [TRL DPO Documentation](https://huggingface.co/docs/trl/dpo_trainer)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Qwen Model Documentation](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

---

**Ready to train your DPO model? Run `./train_dpo.sh` from the project root!** 🚀
