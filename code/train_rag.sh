set -e

echo Installing pip requirements
pip install -r train_rag/requirements.txt
echo Running training script
python train_rag/train.py