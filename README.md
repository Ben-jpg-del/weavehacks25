# Weavehacks (minimal)
Three Python maps, keystroke demos in Parquet, and a tiny imitation-learning (Behavior Cloning) trainer.

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Train BC
python scripts/train_bc.py \
  --train data/processed/level_001.parquet data/processed/level_002.parquet data/processed/level_003.parquet \
  --val data/processed/val_small.parquet

## Quick eval rollout (stub env)
python scripts/eval_rollout.py --policy artifacts/bc.pt --level level_001
