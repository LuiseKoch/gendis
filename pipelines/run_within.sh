#!/usr/bin/env bash
# Within-country supervised baselines: train AND test within each country's
# own folds (per-country cross-validation). This is the in-domain,
# single-country midpoint between pooled CV (trains across all three countries)
# and LOCO (trains on two, tests on the held-out third). Uses the SAME fixed
# fold split and the SAME processed.csv as every other system, so the new
# predictions drop straight into evaluate.py.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:?set HF_HOME to a shared model cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "HF_HOME=$HF_HOME  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

# Writes predictions/setfit_within.csv (system "setfit-within") and
# predictions/ft_xlm-roberta-base_within.csv (system "xlmr-within").
python run_setfit.py   --setting within
python run_finetune.py --setting within --model FacebookAI/xlm-roberta-base

# Re-score everything; per-country tables now include the within-country rows.
python evaluate.py
echo "Done. Pull results/ + predictions/ back and the per-country comparison is ready."
