#!/usr/bin/env bash
# Zero-shot open-decoder baselines for the gendered-speech benchmark.
# Runs on virgil. Produces predictions/zeroshot-<tag>.csv for each model;
# evaluate.py (run afterwards in the benchmark venv) then folds them into the
# tables automatically. Does NOT call evaluate.py itself, because vLLM lives in
# its own venv (see setup note below).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$(dirname "$0")"

# --- shared HF cache (respect an already-exported env; else use the observed
#     path where models--* live directly, not under hub/). ----------------------
export HF_HOME="${HF_HOME:?set HF_HOME to a shared model cache}"
# HF_HUB_CACHE defaults to $HF_HOME/hub — do NOT override to $HF_HOME, that points
# at a stale weights-only mirror without config.json on this shared cache.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# Reuse the cache and FAIL FAST instead of silently downloading 10s of GB.
# If a model errors as "not found", copy the EXACT id from `huggingface-cli
# scan-cache` into MODELS below.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "HF_HOME=$HF_HOME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset -- set me to a free GPU, e.g. export CUDA_VISIBLE_DEVICES=0>}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true

# tp MUST equal the number of GPUs in CUDA_VISIBLE_DEVICES. 8-9B fit on one H100.
TP="${TP:-1}"

# model id (must match scan-cache) | short tag
python "$ROOT"/models/run_zeroshot_open.py --model meta-llama/Llama-3.1-8B-Instruct --tag llama31-8b --tp "$TP" --max-model-len 8192
python "$ROOT"/models/run_zeroshot_open.py --model google/gemma-2-9b-it            --tag gemma2-9b  --tp "$TP" --max-model-len 8192

echo
echo "Done. Wrote predictions/zeroshot-llama31-8b.csv and predictions/zeroshot-gemma2-9b.csv"
echo "Now (in the benchmark venv with sklearn/statsmodels) run:  python "$ROOT"/evaluation/evaluate.py"
