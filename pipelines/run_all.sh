#!/usr/bin/env bash
# Run the full gendered-speech benchmark on virgil (OII H100 box).
# virgil is NOT Slurm: this is a plain launcher. Run it inside tmux.
#
#   ssh virgil
#   tmux new -s woah
#   cd /path/to/gendis/benchmark
#   export OPENAI_KEY_FILE=/path/to/openai.key   # only if re-running zero-shot here
#   bash run_all_virgil.sh
set -euo pipefail

# --- HF cache is SHARED on virgil: reuse it, do NOT create a personal copy. ---
# Respect an HF_HOME already exported in your environment (the shared cache).
# If yours is empty, run `echo $HF_HOME` to find the shared path and either
# export it before this script or replace the fallback below.
if [ -z "${HF_HOME:-}" ]; then
  echo "WARN: HF_HOME was unset; falling back to the default cache" >&2
  export HF_HOME=$HOME/.cache/huggingface
fi
echo "HF_HOME=$HF_HOME (shared)"

# --- pick FREE GPU(s): check nvidia-smi first, then set this. one H100 is plenty here ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- thread caps (shared box, be polite) ---
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# --- preflight: storage (root disk is chronically >90% full) ---
echo "=== storage ==="
df -h "${HF_HOME:-$HOME}" 2>/dev/null || true
root_use=$(df --output=pcent / | tail -1 | tr -dc '0-9')


# --- preflight: GPU usage (SHARED box — make sure your GPU is actually free) ---
echo "=== GPU usage (pick an idle one for CUDA_VISIBLE_DEVICES) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv || true

# --- fail fast if torch can't actually see a GPU (else everything silently runs on CPU) ---
python3 - <<'PY' || { echo "ERROR: torch cannot see a CUDA GPU. Ensure the venv has a CUDA build of torch and a free GPU is visible (nvidia-smi / CUDA_VISIBLE_DEVICES)." >&2; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} | CUDA available: {ok} | devices: {torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
PY

# 1. shared partition + human ceiling
python make_folds.py
python iaa.py

# 2. training-free systems
python run_floor.py
python run_toxicity.py --model unitary/multilingual-toxic-xlm-roberta

# 3. zero-shot: prefer the bundled all-country file (built locally via Batch API).
if [ -f predictions/zeroshot.csv ]; then
  echo "using bundled predictions/zeroshot.csv (all countries)"
elif [ -f ../gpt5nano_eval/predictions_merged.csv ]; then
  python ingest_zeroshot.py ../gpt5nano_eval/predictions_merged.csv   # Kenya-only fallback
fi

# 4. supervised (GPU). loco (leave-one-country-out) needs >=2 countries.
N_COUNTRIES=$(python3 -c "import pandas as pd; print(pd.read_csv('processed.csv')['country'].nunique())")
echo "countries in data: $N_COUNTRIES"

python run_finetune.py --setting cv
python run_setfit.py   --setting cv
if [ "$N_COUNTRIES" -ge 2 ]; then
  python run_finetune.py --setting loco
  python run_setfit.py   --setting loco
else
  echo "skipping loco (need >=2 countries; have $N_COUNTRIES)"
fi

# 5. score everything on identical out-of-fold instances
python evaluate.py

echo "DONE -> results/  predictions/"
