# GENDIS

Code for **GENDIS: Gender-Related Discourse and Stance in Multilingual YouTube
Comments from India, Kenya, and Nigeria** (WOAH 2026).

GENDIS annotates 9,600 YouTube comments (3,200 each from India, Kenya and
Nigeria) with a two-level scheme: Level 1 asks whether a comment references
gender at all, Level 2 classifies the evaluative stance of gender-related
comments as negative, positive or neutral. This repository holds the evaluation
harness and the model predictions behind the paper's tables.

## Data

The dataset is **not** in this repository. It is released separately, under
gated access:

**https://huggingface.co/datasets/worldbank/gendis**

Request access there, then authenticate (`huggingface-cli login`, or set
`HF_TOKEN`) and build the pipeline's input file:

```bash
pip install -r requirements.txt
python prepare_data.py   # writes data/processed.csv
```

`data/processed.csv` is the single input every other script reads. The fold
assignments ship with the release, so the cross-validation splits reproduce
exactly rather than being redrawn.

## Layout

```
prepare_data.py      step 1: download the release and build data/processed.csv
run_all.sh           run the whole pipeline end to end
gendis.py            shared helpers: paths, label constants, prediction schema
make_folds.py        regenerate the fold split (needs the raw annotation files;
                     the released folds reproduce as-is, so this is rarely needed)

models/              scripts that PRODUCE predictions
  run_floor.py          keyword-match floor
  run_toxicity.py       off-the-shelf multilingual toxicity baseline
  run_finetune.py       XLM-R fine-tuning
  run_setfit.py         SetFit (contrastive few-shot)
  run_zeroshot.py       closed models via the OpenAI Batch API
  run_zeroshot_open.py  open models via vLLM
  run_zeroshot_hf.py    transformers fallback for models vLLM cannot serve
  ingest_zeroshot.py    parse batch output into the prediction schema
  prompt.py             the zero-shot prompt, fixed in advance
  run_open.sh           GPU runner for the open zero-shot models
  run_within.sh         GPU runner for the within-country supervised models

evaluation/          scripts that TURN predictions INTO the paper's numbers
  evaluate.py             main metrics, per country, plus paired McNemar
  eval_random_stratum.py  metrics on the uniform random stratum only
  eval_by_language.py     metrics by language group
  iaa.py                  inter-annotator agreement
  make_figures.py         label-distribution and confusion figures

data/                processed.csv lands here (gitignored)
outputs/
  predictions/       per-system predictions (idx + labels, no comment text)
  results/           computed metrics
  figures/           generated figures
```

## Reproducing the results

Every system is scored on the same instances, using out-of-fold predictions for
the trained models, so the numbers are directly comparable.

```bash
python prepare_data.py
python evaluation/evaluate.py               # main results + per-country breakdown
python evaluation/eval_random_stratum.py    # keyword-independent subset
python evaluation/eval_by_language.py       # by language group
python evaluation/iaa.py                    # inter-annotator agreement
```

All five read the released data plus the predictions shipped in `outputs/`, so the
paper's tables reproduce **without a GPU**. To regenerate the predictions:

```bash
python models/run_floor.py
python models/run_toxicity.py --model unitary/multilingual-toxic-xlm-roberta
python models/run_finetune.py --setting cv      # also: within, loco
python models/run_setfit.py   --setting cv      # also: within, loco
```

`run_all.sh` runs the full pipeline in order. The supervised runs need
a GPU; set `HF_HOME` to a shared model cache first. Zero-shot runs against the
OpenAI API expect `OPENAI_KEY_FILE` to point at a file containing the key.

### Training regimes

- `within` — train and test inside each country's own folds
- `cv` — pooled cross-validation across all three countries
- `loco` — leave-one-country-out, to measure cross-cultural transfer

## Notes

- Run SetFit jobs sequentially. It sets no `output_dir`, so two concurrent runs
  in the same directory overwrite each other's checkpoints.
- Every prediction file must have 9,600 rows. A shorter file silently
  mis-joins on `idx` rather than raising.
- Some comments contain bare carriage returns. `prepare_data.py` writes
  `processed.csv` with full quoting for this reason, and verifies the round
  trip; writing it unquoted yields 9,603 rows with shifted columns.

## Citation

```bibtex
@inproceedings{koch-etal-2026-gendis,
  title     = {{GENDIS}: Gender-Related Discourse and Stance in Multilingual
               {Y}ou{T}ube Comments from {I}ndia, {K}enya, and {N}igeria},
  author    = {Koch, Luise and Tonneau, Manuel and Malhotra, Niyati and
               Rai, Sunny and Munoz Boudet, Ana Maria and Orozco-Olvera, Victor and
               Fraiberger, Samuel P. and Guntuku, Sharath Chandra},
  booktitle = {Proceedings of the 10th Workshop on Online Abuse and Harms (WOAH)},
  year      = {2026}
}
```
