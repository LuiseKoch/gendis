"""GPU. Supervised fine-tuning of a multilingual transformer (XLM-R by default).

Runs BOTH layers (L1 and L2) for one evaluation setting and writes a single
combined prediction file in the standard schema, so it drops straight into
evaluate.py alongside the zero-shot and baseline systems.

Settings (the comparable-test-set design):
  cv    : stratified k-fold over the shared `fold` column. Each comment gets an
          out-of-fold prediction from a model that never trained on it.
  loco  : leave-one-country-out. Train on all-but-one country, predict the
          held-out country. Tests cross-cultural transfer.

L1 trains on all rows. L2 trains on gold-gendered rows (gold_l2 in {0,1,2})
with class weighting for the rare positive class, then predicts L2 for ALL
held-out rows so the end-to-end pipeline has a value everywhere.

Usage:
  python run_finetune.py --setting cv   --model FacebookAI/xlm-roberta-base
  python run_finetune.py --setting loco --model FacebookAI/xlm-roberta-base

Tip: --model worldbank/naija-xlm-twitter-base-hate gives a domain-adapted
start (Twitter/abuse, multilingual) and is worth reporting as a variant.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

from common import write_predictions

MAXLEN = 128


def make_args():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["cv", "loco", "within"], default="cv")
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


class WeightedTrainer(Trainer):
    """CrossEntropy with class weights (for the imbalanced L2 task)."""
    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = (self.class_weights.to(out.logits.device)
             if self.class_weights is not None else None)
        loss = nn.functional.cross_entropy(out.logits, labels, weight=w)
        return (loss, out) if return_outputs else loss


def train_predict(model_name, train_df, eval_df, text_col, label_col,
                  n_labels, args, weighted=False):
    tok = AutoTokenizer.from_pretrained(model_name)

    def tokrows(d, with_labels):
        # eval rows can have NaN labels (e.g. not-gendered rows during L2
        # inference); only the train set carries labels. Drop the raw text
        # column too so the data collator never sees a string field.
        sel = [text_col, label_col] if with_labels else [text_col]
        ren = {text_col: "text", label_col: "labels"} if with_labels else {text_col: "text"}
        ds = Dataset.from_pandas(d[sel].rename(columns=ren), preserve_index=False)
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAXLEN),
                    batched=True)
        return ds.remove_columns(["text"])

    train_ds = tokrows(train_df, with_labels=True)
    eval_ds = tokrows(eval_df, with_labels=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels)

    cw = None
    if weighted:
        counts = train_df[label_col].value_counts().reindex(range(n_labels)).fillna(0)
        inv = (counts.sum() / (n_labels * counts.replace(0, np.nan)))
        cw = torch.tensor(inv.fillna(0).values, dtype=torch.float)

    targs = TrainingArguments(
        output_dir="/tmp/ft", per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64, learning_rate=args.lr,
        num_train_epochs=args.epochs, weight_decay=0.01, seed=args.seed,
        fp16=torch.cuda.is_available(), report_to=[], logging_steps=50,
        save_strategy="no", eval_strategy="no")

    trainer = WeightedTrainer(
        class_weights=cw, model=model, args=targs, train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    logits = trainer.predict(eval_ds).predictions
    return logits.argmax(-1)


def run_layer(df, args, task):
    """Return a pred array aligned to df.idx (NaN where not predicted)."""
    label = "gold_l1" if task == "l1" else "gold_l2"
    n_labels = 2 if task == "l1" else 3
    weighted = (task == "l2")
    pred = pd.Series(np.nan, index=df["idx"].values)

    def fit_eval(tr, te):
        if task == "l2":
            tr = tr[tr["gold_l2"].notna()]
        tr = tr.assign(**{label: tr[label].astype(int)})
        return train_predict(args.model, tr, te, "text", label,
                             n_labels, args, weighted)

    if args.setting == "cv":
        for f in sorted(df["fold"].unique()):
            te = df[df["fold"] == f]
            pred.loc[te["idx"].values] = fit_eval(df[df["fold"] != f], te)
    elif args.setting == "loco":
        for c in sorted(df["country"].unique()):
            te = df[df["country"] == c]
            pred.loc[te["idx"].values] = fit_eval(df[df["country"] != c], te)
    else:  # within: per-country cross-validation (train and test within each country)
        for c in sorted(df["country"].unique()):
            dc = df[df["country"] == c]
            for f in sorted(dc["fold"].unique()):
                te = dc[dc["fold"] == f]
                pred.loc[te["idx"].values] = fit_eval(dc[dc["fold"] != f], te)
    return pred.values


if __name__ == "__main__":
    args = make_args()
    df = pd.read_csv("processed.csv")
    tag = args.model.split("/")[-1]
    sysname = f"xlmr-{args.setting}" if "xlm" in tag.lower() else f"{tag}-{args.setting}"
    out = args.out or f"predictions/ft_{tag}_{args.setting}.csv"

    print(f"[{sysname}] L1 ...");  p1 = run_layer(df, args, "l1")
    print(f"[{sysname}] L2 ...");  p2 = run_layer(df, args, "l2")
    write_predictions(df["idx"], df["country"], sysname, p1, p2, out)
