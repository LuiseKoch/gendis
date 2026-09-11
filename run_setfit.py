"""GPU (light). SetFit: contrastive fine-tuning of a multilingual sentence
encoder + classifier head. Strong in the low-data regime, so it is the
recommended variant for Level 2 where the positive class is small (~50/country).

Same settings and output schema as run_finetune.py.

Usage:
  python run_setfit.py --setting cv
  python run_setfit.py --setting loco
"""
import argparse
import numpy as np
import pandas as pd
from setfit import SetFitModel, Trainer as SFTrainer, TrainingArguments as SFArgs
from datasets import Dataset

from common import write_predictions


def make_args():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=["cv", "loco", "within"], default="cv")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def train_predict(model_name, train_df, eval_df, label_col, seed):
    model = SetFitModel.from_pretrained(model_name)
    ds = Dataset.from_pandas(
        train_df[["text", label_col]].rename(columns={label_col: "label"}),
        preserve_index=False)
    targs = SFArgs(num_epochs=1, batch_size=16, seed=seed, num_iterations=20)
    SFTrainer(model=model, args=targs, train_dataset=ds).train()
    return np.asarray(model.predict(eval_df["text"].tolist()))


def run_layer(df, args, task):
    label = "gold_l1" if task == "l1" else "gold_l2"
    pred = pd.Series(np.nan, index=df["idx"].values)

    def fit_eval(tr, te):
        if task == "l2":
            tr = tr[tr["gold_l2"].notna()]
        tr = tr.assign(**{label: tr[label].astype(int)})
        return train_predict(args.model, tr, te, label, args.seed)

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
    sysname = f"setfit-{args.setting}"
    out = args.out or f"predictions/setfit_{args.setting}.csv"
    print(f"[{sysname}] L1 ...");  p1 = run_layer(df, args, "l1")
    print(f"[{sysname}] L2 ...");  p2 = run_layer(df, args, "l2")
    write_predictions(df["idx"], df["country"], sysname, p1, p2, out)
