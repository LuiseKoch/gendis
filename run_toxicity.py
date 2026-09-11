"""Baseline: an off-the-shelf abuse/hate classifier. Its purpose is to test the
paper's core claim that generic abuse detection != gendered-speech detection.

A hate/toxicity score is a proxy, not a gendered-speech label, so we map it
deliberately and report it honestly:
  L1 (gendered)        : abuse>=thr  -> 1   (poor proxy by design)
  L2 (neg/neu/pos)     : abuse>=thr  -> negative(1) ; else neutral(0)
The expected story: it captures the NEGATIVE pole but is blind to neutral and
positive gendered speech, and mis-fires on L1.

Default model is multilingual; swap via --model. Inference only (CPU ok).

Usage:
  python run_toxicity.py --model unitary/multilingual-toxic-xlm-roberta
"""
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common import write_predictions


def make_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unitary/multilingual-toxic-xlm-roberta")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--bs", type=int, default=32)
    return p.parse_args()


def score(texts, model_name, bs):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], truncation=True, max_length=128,
                    padding=True, return_tensors="pt").to(dev)
            logits = model(**b).logits
            # multilabel toxicity heads -> sigmoid, take max over toxic dims;
            # single-label hate heads -> softmax prob of the positive class.
            if logits.shape[-1] == 1 or model.config.problem_type == "multi_label_classification":
                p = torch.sigmoid(logits).max(-1).values
            else:
                p = torch.softmax(logits, -1)[:, -1]
            out.extend(p.cpu().numpy().tolist())
    return np.array(out)


if __name__ == "__main__":
    args = make_args()
    df = pd.read_csv("processed.csv")
    s = score(df["text"].tolist(), args.model, args.bs)
    pred_l1 = (s >= args.thr).astype(int)
    pred_l2 = np.where(s >= args.thr, 1, 0)  # negative if abusive else neutral
    tag = args.model.split("/")[-1]
    write_predictions(df["idx"], df["country"], f"toxicity-{tag}",
                      pred_l1, pred_l2, f"predictions/toxicity_{tag}.csv")
