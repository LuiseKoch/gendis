"""Baseline: trivial floor. Establishes how much signal requires no learning.

L1: predict gendered iff the comment matched a sampling keyword
    (keyword_present) -- the heuristic the sample was built on.
L2: predict the per-fold training-majority class (no leakage).
"""
import numpy as np
import pandas as pd

df = pd.read_csv("processed.csv")

# L1 floor: keyword presence
kp = df["keyword_present"].astype(str).str.upper().eq("TRUE").astype(int)

# L2 floor: per-fold majority among gold-gendered rows (out-of-fold)
pred_l2 = np.full(len(df), np.nan)
for f in sorted(df["fold"].unique()):
    train = df[(df["fold"] != f) & df["gold_l2"].notna()]
    maj = int(train["gold_l2"].value_counts().idxmax())
    pred_l2[df["fold"].values == f] = maj

from common import write_predictions
write_predictions(df["idx"], df["country"], "floor", kp.values, pred_l2,
                  "predictions/floor.csv")
