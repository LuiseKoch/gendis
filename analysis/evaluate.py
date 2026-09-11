"""Unified evaluation. Loads every prediction file in predictions/, scores
each system on the IDENTICAL comparable partition (out-of-fold for trained
systems; the same instances for training-free systems), and reports:

  - Level 1     : gendered vs not          (all rows)
  - Level 2 cond: neg/neu/pos              (rows with a gold L2 majority;
                  conditioned on GOLD gendered, so L1 errors don't leak in)
  - End-to-end  : 4-way notgendered/neu/neg/pos
  - Paired McNemar test vs the zero-shot system on Level 1

Each metric is reported overall and per country. Macro-F1 is the headline
given class imbalance (positive is rare).
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from gendis.paths import PREDICTIONS, PROCESSED, RESULTS

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from statsmodels.stats.contingency_tables import mcnemar

from gendis.common import L2_NAMES, FOUR_WAY

proc = pd.read_csv(str(PROCESSED))
gold = proc.set_index("idx")

preds = {}
for f in sorted(sorted(str(p) for p in PREDICTIONS.glob("*.csv"))):
    p = pd.read_csv(f)
    for sysname, g in p.groupby("system"):
        preds[sysname] = g.set_index("idx")

REF = "zeroshot-gpt5nano"  # system used as reference for paired tests


def pred4(row_l1, row_l2):
    if row_l1 == 0:
        return "notgendered"
    if pd.isna(row_l2):
        return "unk"
    return L2_NAMES[int(row_l2)]


def eval_system(name, g, mask_country=None):
    j = gold.join(g[["pred_l1", "pred_l2"]], how="inner")
    if mask_country:
        j = j[j["country"] == mask_country]
    out = {}

    # ---- Level 1 (all rows) ----
    # NaN pred_l1 (rare parse failure) -> sentinel -1, counts as wrong but keeps
    # denominators identical across systems for paired comparisons.
    y, yh = j["gold_l1"].astype(int), j["pred_l1"].fillna(-1).astype(int)
    out["L1_acc"] = accuracy_score(y, yh)
    out["L1_macroF1"] = f1_score(y, yh, labels=[0, 1], average="macro")

    # ---- Level 2 conditioned on GOLD gendered ----
    sub = j[j["gold_l2"].notna()]
    if len(sub):
        g2 = sub["gold_l2"].astype(int)
        # system abstain (NaN) -> wrong class (-1)
        p2 = sub["pred_l2"].fillna(-1).astype(int)
        out["L2_acc"] = accuracy_score(g2, p2)
        out["L2_macroF1"] = f1_score(g2, p2, labels=[0, 1, 2], average="macro")
        out["L2_n"] = len(sub)

    # ---- End-to-end 4-way (exclude gendered_nolabel) ----
    e = j[j["gold4"].isin(FOUR_WAY)]
    g4 = e["gold4"].values
    p4 = np.array([pred4(a, b) for a, b in zip(e["pred_l1"], e["pred_l2"])])
    out["E2E_acc"] = accuracy_score(g4, p4)
    out["E2E_macroF1"] = f1_score(g4, p4, labels=FOUR_WAY, average="macro")
    out["N"] = len(j)
    return out, j


def fmt(d):
    order = ["N", "L1_acc", "L1_macroF1", "L2_n", "L2_acc", "L2_macroF1",
             "E2E_acc", "E2E_macroF1"]
    return {k: d.get(k) for k in order}


# ---------------- overall table ----------------
rows = {}
joins = {}
for name, g in preds.items():
    res, j = eval_system(name, g)
    rows[name] = fmt(res)
    joins[name] = j

tab = pd.DataFrame(rows).T
pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.3f}")
print("=" * 90)
print("OVERALL  (out-of-fold; identical instances across systems)")
print("=" * 90)
print(tab.to_string())

# ---------------- per country ----------------
for c in sorted(proc["country"].unique()):
    rows = {n: fmt(eval_system(n, g, mask_country=c)[0]) for n, g in preds.items()}
    print(f"\n----- {c} -----")
    print(pd.DataFrame(rows).T.to_string())

# ---------------- paired McNemar vs reference (Level 1) ----------------
if REF in joins:
    print("\n" + "=" * 90)
    print(f"Paired McNemar on Level 1 vs '{REF}' (same instances)")
    print("=" * 90)
    ref = joins[REF]
    ref_correct = (ref["gold_l1"] == ref["pred_l1"]).astype(int)
    for name, j in joins.items():
        if name == REF:
            continue
        common_idx = ref.index.intersection(j.index)
        a = ref_correct.loc[common_idx]
        b = (j.loc[common_idx, "gold_l1"] == j.loc[common_idx, "pred_l1"]).astype(int)
        n01 = int(((a == 0) & (b == 1)).sum())  # ref wrong, sys right
        n10 = int(((a == 1) & (b == 0)).sum())  # ref right, sys wrong
        res = mcnemar([[0, n01], [n10, 0]], exact=False, correction=True)
        print(f"{name:<24} sys>ref={n01:<4} ref>sys={n10:<4} "
              f"chi2={res.statistic:.2f} p={res.pvalue:.4f}")

# ---------------- detailed per-class for the headline system ----------------
HEAD = REF if REF in joins else list(joins)[0]
print("\n" + "=" * 90)
print(f"Per-class detail (end-to-end 4-way) -- {HEAD}")
print("=" * 90)
j = joins[HEAD]
e = j[j["gold4"].isin(FOUR_WAY)]
g4 = e["gold4"].values
p4 = np.array([pred4(a, b) for a, b in zip(e["pred_l1"], e["pred_l2"])])
print(classification_report(g4, p4, labels=FOUR_WAY, digits=3, zero_division=0))

tab.to_csv(str(RESULTS / "metrics_overall.csv"))
print("\nwrote results/metrics_overall.csv")
