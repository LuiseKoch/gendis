"""Break down model performance by language group, joining the automatic
language labels (language_id/) to the benchmark predictions.

Language group is derived from the per-comment primary language category:
  English | Local (non-English monolingual) | Code-mixed | Other (off-inventory).
Outputs a tidy CSV and prints LaTeX-ready rows for the appendix table.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from gendis import FOUR_WAY, L2_NAMES, PREDICTIONS, PROCESSED, RESULTS

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

PROC = str(PROCESSED)

# display name -> prediction file (CV variants for the trained systems)
SYSTEMS = [
    ("SetFit (CV)",      "setfit_cv.csv"),
    ("\\texttt{gpt-5-nano}", "zeroshot.csv"),
    ("Llama-3.1-8B",     "zeroshot-llama31-8b.csv"),
    ("Gemma-2-9B",       "zeroshot-gemma2-9b.csv"),
    ("XLM-R (CV)",       "ft_xlm-roberta-base_cv.csv"),
    ("Toxicity",         "toxicity_multilingual-toxic-xlm-roberta.csv"),
    ("Keyword floor",    "floor.csv"),
]
GROUPS = ["English", "Local", "Code-mixed", "Other"]


def pred4(l1, l2):
    if l1 == 0:
        return "notgendered"
    if pd.isna(l2):
        return "unk"
    return L2_NAMES[int(float(l2))]


def lang_group(pc):
    return {"English": "English", "Code-mixed": "Code-mixed",
            "Other (foreign)": "Other"}.get(pc, "Local")


def metrics(d):
    """Return (L1, L2, E2E) macro-F1 for a slice; nan if too small."""
    if len(d) < 15:
        return (np.nan, np.nan, np.nan)
    y = d.gold_l1.astype(int)
    yh = d.pred_l1.astype(float).round().astype(int)
    l1 = f1_score(y, yh, labels=[0, 1], average="macro", zero_division=0)
    s = d[d.gold_l2.notna()]
    l2 = (f1_score(s.gold_l2.astype(int),
                   s.pred_l2.fillna(-1).astype(float).round().astype(int),
                   labels=[0, 1, 2], average="macro", zero_division=0)
          if len(s) >= 15 else np.nan)
    e = d[d.gold4.isin(FOUR_WAY)]
    p4 = [pred4(a, b) for a, b in zip(e.pred_l1.astype(float).round().astype(int), e.pred_l2)]
    e2e = f1_score(e.gold4.values, p4, labels=FOUR_WAY, average="macro", zero_division=0)
    return (l1, l2, e2e)


proc = pd.read_csv(PROC)
m = proc.rename(columns={"language_group": "lgroup"})

ns = {g: int((m.lgroup == g).sum()) for g in GROUPS}
print("N per group:", ns)

rows = []
import os
for name, f in SYSTEMS:
    if not os.path.exists(str(PREDICTIONS / f)):
        print(f"  SKIP {name}: predictions/{f} not present")
        continue
    pr = pd.read_csv(str(PREDICTIONS / f))[["idx", "pred_l1", "pred_l2"]]
    d = m.merge(pr, on="idx").dropna(subset=["pred_l1"])
    rec = {"system": name}
    for g in GROUPS:
        l1, l2, e2e = metrics(d[d.lgroup == g])
        rec[f"{g}_L1"], rec[f"{g}_L2"], rec[f"{g}_E2E"] = l1, l2, e2e
    rows.append(rec)
res = pd.DataFrame(rows)
res.to_csv(str(RESULTS / "by_language.csv"), index=False)
print("wrote results/by_language.csv")


def cell(x):
    return f"{x:.3f}".lstrip("0") if pd.notna(x) else "--"


print("\n% --- LaTeX rows: Level 1 | Level 2 (Eng Loc Mix Oth each) ---")
for _, r in res.iterrows():
    l1 = " & ".join(cell(r[f"{g}_L1"]) for g in GROUPS)
    l2 = " & ".join(cell(r[f"{g}_L2"]) for g in GROUPS)
    print(f"{r.system} & {l1} & {l2} \\\\")
print("% group Ns:", " ".join(f"{g}={ns[g]}" for g in GROUPS))
