"""Score every system on the RANDOM stratum only (stratum == 'random_uniform').

This is the keyword-independent, uniformly drawn half of the round-2 sample, so
it approximates real-world deployment rather than the keyword-enriched pool.
Metrics mirror evaluate.py exactly so the two tables are directly comparable.

Unlike round 1 (where the keyword-free stratum held only 200 comments/country and
supported Level 1 only), the random stratum is now 1,500/country, so Level 2 and
the end-to-end 4-way score are reported as well.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from gendis.paths import PREDICTIONS, PROCESSED, RESULTS

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from gendis.common import L2_NAMES, FOUR_WAY

proc = pd.read_csv(str(PROCESSED))
rand = proc[proc["stratum"] == "random_uniform"]
print(f"random stratum: {len(rand)} rows | " +
      rand.groupby("country").size().to_string().replace("\n", " | "))
gold = rand.set_index("idx")

preds = {}
for f in sorted(sorted(str(p) for p in PREDICTIONS.glob("*.csv"))):
    p = pd.read_csv(f)
    for sysname, g in p.groupby("system"):
        preds[sysname] = g.set_index("idx")
if not preds:
    raise SystemExit("no prediction files found")


def pred4(a, b):
    if a == 0:
        return "notgendered"
    return "unk" if pd.isna(b) else L2_NAMES[int(b)]


def score(g, country=None):
    j = gold.join(g[["pred_l1", "pred_l2"]], how="inner")
    if country:
        j = j[j["country"] == country]
    o = {"N": len(j)}
    if not len(j):
        return o
    y, yh = j["gold_l1"].astype(int), j["pred_l1"].fillna(-1).astype(int)
    o["L1_acc"] = accuracy_score(y, yh)
    o["L1_macroF1"] = f1_score(y, yh, labels=[0, 1], average="macro")
    sub = j[j["gold_l2"].notna()]
    if len(sub):
        g2, p2 = sub["gold_l2"].astype(int), sub["pred_l2"].fillna(-1).astype(int)
        o["L2_n"] = len(sub)
        o["L2_acc"] = accuracy_score(g2, p2)
        o["L2_macroF1"] = f1_score(g2, p2, labels=[0, 1, 2], average="macro")
    e = j[j["gold4"].isin(FOUR_WAY)]
    if len(e):
        g4 = e["gold4"].values
        p4 = np.array([pred4(a, b) for a, b in zip(e["pred_l1"], e["pred_l2"])])
        o["E2E_acc"] = accuracy_score(g4, p4)
        o["E2E_macroF1"] = f1_score(g4, p4, labels=FOUR_WAY, average="macro")
    return o


ORDER = ["N", "L1_acc", "L1_macroF1", "L2_n", "L2_acc", "L2_macroF1",
         "E2E_acc", "E2E_macroF1"]
pd.set_option("display.width", 170, "display.float_format", lambda x: f"{x:.3f}")

tab = pd.DataFrame({n: {k: score(g).get(k) for k in ORDER}
                    for n, g in preds.items()}).T
print("=" * 92)
print("RANDOM STRATUM ONLY (keyword-independent, uniform draw)")
print("=" * 92)
print(tab.to_string())
tab.to_csv(str(RESULTS / "metrics_random_stratum.csv"))

for c in sorted(rand["country"].unique()):
    t = pd.DataFrame({n: {k: score(g, c).get(k) for k in ORDER}
                      for n, g in preds.items()}).T
    print(f"\n----- {c} -----")
    print(t.to_string())
    t.to_csv(RESULTS / f"metrics_random_stratum_{c}.csv")

print("\nwrote results/metrics_random_stratum*.csv")
