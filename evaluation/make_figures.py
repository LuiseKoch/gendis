"""Generate paper figures (CMU Serif to match the ACL body font).
Writes PDFs into the Overleaf latex/ directory.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from gendis import PREDICTIONS, PROCESSED

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

for f in font_manager.findSystemFonts(fontpaths=os.environ.get("SERIF_FONT_DIR", ""), fontext="ttf"):
    font_manager.fontManager.addfont(f)
rc("font", **{"family": "serif", "serif": ["CMU Serif"]})
rc("text", usetex=False)

OUT = os.environ.get("FIGURE_OUT_DIR", "figures")

# ---------------------------------------------------------------- Figure 2
# Cross-country label distribution (each country = 100%).
cats = ["Not gendered", "Negative", "Neutral", "Positive", "Unsure", "No L2 majority"]
# Recomputed on the full 9,600-comment dataset (rounds 1+2).
counts = {
    "India":   [1730, 631, 704, 113, 8, 14],
    "Kenya":   [1850, 503, 680, 125, 0, 42],
    "Nigeria": [1577, 734, 237, 337, 9, 306],
}
colors = ["#bdbdbd", "#d73027", "#4575b4", "#1a9850", "#fee090", "#7b3294"]
countries = list(counts)
data = np.array([counts[c] for c in countries], dtype=float)
data = data / data.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(6.2, 3.7))
left = np.zeros(len(countries))
for j, (cat, col) in enumerate(zip(cats, colors)):
    vals = data[:, j]
    ax.barh(countries, vals, left=left, color=col, label=cat, edgecolor="white", linewidth=0.6)
    for i, v in enumerate(vals):
        if v >= 6:
            ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=19)
    left += vals
ax.set_xlim(0, 100)
ax.set_xlabel("Share of comments (%)", fontsize=21)
ax.tick_params(axis="y", labelsize=23)
ax.tick_params(axis="x", labelsize=17)
ax.invert_yaxis()
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.legend(ncol=3, fontsize=16, loc="lower center", bbox_to_anchor=(0.5, 1.03),
          frameon=False, columnspacing=1.0, handletextpad=0.4)
fig.savefig(f"{OUT}/fig_label_distribution.pdf", bbox_inches="tight")
print("wrote fig_label_distribution.pdf")

# ---------------------------------------------------------------- Figure 3
# End-to-end 4-way confusion for SetFit-CV (row-normalised = recall).
labels = ["not gendered", "neutral", "negative", "positive"]
# Computed from predictions rather than hardcoded, so it tracks the data.
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
CM_SYS = os.environ.get("CM_SYS", "setfit_cv.csv")
_proc = pd.read_csv(str(PROCESSED))
_pr = pd.read_csv(str(PREDICTIONS / CM_SYS))[["idx", "pred_l1", "pred_l2"]]
_j = _proc.merge(_pr, on="idx")
_L2 = {0: "neutral", 1: "negative", 2: "positive"}
def _p4(a, b):
    if int(a) == 0:
        return "not gendered"
    return "unk" if pd.isna(b) else _L2[int(float(b))]
_g = {"notgendered": "not gendered", "neutral": "neutral",
      "negative": "negative", "positive": "positive"}
_e = _j[_j.gold4.isin(_g)]
cm = confusion_matrix([_g[x] for x in _e.gold4],
                      [_p4(a, b) for a, b in zip(_e.pred_l1, _e.pred_l2)],
                      labels=labels).astype(float)
print(f"confusion from {CM_SYS} on {len(_e)} rows")
cmn = cm / cm.sum(axis=1, keepdims=True)

import seaborn as sns
fig, ax = plt.subplots(figsize=(5.2, 4.6))
annot = np.array([[f"{v*100:.0f}" for v in row] for row in cmn])
sns.heatmap(cmn, cmap="Blues", vmin=0, vmax=1, annot=annot, fmt="", cbar=False,
            square=True, linewidths=0.5, linecolor="white",
            annot_kws={"fontsize": 14}, xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=13)
ax.set_yticklabels(labels, rotation=0, fontsize=13)
ax.set_xlabel("Predicted", fontsize=15)
ax.set_ylabel("Gold", fontsize=15)
plt.tight_layout()
fig.savefig(f"{OUT}/fig_confusion.pdf", bbox_inches="tight")
print("wrote fig_confusion.pdf")
