"""Shared helpers for the GENDIS benchmark: repository paths, label
constants, fold assignment, and the common prediction schema. Every
script imports from here so gold labels, the comparable test partition,
and the prediction format are identical across all systems."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
PROCESSED = DATA / "processed.csv"
OUTPUTS = ROOT / "outputs"
PREDICTIONS = OUTPUTS / "predictions"
RESULTS = OUTPUTS / "results"
FIGURES = OUTPUTS / "figures"

for _d in (DATA, OUTPUTS, PREDICTIONS, RESULTS):
    _d.mkdir(exist_ok=True)

import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
# country files are inconsistently named (Majority_votes / majority_vote /
# Majority_vote) -> match the common substring, case-insensitively handled below.
DATA_GLOB = os.environ.get("DATA_GLOB", "../data/*ajority_vote*.csv")
TEXT_COLS = ["post", "comment"]        # text column name varies across countries
ANNOT_COLS = ["A1_level1", "A2_level1", "A3_level1",
              "A1_level2", "A2_level2", "A3_level2"]

L2_NAMES = {0: "neutral", 1: "negative", 2: "positive"}
# Note: a Level-2 majority of 3 = "Unsure" (present in India/Nigeria). We treat
# it as no usable L2 gold (-> excluded from L2/4-way eval), since the zero-shot
# prompt and the supervised L2 head have no "unsure" class.
FOUR_WAY = ["notgendered", "neutral", "negative", "positive"]


def _load_one(f: str) -> pd.DataFrame:
    """Read one country CSV and select canonical columns, tolerating the
    naming differences and duplicate columns across the per-country files."""
    d = pd.read_csv(f)
    tcol = next((c for c in TEXT_COLS if c in d.columns), None)
    if tcol is None:
        raise ValueError(f"{f}: no text column among {TEXT_COLS}; has {list(d.columns)[:8]}")
    out = pd.DataFrame({
        "country": d["country"],
        "text": d[tcol].astype(str),
        "keyword_present": d["keyword_present"] if "keyword_present" in d else pd.NA,
        "majority_level1": d["majority_level1"],
        "majority_level2": d["majority_level2"],
    })
    for a in ANNOT_COLS:
        out[a] = d[a] if a in d.columns else pd.NA
    return out


def load_data(data_glob: str = DATA_GLOB) -> pd.DataFrame:
    """Load and concatenate all per-country majority-vote CSVs.

    Produces canonical columns used everywhere downstream:
      text, gold_l1 (0/1), gold_l2 (0/1/2 or NaN), gold4 (string), idx.
    """
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No CSVs match {data_glob}")
    df = pd.concat([_load_one(f) for f in files], ignore_index=True)

    df["gold_l1"] = df["majority_level1"].astype(int)
    g2 = pd.to_numeric(df["majority_level2"], errors="coerce")
    df["gold_l2"] = g2.where(g2.isin([0, 1, 2]))   # 3 (unsure) and NaN -> NaN

    def comb(r):
        if r.gold_l1 == 0:
            return "notgendered"
        if pd.isna(r.gold_l2):
            return "gendered_nolabel"   # gendered but no usable L2 majority
        return L2_NAMES[int(r.gold_l2)]

    df["gold4"] = df.apply(comb, axis=1)
    df = df.reset_index(drop=True)
    df["idx"] = df.index
    return df


def assign_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Single fixed stratified k-fold assignment shared by ALL systems.

    Stratified by country x 4-way label so each fold is balanced on both
    country and the rare positive class. Strata too small to split are
    coarsened to country level to keep StratifiedKFold valid.
    """
    strat = (df["country"].astype(str) + "|" + df["gold4"].astype(str))
    counts = strat.value_counts()
    small = counts[counts < n_folds].index
    strat = strat.where(~strat.isin(small), df["country"].astype(str) + "|rare")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(df)), strat.values)):
        fold[te] = f
    out = df.copy()
    out["fold"] = fold
    return out


def write_predictions(idx, country, system, pred_l1, pred_l2, path):
    """Standard prediction schema: one row per comment.

    pred_l1 : 0/1
    pred_l2 : 0/1/2  (the system's 3-way judgment assuming gendered),
              or NaN  (system declined / not applicable).
    """
    pd.DataFrame({
        "idx": idx, "country": country, "system": system,
        "pred_l1": pred_l1, "pred_l2": pred_l2,
    }).to_csv(path, index=False)
    print(f"wrote {path} ({len(idx)} rows)")


def countries(df):
    return sorted(df["country"].unique())
