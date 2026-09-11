"""Quantify near-duplicate leakage across the cross-validation folds.

Motivation: the folds in ``processed.csv`` are a random comment-level split.
A reviewer may worry that comments from the same video (e.g. repeated reactions
to one viral clip) land on both sides of a fold and inflate the *trained*
systems (SetFit, XLM-R) by memorisation. Zero-shot and toxicity systems are
immune. This script measures how many comments have an exact or near-duplicate
in a *different* fold, which is the only channel through which a text-only model
can leak (there is no video-id feature).

Result (as reported in the paper): fewer than 3% of comments in any country
have a near-duplicate in a different fold, and those are almost all short
same-video reactions (emojis, one-word replies) in the not-gendered class, so
the effect on the reported macro-F1 is negligible.

Run from the repo root:  python benchmark/check_fold_duplicates.py
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED = "benchmark/processed.csv"
# Sampling files carry video_id; used only to characterise the near-dups.
SAMPLES = {
    "india": "data/Subsample_India - majority_vote.csv",
    "kenya": "data/Subsample_kenya - Majority_votes.csv",
    "nigeria": "data/Subsample_nigeria - Majority_vote.csv",
}
TEXT_COLS = ["post", "comment"]
NEAR_THRESHOLDS = (0.9, 0.8)


def normalise(s):
    """Lowercase, drop URLs and punctuation, collapse whitespace. Unicode-aware
    so Devanagari/Tamil comments are preserved."""
    s = str(s).lower().strip()
    s = re.sub(r"http\S+", "", s)
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def load_video_ids():
    frames = []
    for country, path in SAMPLES.items():
        d = pd.read_csv(path)
        d = d.loc[:, ~d.columns.duplicated()]  # Nigeria has duplicate columns
        tc = next(c for c in TEXT_COLS if c in d.columns)
        frames.append(pd.DataFrame({"country": country,
                                    "n": d[tc].map(normalise),
                                    "video_id": d["video_id"]}))
    s = pd.concat(frames, ignore_index=True)
    return s.drop_duplicates(["country", "n"]).set_index(["country", "n"])["video_id"]


def exact_report(df):
    print("\n===== Exact (normalised) duplicates =====")
    scopes = [("POOLED", df)] + [(c, df[df.country == c]) for c in SAMPLES]
    for scope, sub in scopes:
        sub = sub[sub["n"].str.len() > 0]
        if not len(sub):
            continue
        nfolds = sub.groupby("n")["fold"].nunique()
        cross = nfolds[nfolds > 1].index
        n_cross = int(sub["n"].isin(cross).sum())
        print(f"{scope:8s}: {n_cross:4d}/{len(sub)} comments "
              f"({100 * n_cross / len(sub):.1f}%) sit in an exact-dup group "
              f"spanning >1 fold")


def near_report(df):
    print("\n===== Near-duplicates (TF-IDF char 3-5gram cosine) =====")
    scopes = [("POOLED", df)] + [(c, df[df.country == c]) for c in SAMPLES]
    for thr in NEAR_THRESHOLDS:
        print(f"-- cosine >= {thr} --")
        for scope, sub in scopes:
            sub = sub.reset_index(drop=True)
            if len(sub) < 2:
                continue
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            sim = cosine_similarity(vec.fit_transform(sub["text"].astype(str)))
            np.fill_diagonal(sim, 0.0)
            folds = sub["fold"].values
            exposed = sum(1 for i in range(len(sub))
                          if np.any((sim[i] >= thr) & (folds != folds[i])))
            print(f"{scope:8s}: {exposed:4d}/{len(sub)} comments "
                  f"({100 * exposed / len(sub):.1f}%) have a near-twin in a "
                  f"different fold")


def main():
    df = pd.read_csv(PROCESSED)
    df["n"] = df["text"].map(normalise)
    vmap = load_video_ids()
    df["video_id"] = [vmap.get((c, n)) for c, n in zip(df["country"], df["n"])]
    print(f"rows: {len(df)} | video_id matched: {int(df['video_id'].notna().sum())} "
          f"| folds: {sorted(df['fold'].unique())}")
    exact_report(df)
    near_report(df)


if __name__ == "__main__":
    main()
