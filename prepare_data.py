"""Build processed.csv from the GENDIS release on the Hugging Face Hub.

The dataset is gated: request access at
https://huggingface.co/datasets/<OWNER>/gendis, then authenticate with
`huggingface-cli login` (or set HF_TOKEN) before running this.

    python prepare_data.py

Writes processed.csv, the single input every other script in this repo reads.
Fold assignments ship with the release, so the splits reproduce exactly.
"""

from gendis import PROCESSED

import csv
import os

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = os.environ.get("GENDIS_HF_REPO", "manueltonneau/gendis")
L2_NAMES = {0: "neutral", 1: "negative", 2: "positive"}


def main() -> None:
    path = hf_hub_download(REPO, "gendis.parquet", repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    d = pd.read_parquet(path)

    out = pd.DataFrame({
        "idx": d["id"].astype(int),
        "country": d["country"],
        "text": d["text"],
        "gold_l1": d["level1"].astype(int),
        "gold_l2": pd.to_numeric(d["level2"], errors="coerce"),
        "fold": d["fold"].astype(int),
        "keyword_present": d["keyword_present"].astype(bool),
        # carried through so the analysis scripts are self-contained
        "stratum": d["stratum"],
        "language_group": d["language_group"],
        "code_mixed": d["code_mixed"],
        "A1_level1": d["level1_annotator1"], "A2_level1": d["level1_annotator2"],
        "A3_level1": d["level1_annotator3"],
        "A1_level2": d["level2_annotator1"], "A2_level2": d["level2_annotator2"],
        "A3_level2": d["level2_annotator3"],
    })

    def comb(r):
        if r.gold_l1 == 0:
            return "notgendered"
        if pd.isna(r.gold_l2):
            return "gendered_nolabel"
        return L2_NAMES[int(r.gold_l2)]

    out["gold4"] = out.apply(comb, axis=1)
    out = out[["idx", "country", "text", "gold_l1", "gold_l2", "gold4",
               "fold", "keyword_present", "stratum", "language_group",
               "code_mixed", "A1_level1", "A2_level1", "A3_level1",
               "A1_level2", "A2_level2", "A3_level2"]]

    assert len(out) == 9600, len(out)
    assert out["gold_l2"].notna().sum() == 4064, out["gold_l2"].notna().sum()

    # QUOTE_ALL is required: some comments contain a bare carriage return, which
    # pandas does not quote by default, and the CSV parser then reads it as a
    # row break (9,603 rows instead of 9,600, with shifted columns).
    out.to_csv(str(PROCESSED), index=False, quoting=csv.QUOTE_ALL)

    back = pd.read_csv(str(PROCESSED))
    assert len(back) == 9600, f"round-trip produced {len(back)} rows"
    assert back["country"].isin(["india", "kenya", "nigeria"]).all()
    print(f"wrote processed.csv ({len(out)} rows, round-trip verified)")
    print(out["gold4"].value_counts().to_dict())


if __name__ == "__main__":
    main()
