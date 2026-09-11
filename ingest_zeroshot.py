"""Convert zero-shot LLM batch output into the standard prediction schema.

The zero-shot run (see ../gpt5nano_eval/) already produced one label per
comment over the full sample, so it requires no training and is simply scored
on the same comparable partition. This maps that output onto processed.csv.

Two supported inputs (whichever you have):
  A) ../gpt5nano_eval/predictions_merged.csv  (already merged; Kenya done)
  B) a raw OpenAI batch results .jsonl         (for new countries)

Alignment is by `post` text (exact match), since processed.csv is rebuilt
when new countries are added.
"""
import json
import sys
import numpy as np
import pandas as pd

proc = pd.read_csv("processed.csv")
SYS = "zeroshot-gpt5nano"


def from_merged(path):
    m = pd.read_csv(path)
    m = m[["post", "pred_level1", "pred_level2"]].copy()
    m["text"] = m["post"].astype(str)
    return m.rename(columns={"pred_level1": "l1", "pred_level2": "l2"})


def from_jsonl(path, batch_input):
    # map custom_id -> text via the batch input file
    id2text = {}
    for line in open(batch_input):
        o = json.loads(line)
        id2text[o["custom_id"]] = o["body"]["messages"][-1]["content"]
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        o = json.loads(line)
        c = o["response"]["body"]["choices"][0]["message"]["content"]
        d = json.loads(c)
        rows.append({"cid": o["custom_id"], "l1": int(d["level1"]),
                     "l2": int(d["level2"])})
    r = pd.DataFrame(rows)
    r["text"] = r["cid"].map(id2text)  # contains the wrapped prompt; align by row order instead
    return r


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "../gpt5nano_eval/predictions_merged.csv"
    m = from_merged(src)
    # identical comment texts get identical zero-shot labels; keep one per text
    m = m.drop_duplicates(subset="text", keep="first")

    merged = proc.merge(m[["text", "l1", "l2"]], on="text", how="left")
    merged = merged.drop_duplicates(subset="idx", keep="first")
    pred_l1 = merged["l1"].values
    pred_l2 = merged["l2"].where(merged["l2"].isin([0, 1, 2]), np.nan).values

    missing = np.isnan(pred_l1).sum()
    if missing:
        print(f"WARN: {missing} comments had no zero-shot match by text")

    from common import write_predictions
    write_predictions(merged["idx"], merged["country"], SYS,
                      pred_l1, pred_l2, "predictions/zeroshot.csv")
