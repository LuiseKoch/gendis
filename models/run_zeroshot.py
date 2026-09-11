"""Zero-shot LLM over processed.csv via the OpenAI Batch API (cost-saving).

End-to-end: builds the batch, submits, polls to completion, fetches, and
writes predictions/zeroshot.csv in the standard schema. Training-free, so it
is simply scored on the same comparable partition as everything else.

For Kenya this was already run (see ../gpt5nano_eval/); use ingest_zeroshot.py
to reuse those results. Use THIS script for the newly added countries.

  export OPENAI_KEY_FILE=/path/to/openai.key
  python run_zeroshot.py
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from gendis import PREDICTIONS, PROCESSED, write_predictions

import json
import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI

from prompt import SYSTEM_PROMPT, USER_TEMPLATE, JSON_SCHEMA, MODEL

KEY = open(os.environ["OPENAI_KEY_FILE"]).read().strip()
client = OpenAI(api_key=KEY)

df = pd.read_csv(str(PROCESSED))

with open(str(PREDICTIONS / "zeroshot_input.jsonl"), "w") as f:
    for _, r in df.iterrows():
        f.write(json.dumps({
            "custom_id": f"idx-{int(r.idx)}",
            "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": MODEL,
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": USER_TEMPLATE.format(text=str(r.text))}],
                     "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA}},
        }) + "\n")

up = client.files.create(file=open(str(PREDICTIONS / "zeroshot_input.jsonl"), "rb"),
                         purpose="batch")
batch = client.batches.create(input_file_id=up.id,
                              endpoint="/v1/chat/completions",
                              completion_window="24h")
print("batch:", batch.id)
while True:
    b = client.batches.retrieve(batch.id)
    print(b.status, b.request_counts, flush=True)
    if b.status in ("completed", "failed", "expired", "cancelled"):
        break
    time.sleep(30)
assert b.status == "completed", b.status

preds = {}
for line in client.files.content(b.output_file_id).text.splitlines():
    if not line.strip():
        continue
    o = json.loads(line)
    idx = int(o["custom_id"].split("-")[1])
    d = json.loads(o["response"]["body"]["choices"][0]["message"]["content"])
    preds[idx] = (int(d["level1"]), int(d["level2"]))

p1 = [preds.get(i, (np.nan, np.nan))[0] for i in df["idx"]]
p2 = [preds.get(i, (np.nan, np.nan))[1] for i in df["idx"]]
p2 = [x if x in (0, 1, 2) else np.nan for x in p2]
write_predictions(df["idx"], df["country"], "zeroshot-gpt5nano", p1, p2,
                  str(PREDICTIONS / "zeroshot.csv"))
