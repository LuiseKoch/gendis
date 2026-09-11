"""Zero-shot gendered-speech classification with OPEN decoder LLMs (vLLM).

Faithful mirror of the gpt-5-nano setup: it reuses the SAME SYSTEM_PROMPT,
USER_TEMPLATE, and JSON label schema from prompt.py, and writes predictions in
the SAME schema as every other system, so evaluate.py picks the results up
automatically and the comparison is apples-to-apples.

Output: predictions/zeroshot-<tag>.csv with system name "zeroshot-<tag>"
        columns idx,country,system,pred_l1,pred_l2  (pred_l2 in {0,1,2} or NaN).

Run on a GPU box (virgil). Examples (one H100 each, ~minutes):

  python run_zeroshot_open.py --model meta-llama/Llama-3.1-8B-Instruct --tag llama31-8b --tp 1
  python run_zeroshot_open.py --model google/gemma-2-9b-it            --tag gemma2-9b  --tp 1

Match --model EXACTLY to the id shown by `huggingface-cli scan-cache` so the
shared cache is reused (and set HF_HUB_OFFLINE=1 to fail fast instead of
silently downloading). --tp must equal the number of visible GPUs.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from gendis import PROCESSED

import argparse
import json
import re
import numpy as np
import pandas as pd

from prompt import SYSTEM_PROMPT, USER_TEMPLATE, JSON_SCHEMA


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id (match `huggingface-cli scan-cache` exactly to reuse the cache)")
    ap.add_argument("--tag", required=True,
                    help="short tag -> system 'zeroshot-<tag>', file predictions/zeroshot-<tag>.csv")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (= number of visible GPUs)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--data", default=str(PROCESSED))
    ap.add_argument("--no-guided", action="store_true",
                    help="disable guided JSON decoding (rely on free-form JSON parsing instead)")
    return ap.parse_args()


_JSON_RE = re.compile(r"\{.*?\}", re.S)
# Fallback for models that ignore guided JSON and emit YAML-style key:value
# (observed with Gemma-2 under VLLM_ATTENTION_BACKEND=XFORMERS).
_KV_RE = re.compile(r"level\s*1\s*[:=]\s*(-?\d+).*?level\s*2\s*[:=]\s*(-?\d+)",
                    re.S | re.I)


def parse_label(txt):
    """Return (l1, l2). l1 in {0,1} or NaN; l2 mapped to {0,1,2} else NaN."""
    d = None
    try:
        d = json.loads(txt)
    except Exception:
        m = _JSON_RE.search(txt or "")
        if m:
            try:
                d = json.loads(m.group(0))
            except Exception:
                d = None
    if not isinstance(d, dict):
        m = _KV_RE.search(txt or "")
        if m:
            d = {"level1": m.group(1), "level2": m.group(2)}
    if not isinstance(d, dict):
        return (np.nan, np.nan)
    try:
        l1 = int(d["level1"])
        l2 = int(d["level2"])
    except Exception:
        return (np.nan, np.nan)
    l1 = l1 if l1 in (0, 1) else np.nan
    l2 = l2 if l2 in (0, 1, 2) else np.nan
    return (l1, l2)


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    df = pd.read_csv(args.data)

    # Guided JSON decoding guarantees schema-valid output. Optional: degrade
    # gracefully if this vLLM build exposes a different structured-output API.
    guided, guided_kw = None, None
    if not args.no_guided:
        # vLLM >= 0.11 renamed GuidedDecodingParams -> StructuredOutputsParams
        # and the SamplingParams field guided_decoding -> structured_outputs.
        try:
            from vllm.sampling_params import StructuredOutputsParams
            guided = StructuredOutputsParams(json=JSON_SCHEMA["schema"])
            guided_kw = "structured_outputs"
        except Exception:  # noqa: BLE001
            try:
                from vllm.sampling_params import GuidedDecodingParams
                guided = GuidedDecodingParams(json=JSON_SCHEMA["schema"])
                guided_kw = "guided_decoding"
            except Exception as e:  # noqa: BLE001
                print(f"WARN: guided decoding unavailable ({e}); using free-form JSON parsing")
    if guided is not None:
        print(f"guided decoding enabled via {guided_kw}")

    sp_kwargs = dict(temperature=0.0, max_tokens=32, seed=42)
    if guided is not None:
        sp_kwargs[guided_kw] = guided
    sp = SamplingParams(**sp_kwargs)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              trust_remote_code=True)

    # Some chat templates (e.g. Gemma) reject the `system` role. Probe once
    # against the tokenizer; if it fails, fold the system prompt into the user
    # turn rather than dropping it.
    tok = llm.get_tokenizer()
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            add_generation_prompt=True, tokenize=False)
        convos = [[{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": USER_TEMPLATE.format(text=str(t))}]
                  for t in df["text"]]
    except Exception:
        print(f"[{args.tag}] model rejects system role; merging system into user")
        convos = [[{"role": "user",
                    "content": SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(text=str(t))}]
                  for t in df["text"]]
    outs = llm.chat(convos, sp)
    texts = [o.outputs[0].text for o in outs]

    pairs = [parse_label(t) for t in texts]
    p1 = [a for a, _ in pairs]
    p2 = [b for _, b in pairs]

    nfail = sum(1 for a in p1 if not (a in (0, 1)))
    print(f"\n[{args.tag}] parse failures (no valid level1): {nfail}/{len(p1)} = {nfail/len(p1):.1%}")
    gendered = sum(1 for a in p1 if a == 1)
    print(f"[{args.tag}] predicted gendered (L1=1): {gendered}/{len(p1)}")

    system = f"zeroshot-{args.tag}"
    out_path = f"predictions/zeroshot-{args.tag}.csv"
    pd.DataFrame({
        "idx": df["idx"], "country": df["country"], "system": system,
        "pred_l1": p1, "pred_l2": p2,
    }).to_csv(out_path, index=False)
    print(f"[{args.tag}] wrote {out_path} ({len(p1)} rows)")


if __name__ == "__main__":
    main()
