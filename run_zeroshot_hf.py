"""Transformers fallback for zero-shot open models.

Same prompt, same parser, same output schema as run_zeroshot_open.py. Exists
because vLLM's flash-attn build on this machine cannot do the tanh attention
logit softcapping that Gemma-2 requires, and the attention backend cannot be
overridden in this build. Slower than vLLM but produces identical-format output.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import SYSTEM_PROMPT, USER_TEMPLATE
from run_zeroshot_open import parse_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="processed.csv")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"[{args.tag}] rows: {len(df)}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",   # required for Gemma-2 softcapping
    ).eval()

    # Gemma has no system role: fold the system prompt into the user turn,
    # exactly as run_zeroshot_open.py does.
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"},
                                 {"role": "user", "content": "y"}], tokenize=False)
        supports_system = True
    except Exception:
        supports_system = False
    print(f"[{args.tag}] system role supported: {supports_system}")
    prompts = []
    for t in df["text"].astype(str):
        if supports_system:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(text=t)}]
        else:
            msgs = [{"role": "user",
                     "content": SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(text=t)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True))

    texts = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1536, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
        if (i // args.batch_size) % 25 == 0:
            print(f"  {min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    pairs = [parse_label(t) for t in texts]
    p1 = [a for a, _ in pairs]
    p2 = [b for _, b in pairs]
    nfail = sum(1 for a in p1 if a not in (0, 1))
    print(f"\n[{args.tag}] parse failures: {nfail}/{len(p1)} = {nfail/len(p1):.1%}")
    print(f"[{args.tag}] predicted gendered: {sum(1 for a in p1 if a == 1)}/{len(p1)}")

    out_path = f"predictions/zeroshot-{args.tag}.csv"
    pd.DataFrame({"idx": df["idx"], "country": df["country"],
                  "system": f"zeroshot-{args.tag}",
                  "pred_l1": p1, "pred_l2": p2}).to_csv(out_path, index=False)
    print(f"[{args.tag}] wrote {out_path} ({len(p1)} rows)")


if __name__ == "__main__":
    main()
