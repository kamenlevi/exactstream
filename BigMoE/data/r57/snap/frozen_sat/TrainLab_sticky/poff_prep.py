"""P-OFF prep (MTP Head, Wed 11-04 18:1x; probe 2 of notes/DRAFTER_PLAN_1104.md, Kamen's download OK 17:5x): off-policy
code windows for the 125B's prefill, in L1-20's window format (~/Leap/data/l120_windows.jsonl).
Source: Magicoder-OSS-Instruct-75K (ise-uiuc, MIT), data-oss_instruct-decontaminated.jsonl. It is untrusted data: this
script only parses it as JSON (run with python -I; the data folder holds no code).
Each window: "<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" (the 125B's chat
template, enable_thinking=False, as L1-20's windows) + {solution} + <|im_end|>; asst = 1 on the solution tokens and the
final <|im_end|> (as L1-20's), gen = the first solution token. Python items only (the walk's code set is HumanEval+),
shuffled (seed 0), windows <= 2048 tokens, solutions >= 32 tokens, until --tokens assistant tokens.
Usage: [BigMoE venv] python -I sticky/poff_prep.py SRC.jsonl OUT.jsonl [--tokens 400000]"""
import argparse, json, random
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--tokens", type=int, default=400_000); ap.add_argument("--max-len", type=int, default=2048)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("/Users/kamen/BigMoE/models/bare")
END = tok.convert_tokens_to_ids("<|im_end|>")
items = []
for line in open(a.src, encoding="utf-8"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get("lang") == "python" and isinstance(r.get("problem"), str) and isinstance(r.get("solution"), str):
        items.append((int(r.get("index", len(items))), r["problem"], r["solution"]))
n_py = len(items)
random.Random(0).shuffle(items)
out, n_asst, n_tok, skipped = [], 0, 0, 0
for idx, prob, sol in items:
    pre = tok.apply_chat_template([{"role": "user", "content": prob}], tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    pre = [int(x) for x in (pre["input_ids"] if hasattr(pre, "keys") else pre)]   # a BatchEncoding in this transformers
    ans = tok.encode(sol, add_special_tokens=False) + [END]
    if len(ans) < 33 or len(pre) + len(ans) > a.max_len:
        skipped += 1
        continue
    out.append({"key": f"mc_{idx}", "project": "", "kind": "code", "ids": pre + ans,
                "asst": [0] * len(pre) + [1] * len(ans), "gen": len(pre), "split": "train"})
    n_asst += len(ans); n_tok += len(pre) + len(ans)
    if n_asst >= a.tokens:
        break
with open(a.out, "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"P-OFF prep: {n_py} python items in the source; wrote {len(out)} windows ({skipped} skipped: too long / short), "
      f"{n_tok} tokens to prefill, {n_asst} assistant tokens; mean window {n_tok / max(1, len(out)):.0f} tokens -> {a.out}")
print("check the first window's text:", repr(tok.decode(out[0]["ids"][:40])), "...",
      repr(tok.decode(out[0]["ids"][out[0]["gen"] - 8:out[0]["gen"] + 12])))
