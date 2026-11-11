"""CERT's extra math items (docs/PREREG_cert_2026-11-08.md): dec_mathx (keys mathx/<the source key>: BigMoE's screen maps the prefix to dec_mathx) = 25 MATH-500 problems of level >= 3 outside dec_math's
25 (random.Random(1107)) + GSM8K test 0-24 (the range MTP Head's drafter data excluded), with dec_math's template and caps.
  ~/BigMoE/.venv/bin/python tools/cert_build.py      # the chat template needs BigMoE's venv"""
import hashlib, json, os, random, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from evalsuite import data as D
RAW = os.path.join(ROOT, "data", "raw")
instr = "Solve the problem. Put the final answer in \\boxed{}.\n\n"
tok = D.tokenizer()
core = {it["key"] for it in D.load("dec_math")}
m5 = [json.loads(l) for l in open(os.path.join(RAW, "math500_test.jsonl"))]
pool = [p for p in m5 if int(p.get("level", 0)) >= 3 and f"math500/{p['unique_id']}" not in core]
pick = random.Random(1107).sample(pool, 25)
gsm = [json.loads(l) for l in open(os.path.join(RAW, "gsm8k_test.jsonl"))][0:25]
rows = [dict(key=f"mathx/math500/{p['unique_id']}", set="math", gold=p["answer"], max_tokens=1536, user=instr + p["problem"],
             prompt_ids=D.chat(tok, instr + p["problem"])) for p in pick]
rows += [dict(key=f"mathx/gsm8k/{j}", set="math", gold=p["answer"].split("####")[-1].strip().replace(",", ""), max_tokens=768,
              user=instr + p["question"], prompt_ids=D.chat(tok, instr + p["question"])) for j, p in enumerate(gsm)]
assert len(rows) == 50 and not ({r["key"].split("/", 1)[1] for r in rows} & core)
D.write("dec_mathx", rows)
p = os.path.join(D.ITEMS, "dec_mathx.jsonl")
print(f"data/items/dec_mathx.jsonl: {len(rows)} items, sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]}")
