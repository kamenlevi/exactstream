"""Leap L5 data (Sat 2026-10-24, QUEUE L1-04 block-16 DFlash): code / math prompts the evaluation never uses, rendered
exactly as EvalSuite's tf set (evalsuite/data.py build_tf: chat() with thinking off, the same user text).
- Code: HumanEval+ (local evalplus cache, no download). EvalSuite's random pick (Random(77): 130 IFEval, then 84 of
  the 164 HumanEval+) gives tf_code = the first 24 and its decode set = the next 60. The other 80 are used here.
- Math: GSM8K test problems 0-119 (tf_gsm uses 600-659, the decode set 700-724), with EvalSuite's GSM instruction.
- Check: this rendering must reproduce the 24 tf_code prompts' token ids in cache/ref/tf_gen.jsonl exactly.
Output ~/Leap/data/dfl2e_samples.jsonl (key, kind, prompt, split: val every 10th). Usage: [BigMoE venv] dfl2e_data.py"""
import json, os, random, sys
from pathlib import Path
from mlx_lm.tokenizer_utils import load as load_tok

E = os.path.expanduser("~/EvalSuite")
OUT = os.path.expanduser("~/Leap/data/dfl2e_samples.jsonl")
GSM_INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
tok = load_tok(Path(os.path.expanduser("~/BigMoE/models/bare")))

def chat(user):                                      # = evalsuite.data.chat(tok, user)
    text = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False,
                                   tokenize=False)
    return tok.encode(text, add_special_tokens=False)

def he_user(prompt):
    return ("Complete the following Python function. Reply with the whole function in one ```python code block.\n\n"
            f"```python\n{prompt}```")

he = sorted((r["task_id"], r) for r in map(json.loads, open(os.path.expanduser(
    "~/Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl"))))
n_ife = sum(1 for _ in open(os.path.join(E, "data/raw/ifeval_input_data.jsonl")))
rng = random.Random(77)
rng.sample(range(n_ife), 130)                        # build_tf's IFEval pick comes first
he_pick = rng.sample(range(len(he)), 84)
ref = {r["key"]: r["prompt_ids"] for r in map(json.loads, open(os.path.join(E, "cache/ref/tf_gen.jsonl")))
       if r["key"].startswith("tf_code/")}
mine = {f"tf_code/{he[j][0]}": chat(he_user(he[j][1]["prompt"])) for j in he_pick[:24]}
same = sum(mine.get(k) == list(v) for k, v in ref.items())
print(f"[l5data] {len(he)} HumanEval+ problems; rendering check: {same}/{len(ref)} tf_code prompts identical", flush=True)
if same != len(ref) or len(ref) != 24:
    sys.exit("[l5data] rendering differs from EvalSuite's: stopping")
used = set(he_pick)
rows = [{"key": f"he_{he[j][0].split('/')[1]}", "kind": "code", "prompt": chat(he_user(he[j][1]["prompt"]))}
        for j in range(len(he)) if j not in used]
gsm = [json.loads(l) for l in open(os.path.join(E, "data/raw/gsm8k_test.jsonl"))]
rows += [{"key": f"gsm_{j}", "kind": "math", "prompt": chat(GSM_INSTR + gsm[j]["question"])} for j in range(120)]
for i, r in enumerate(rows):
    r["split"] = "val" if i % 10 == 9 else "train"
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"[l5data] {sum(r['kind'] == 'code' for r in rows)} code + {sum(r['kind'] == 'math' for r in rows)} math prompts "
      f"({sum(r['split'] == 'val' for r in rows)} val), {sum(len(r['prompt']) for r in rows)} prompt tokens -> {OUT}", flush=True)
