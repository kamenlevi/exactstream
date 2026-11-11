"""Leap L3 data (2026-10-23): agentic training windows for the DFlash2 fine-tune, from Kamen's Claude Code sessions,
rendered like the route4 eval prompts: SpecDecode's message loader (extract.py) and prompt format (render.py: SYSTEM +
TOOLS, the 125B chat template, thinking off). One window per session (the last <= CTX chars ending on an assistant
message), sessions shuffled and taken round-robin across projects.
Excluded: any session whose messages hold (a) an assistant message of SpecDecode's 29 route4 samples (contexts and
targets), (b) the start of any of the 29 targets, or (c) a piece of the 125B's own outputs on the 12 eval turns
(Chat1000 agent125). Output ~/Leap/data/dfl2c_windows.jsonl: key, project, ids, asst (1 = assistant-turn token),
split (train / val by session). Never leaves this Mac. Usage: [BigMoE venv] dfl2c_data.py [TOKENS] [CTX_CHARS]"""
import glob, json, os, random, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import extract as X                                  # read-only use of SpecDecode's loader and prompt format
import render as R
from mlx_lm.tokenizer_utils import load as load_tok

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 250_000
X.CTX_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 28_000
OUT = os.path.expanduser("~/Leap/data/dfl2c_windows.jsonl")
tok = load_tok(Path(os.path.expanduser("~/BigMoE/models/bare")))

def payload(m):
    return (m.get("content") or "") + json.dumps(m.get("tool_calls") or "", ensure_ascii=False)

# (a) + (b): signatures of the 29 route4 samples; (c): pieces of the 125B's outputs on the 12 eval turns
sigs, pieces = set(), []
for l in open(os.path.expanduser("~/SpecDecode/data/samples.jsonl")):
    s = json.loads(l)
    for m in s["messages"] + [s["target"]]:
        if m["role"] == "assistant" and len(payload(m)) >= 80:
            sigs.add(payload(m)[:300])
    t = s["target"]
    body = t.get("content") or ""
    for c in t.get("tool_calls") or []:
        a = c["function"]["arguments"]
        body += a.get("old_string") or a.get("content") or a.get("command") or ""
    if len(body.strip()) >= 60:
        pieces.append(body.strip()[:80])
for f in sorted(glob.glob(os.path.expanduser("~/Chat1000/data/agent125/*.npz"))):
    txt = tok.decode([int(x) for x in np.load(f)["out"]])
    for k in range(0, max(1, len(txt) - 60), max(1, len(txt) // 3)):
        p = txt[k:k + 60]
        if len(p.strip()) >= 40:
            pieces.append(p)
print(f"[data] {len(sigs)} assistant signatures, {len(pieces)} text pieces to exclude", flush=True)

paths = sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")))
random.seed(0); random.shuffle(paths)
by_proj = {}
for p in paths:
    by_proj.setdefault(p.split("/")[-2], []).append(p)
order, k = [], 0
while any(by_proj.values()):                     # round-robin over projects
    for proj in sorted(by_proj):
        if by_proj[proj]:
            order.append(by_proj[proj].pop())
print(f"[data] {len(order)} session files in {len(set(p.split('/')[-2] for p in order))} projects", flush=True)

stats = {"excluded": 0, "short": 0, "template": 0}
rows, total, asst_total, n_sess = [], 0, 0, 0
for path in order:
    if total >= TOKENS:
        break
    msgs = X.load_session(path)
    if len(msgs) < 8 or sum(1 for m in msgs if m["role"] == "assistant") < 3:
        stats["short"] += 1
        continue
    blob = "\n".join(payload(m) for m in msgs)
    if any(payload(m)[:300] in sigs for m in msgs if m["role"] == "assistant") or any(p in blob for p in pieces):
        stats["excluded"] += 1
        continue
    ends = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i >= 6]
    if not ends:
        stats["short"] += 1
        continue
    win = X.context_before(msgs, random.choice(ends) + 1)
    try:
        text = tok.apply_chat_template([{"role": "system", "content": R.SYSTEM}] + win, tools=R.TOOLS, tokenize=False,
                                       enable_thinking=False)
    except Exception:
        stats["template"] += 1
        continue
    enc = getattr(tok, "_tokenizer", tok)(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if not 1500 <= len(ids) <= 12000:
        stats["short"] += 1
        continue
    assert ids == tok.encode(text, add_special_tokens=False)
    cm = np.zeros(len(text) + 1, bool)
    for m in re.finditer(r"<\|im_start\|>assistant\n", text):
        e = text.find("<|im_end|>", m.end())
        cm[m.end():(e + len("<|im_end|>")) if e >= 0 else len(text)] = True
    asst = [int(cm[s] and e > s) for s, e in enc["offset_mapping"]]
    proj = path.split("/")[-2]
    rows.append({"key": f"{proj}__{Path(path).stem[:8]}", "project": proj, "ids": ids, "asst": asst,
                 "split": "val" if n_sess % 10 == 9 else "train"})
    n_sess += 1; total += len(ids); asst_total += sum(asst)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
tr = [r for r in rows if r["split"] == "train"]
print(f"[data] {len(rows)} windows ({len(tr)} train / {len(rows) - len(tr)} val) from {len(set(r['project'] for r in rows))} "
      f"projects: {total} tokens, assistant {asst_total} ({asst_total / max(1, total):.0%}); skipped {stats}", flush=True)
print(f"[data] train tokens {sum(len(r['ids']) for r in tr)}, train assistant {sum(sum(r['asst']) for r in tr)} -> {OUT}")
