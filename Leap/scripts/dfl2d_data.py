"""Leap L4 data (Sat 2026-10-24, DFL2D_PREREG.md part 2): agentic turns for the 125B to generate, sampled from Kamen's
sessions exactly as SpecDecode built the eval (extract.load_session / context_before / classify: edit / write / bash /
text; render.render: the forced assistant prefix up to the bulk argument). Mixed like the 12 eval turns (4:2:3:3), at
most 2 turns per session, sessions round-robin over projects.
Excluded as in L3 (dfl2c_data.py): every session holding an assistant message of SpecDecode's 29 route4 samples or the
start of one of their targets, or a piece of the 125B's outputs on the 12 eval turns (Chat1000 agent125); and
SpecDecode's own sessions. Context CTX_CHARS 12,000 (the eval's: 28,000).
Output ~/Leap/data/dfl2d_samples.jsonl: key, project, kind, prompt (ids), split (train / val by session). Never
leaves this Mac. Usage: [BigMoE venv] dfl2d_data.py [N_SAMPLES] [CTX_CHARS]"""
import glob, json, os, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/SpecDecode"))
import extract as X                                  # read-only use of SpecDecode's loader, classifier and renderer
import render as R
from mlx_lm.tokenizer_utils import load as load_tok

N = int(sys.argv[1]) if len(sys.argv) > 1 else 110
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 12_000
# 02:12 fix, after the first run found no edits and only 55 turns: an Edit qualifies only if the text it copies is in
# the context, which 12,000 chars rarely holds, so edits get 20,000 (capped at 20 turns: prefill cost); up to 4 turns per
# session (2 allowed only ~62 in all).
CTX_EDIT, MAXPER, EDIT_CAP = 28_000, 4, 20
MIX = {"edit": 4, "write": 2, "bash": 3, "text": 3}
WANT = {k: round(N * v / sum(MIX.values())) for k, v in MIX.items()}
WANT["edit"] = min(WANT["edit"], EDIT_CAP)
OUT = os.path.expanduser("~/Leap/data/dfl2d_samples.jsonl")
tok = load_tok(Path(os.path.expanduser("~/BigMoE/models/bare")))

def payload(m):
    return (m.get("content") or "") + json.dumps(m.get("tool_calls") or "", ensure_ascii=False)

sigs, pieces = set(), []                             # the exclusions of dfl2c_data.py, unchanged
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
print(f"[l4data] {len(sigs)} assistant signatures, {len(pieces)} text pieces to exclude; want {WANT}", flush=True)

paths = [p for p in sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))) if "SpecDecode" not in p]
random.seed(1); random.shuffle(paths)
by_proj = {}
for p in paths:
    by_proj.setdefault(p.split("/")[-2], []).append(p)
order = []
while any(by_proj.values()):                         # round-robin over projects
    for proj in sorted(by_proj):
        if by_proj[proj]:
            order.append(by_proj[proj].pop())

got = {k: [] for k in MIX}
stats = {"excluded": 0, "render": 0, "sessions": 0}
for path in order:
    if all(len(got[k]) >= WANT[k] for k in MIX):
        break
    msgs = X.load_session(path)
    if len(msgs) < 6:
        continue
    blob = "\n".join(payload(m) for m in msgs)
    if any(payload(m)[:300] in sigs for m in msgs if m["role"] == "assistant") or any(p in blob for p in pieces):
        stats["excluded"] += 1
        continue
    cands = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and i >= 4:
            X.CTX_CHARS = CTX
            ctx = X.context_before(msgs, i)
            kind = X.classify(m, ctx)
            if kind is None and [c["function"]["name"] for c in m.get("tool_calls") or []] == ["Edit"]:
                X.CTX_CHARS = CTX_EDIT
                ctx = X.context_before(msgs, i)
                kind = X.classify(m, ctx)
            if kind and len(got[kind]) < WANT[kind]:
                cands.append({"kind": kind, "project": path.split("/")[-2], "messages": ctx, "target": m})
    random.shuffle(cands)
    taken = 0
    for c in cands:
        if taken == MAXPER or len(got[c["kind"]]) >= WANT[c["kind"]]:
            continue
        try:
            ids, _ = R.render(c, tok)
        except ValueError:
            stats["render"] += 1
            continue
        got[c["kind"]].append({"key": f"{c['project']}__{Path(path).stem[:8]}_{c['kind']}{taken}", "project": c["project"],
                               "kind": c["kind"], "prompt": ids,
                               "split": "val" if stats["sessions"] % 10 == 9 else "train"})
        taken += 1
    if taken:
        stats["sessions"] += 1
rows = [r for k in MIX for r in got[k]]
random.shuffle(rows)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"[l4data] {len(rows)} turns ({ {k: len(v) for k, v in got.items()} }) from {stats['sessions']} sessions in "
      f"{len(set(r['project'] for r in rows))} projects; val {sum(r['split'] == 'val' for r in rows)}; prompt tokens "
      f"{sum(len(r['prompt']) for r in rows)}; skipped {stats} -> {OUT}", flush=True)
