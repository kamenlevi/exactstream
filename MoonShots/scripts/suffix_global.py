"""#17: does a global index over *earlier sessions* add accepted lookup drafts beyond the current context?
Replays Kamen's Claude Code transcripts (~/.claude/projects/*/*.jsonl) in time order as one token stream
(Qwen tokenizer). At each assistant-output position: find the most recent earlier occurrence of the last
NG tokens; draft the K tokens that followed it; accepted = common prefix with the truth. Context-only lookup may
use matches inside the current session; global lookup may also use any earlier session.
Decode walk: a step emits accepted+1 tokens. Reports tokens per step (lookup only, no MTP) on the
evaluated (most recent) sessions. Usage: suffix_global.py [MAX_MB] [EVAL_FRAC]"""
import glob, json, os, sys
import numpy as np
from tokenizers import Tokenizer

MAX_MB = float(sys.argv[1]) if len(sys.argv) > 1 else 120
EVAL_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
NG, K, CAP = 4, 16, 20000
tk = Tokenizer.from_file("/Users/kamen/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit/tokenizer.json")


def render(rec):
    """-> list of (text, is_output) for one transcript line."""
    m = rec.get("message") or {}
    c = m.get("content")
    out = rec.get("type") == "assistant"
    if isinstance(c, str):
        return [(c[:CAP], out)]
    res = []
    for b in c or []:
        t = b.get("type")
        if t == "text":
            res.append((b.get("text", "")[:CAP], out))
        elif t == "tool_use":
            res.append((json.dumps(b.get("input", {}), ensure_ascii=False)[:CAP], out))
        elif t == "tool_result":
            cc = b.get("content")
            if isinstance(cc, list):
                cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
            res.append((str(cc)[:CAP], False))
    return res


files = sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")), key=os.path.getmtime)
picked, tot = [], 0
for f in reversed(files):                         # most recent files up to MAX_MB
    s = os.path.getsize(f) / 1e6
    if tot + s > MAX_MB and picked:
        continue
    picked.append(f); tot += s
picked = picked[::-1]
toks, isout, sess = [], [], []
for si, f in enumerate(picked):
    pieces = []
    for line in open(f, errors="ignore"):
        try:
            pieces += render(json.loads(line))
        except Exception:
            pass
    pieces = [p for p in pieces if p[0]]
    if not pieces:
        continue
    enc = tk.encode_batch([p[0] for p in pieces])
    for (txt, o), e in zip(pieces, enc):
        toks += e.ids; isout += [o] * len(e.ids); sess += [si] * len(e.ids)
S = np.array(toks, np.int64); O = np.array(isout); SE = np.array(sess)
N = len(S)
sess_start = np.zeros(SE.max() + 1, np.int64)
first = np.r_[0, np.flatnonzero(np.diff(SE)) + 1]
sess_start[SE[first]] = first
print(f"{len(picked)} sessions, {tot:.0f} MB jsonl, {N/1e6:.1f}M tokens, outputs {O.mean():.0%}", flush=True)

# hash of the NG-gram ending at each position; sorted (hash, pos) for "most recent earlier occurrence"
h = np.zeros(N, np.uint64)
for j in range(NG):
    h = h * np.uint64(1000003) + np.roll(S, j).astype(np.uint64)
h[:NG] = 0
order = np.lexsort((np.arange(N), h))
hs, ps = h[order], order


def prev_occ(i):
    """most recent position p < i with the same NG-gram hash ending at p, or -1."""
    lo = np.searchsorted(hs, h[i], "left")
    hi = np.searchsorted(hs, h[i], "right")
    k = lo + np.searchsorted(ps[lo:hi], i, "left") - 1
    return ps[k] if k >= lo else -1


eval_from = sess_start[int(SE.max() * (1 - EVAL_FRAC))]
stats = {"none": 0, "ctx": 0, "glob": 0}
steps = {"none": 0, "ctx": 0, "glob": 0}
ntok = 0
i = eval_from
outpos = np.flatnonzero(O[eval_from:]) + eval_from
# walk each policy over the output tokens
for pol in ("none", "ctx", "glob"):
    i_idx = 0
    st = 0
    while i_idx < len(outpos):
        i = outpos[i_idx]
        acc = 0
        if pol != "none" and i >= NG:
            p = prev_occ(i)
            if p >= 0 and (pol == "glob" or p >= sess_start[SE[i]]):
                d = S[p + 1:p + 1 + K]; t = S[i + 1:i + 1 + K]
                m = min(len(d), len(t))
                neq = np.flatnonzero(d[:m] != t[:m])
                acc = int(neq[0]) if len(neq) else m
                acc = min(acc, int(np.argmin(O[i + 1:i + 1 + K])) if not O[i + 1:i + 1 + K].all() else acc)
        st += 1
        i_idx += acc + 1
    steps[pol] = st
n = len(outpos)
print(f"evaluated {n/1e3:.0f}K output tokens in the latest {EVAL_FRAC:.0%} of sessions (lookup only, K={K}, {NG}-gram)")
for pol in ("none", "ctx", "glob"):
    print(f"  {pol:5s}: {n/steps[pol]:.3f} tokens/step")
print(f"global over context-only: {steps['ctx']/steps['glob']:.3f}x fewer steps")
