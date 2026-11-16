"""LIM800 SLACK (Lunatic, THINK_1116b, Mon 16 Nov 18:0x). CPU only, read-only, one process, < 0.5 GB.
Where is the slack in LOSSY kept per 32-row check? Chain-32 walk (certify98.kept_walk, cap 31) on BigMoE's t30b cells +
TOPK dumps (HE+ heplus24, MBPP mbppbank), the same data and walk as NewIdeas' margin_ceilings / tree_cc32.
Per cell: right (head top-1 == the 125B's text token), m (125B top1 - top2), dlp (125B logp(drafted) - logp(top1)),
the drafted id (TOPK top_ids[0]) and the text token y. Each predicted token gets a lexer state from the decoded answer
text: PROSE (outside ``` fences), COMMENT (# to end of line), DOC (triple-quoted), STR (one-line string), CODE.
Rules (a cell is accepted if right, or if the rule accepts the miss; after an accepted miss the walk follows the text's
cells, the same approximation as the existing close-call walks):
  exact; CC delta (dlp >= -delta); INERT (any miss in PROSE / COMMENT / DOC); WS (code misses that differ only by
  intra-line whitespace); and ceilings that FIX code misses with m >= T (what a drift fix would have to buy).
usage: ~/BigMoE/.venv/bin/python3 -I lim800_slack.py"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98
from tokenizers import Tokenizer

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
TOK = Tokenizer.from_file(f"{H}/BigMoE/models/bare/tokenizer.json")
DEP = 31
OUT = f"{H}/FableGod/lunatic/lim800_slack.out"
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

_cache = {}
def txt(i):
    i = int(i)
    if i not in _cache:
        _cache[i] = TOK.decode([i], skip_special_tokens=False)
    return _cache[i]

def lex_states(ids):
    """state at the START of each token: 0 CODE, 1 COMMENT, 2 DOC, 3 STR, 4 PROSE"""
    full = "".join(txt(t) for t in ids)
    start_prose = "```" in full
    st = 4 if start_prose else 0
    q = None; out = []; line_start = True; i_char = 0
    for t in ids:
        out.append(st)
        s = txt(t); k = 0
        while k < len(s):
            ch = s[k]
            if st == 4:
                if line_start and s.startswith("```", k):
                    st = 0; k += 3
                    while k < len(s) and s[k] != "\n": k += 1   # skip the fence's language tag
                    continue
            elif st == 0:
                if line_start and s.startswith("```", k):
                    st = 4; k += 3; continue
                if ch == "#": st = 1
                elif s.startswith('"""', k) or s.startswith("'''", k):
                    st = 2; q = s[k:k+3]; k += 3; line_start = False; continue
                elif ch in "\"'":
                    st = 3; q = ch
            elif st == 1:
                if ch == "\n": st = 0
            elif st == 2:
                if s.startswith(q, k):
                    st = 0; k += 3; line_start = False; continue
            elif st == 3:
                if ch == "\\": k += 2; continue
                if ch == q or ch == "\n": st = 0
            line_start = (ch == "\n") or (line_start and ch in " \t")
            k += 1
    return np.array(out, np.int8), full

def load(tag):
    cz, tz = np.load(f"{CE}/cells_t30b_{tag}.npz"), np.load(f"{CE}/topk_t30b_{tag}.npz")
    for k in ("depth", "fi", "row", "right"):
        assert (cz[k] == tz[k]).all(), f"{tag}: {k} differs"
    dep = tz["depth"].astype(int); right = tz["right"].astype(int)
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    ref = C98.windows(f"{CE}/cells_t30b_{tag}.npz")
    W = []
    for wi, (s, t) in enumerate(zip(st, en)):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        A = {k: np.full((n0 + 1, DEP), fill, dt) for k, fill, dt in
             (("rt", -1, np.int8), ("M", np.nan, np.float32), ("DL", np.nan, np.float32), ("DR", -1, np.int64),
              ("Y", -1, np.int64))}
        for j in range(DEP):
            msk = dw == j; c = int(msk.sum())
            A["rt"][:c, j] = right[s:t][msk]; A["M"][:c, j] = cz["m"][s:t][msk]; A["DL"][:c, j] = cz["dlp"][s:t][msk]
            A["DR"][:c, j] = tz["top_ids"][s:t][msk][:, 0]; A["Y"][:c, j] = tz["y"][s:t][msk]
        assert (A["rt"] == ref[wi][0]).all()
        y0 = A["Y"][:n0, 0]
        for j in range(1, DEP):                      # depth j from anchor i predicts y0[i + j]
            c = int((A["Y"][:, j] >= 0).sum())
            assert (A["Y"][:c, j] == y0[j:j + c]).all(), f"{tag} w{wi}: depth {j} y != y0 shift"
        S0, full = lex_states(y0)
        S = np.full((n0 + 1, DEP), -1, np.int8)
        for j in range(DEP):
            c = int((A["Y"][:, j] >= 0).sum()); S[:c, j] = S0[j:j + c]
        A["S"] = S; A["full"] = full; A["n0"] = n0
        W.append(A)
    return W

def walk(W, acc_fn):
    ks = []
    for A in W:
        r = A["rt"].copy(); miss = r == 0
        r[miss & acc_fn(A)] = 1
        ks += C98.kept_walk(r, C98.R0)
    return float(np.mean(ks)), len(ks)

def code_ws(A):
    out = np.zeros(A["rt"].shape, bool)
    for i, j in zip(*np.nonzero(A["rt"] == 0)):
        a, b = txt(A["DR"][i, j]), txt(A["Y"][i, j])
        out[i, j] = ("\n" not in a + b) and a != b and a.strip() == b.strip()
    return out

def miss_table(W, acc_fn, label):
    """first-miss cells on the walk (the misses that END checks), after rule acc_fn, by state / class / m bin"""
    rows = []
    for A in W:
        r = A["rt"].copy(); r[(r == 0) & acc_fn(A)] = 1
        sub = r[C98.R0:]; p = 0
        while p < sub.shape[0]:
            k = 0
            while k < 31 and sub[p, k] == 1: k += 1
            if k < 31 and sub[p, k] == 0:
                i = p + C98.R0
                a, b = txt(A["DR"][i, k]), txt(A["Y"][i, k])
                cls = ("nl" if "\n" in a + b else "ident" if b.strip().replace("_", "").isalnum() and not b.strip().isdigit()
                       else "num" if b.strip().isdigit() else "punct")
                rows.append((int(A["S"][i, k]), cls, float(A["M"][i, k]), k))
            p += k + 1
    say(f"   check-ending misses after {label}: n {len(rows)}")
    names = {0: "CODE", 1: "COMMENT", 2: "DOC", 3: "STR", 4: "PROSE"}
    for sv in sorted(set(r[0] for r in rows)):
        rr = [r for r in rows if r[0] == sv]
        say(f"     {names[sv]:8s} {100*len(rr)/max(1,len(rows)):5.1f} %")
    cr = [r for r in rows if r[0] == 0]
    for cls in ("nl", "ident", "num", "punct"):
        rr = [r for r in cr if r[1] == cls]
        if not rr: continue
        mm = np.array([r[2] for r in rr])
        say(f"       CODE/{cls:5s} {100*len(rr)/max(1,len(rows)):5.1f} %  m<1 {100*np.mean(mm<1):4.0f} %  1-2 {100*np.mean((mm>=1)&(mm<2)):4.0f} %"
            f"  2-4 {100*np.mean((mm>=2)&(mm<4)):4.0f} %  >=4 {100*np.mean(mm>=4):4.0f} %  depth med {np.median([r[3] for r in rr]):.0f}")
    return rows

RES = {}
for tag in ("heplus24", "mbppbank"):
    W = load(tag)
    say(f"== {tag}: {len(W)} windows, {sum(A['n0'] for A in W)} answer tokens")
    st_all = np.concatenate([A["S"][:A["n0"], 0] for A in W])
    say("   token states: " + ", ".join(f"{n} {100*np.mean(st_all==v):.1f} %" for v, n in
                                         ((0, "CODE"), (1, "COMMENT"), (2, "DOC"), (3, "STR"), (4, "PROSE"))))
    inert = lambda A: np.isin(A["S"], (1, 2, 4))
    cc = lambda d: (lambda A: np.nan_to_num(A["DL"], nan=-99) >= -d)
    R = {}
    R["exact"] = walk(W, lambda A: np.zeros(A["rt"].shape, bool))
    for d in (0.5, 1, 2, 3, 4, 6, 99):
        R[f"cc{d:g}"] = walk(W, cc(d))
    R["inert"] = walk(W, inert)
    R["cc1+inert"] = walk(W, lambda A: cc(1)(A) | inert(A))
    R["cc2+inert"] = walk(W, lambda A: cc(2)(A) | inert(A))
    R["cc1+inert+ws"] = walk(W, lambda A: cc(1)(A) | inert(A) | code_ws(A))
    R["cc1+inert+str"] = walk(W, lambda A: cc(1)(A) | inert(A) | (A["S"] == 3))
    for T in (8, 4, 2, 1, 0):
        R[f"cc1+inert+fix_code_m>={T}"] = walk(W, lambda A, T=T: cc(1)(A) | inert(A) |
                                                ((A["S"] == 0) & (np.nan_to_num(A["M"], nan=0) >= T)))
    R["all_fixed"] = walk(W, lambda A: np.ones(A["rt"].shape, bool))
    for k, (v, n) in R.items():
        say(f"   {k:28s} kept/32 {v:6.2f}  x{v / R['exact'][0]:5.3f} vs exact  ({n} checks)")
    miss_table(W, lambda A: cc(1)(A) | inert(A), "cc1 + inert")
    RES[tag] = {k: v[0] for k, v in R.items()}
    say()
json.dump(RES, open(OUT.replace(".out", ".json"), "w"), indent=1)
open(OUT, "w").write("\n".join(lines) + "\n")
