"""nocomment_alt.py (Fable FULL, LOSSY350): where the 125B opens a '#' comment, what is its best NON-comment token,
how far below the comment it sits, and is it the token the recorded text resumes with after the comment?
Data [K]: the 125B's top-8 per position on the 12 map windows (decision_map/out/baseline.json: {name: {a, tk, lp}},
row r predicts position a + r + 1), the HPD-0 code walk for the windows' ids, the tokenizer, relax_core.CodeScan.
The npz is the project's own recorded walk (allow_pickle only for its names array). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I nocomment_alt.py > nocomment_alt.out"""
import importlib.util, json, math
import numpy as np
from tokenizers import Tokenizer
TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", "/Users/kamen/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
B = json.load(open("/Users/kamen/FableGod/decision_map/out/baseline.json"))
W = np.load("/Users/kamen/TrainLab/data/d02b_walk_hpd0_code.npz", allow_pickle=True)
_d = {}
def dec(i):
    i = int(i)
    if i not in _d: _d[i] = TOK.decode([i])
    return _d[i]
n_open = n_top1 = n_alt = n_resume = n_line = n_inline = res_line = res_inline = 0
gaps = []; ranks = []; ex = []
for wi, nm in enumerate(W["names"]):
    key = str(nm)[:-4] if str(nm).endswith(".npz") else str(nm)
    b = B.get(key) or B.get(str(nm))
    if not isinstance(b, dict) or "tk" not in b: continue
    row0, TK, LP = int(b["a"]), b["tk"], b["lp"]
    ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]
    texts = [dec(i) for i in ids]
    sc = rc.CodeScan(); flag = []; pre_code = []
    for s in texts:
        before = bool(sc.code and sc.com); pre_code.append(bool(sc.code and not sc.com and not sc.str_ and not sc.doc and not sc.qn)); anyc = False
        for ch in s:
            sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
        flag.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
    g0 = int(ck[0, 2]) + 1
    for q in range(max(g0, 1), len(ids)):
        if not (flag[q] == 1 and flag[q - 1] != 1 and pre_code[q]): continue
        r = (q - 1) - row0
        if not (0 <= r < len(TK)): continue
        tk, lp = TK[r], LP[r]
        n_open += 1; n_top1 += int(tk[0] == ids[q])
        # comment-only line? only blanks between the last newline and the '#'
        j = q - 1
        while j >= 0 and texts[j].strip() == "" and "\n" not in texts[j]: j -= 1
        only = (j < 0 or "\n" in texts[j]) and texts[q].split("#", 1)[0].strip() == ""
        e = q
        while e < len(ids) and flag[e] == 1: e += 1          # e = the token that ends the comment (flag 2) or the end
        if e >= len(ids): continue
        # the token the recorded text resumes with: for a comment-only line, the first non-blank token after the
        # comment's newline (the indent before the '#' is already written); for an inline comment, the newline itself
        f = e + 1
        while only and f < len(ids) and (texts[f].strip() == "" or flag[f] != 0): f += 1   # skip blanks and further comment lines
        if only and f >= len(ids): continue
        resume = ids[f] if only else ids[e]
        alt = next((i for i in range(len(tk)) if "#" not in dec(tk[i])), None)
        if alt is None: continue
        n_alt += 1; gaps.append(lp[alt] - lp[0]); ranks.append(alt)
        hit = int(tk[alt] == resume); n_resume += hit
        if only: n_line += 1; res_line += hit
        else: n_inline += 1; res_inline += hit
        if len(ex) < 6: ex.append((repr(texts[q]), repr(dec(tk[alt])), repr(dec(resume)), round(lp[alt] - lp[0], 2), only))
g = np.array(gaps)
print(f"comment openers in the generated text of the 12 map windows (pre-state: code, no string): {n_open}; the 125B's top-1 is the opener on {n_top1}")
print(f"with a non-comment token in the top-8: {n_alt} ({100 * n_alt / max(1, n_open):.0f} %); its rank: " + ", ".join(f"{k}:{ranks.count(k)}" for k in sorted(set(ranks))))
print(f"log-prob gap to the comment (nats): median {np.median(g):.2f}, quartiles {np.percentile(g, 25):.2f} / {np.percentile(g, 75):.2f}; within 1 nat {100 * (g >= -1).mean():.0f} %, within 2.3 nats (p >= 0.1 p1) {100 * (g >= -2.303).mean():.0f} %, below 4.6 nats (p < 0.01 p1) {100 * (g < -4.605).mean():.0f} %")
print(f"best non-comment token == the token the recorded text resumes with after the comment: {n_resume} / {n_alt} = {100 * n_resume / max(1, n_alt):.0f} % "
      f"(comment-only lines {res_line} / {n_line}; inline comments {res_inline} / {n_inline})")
for x in ex: print("  example: opener %s | best non-comment %s | resume %s | gap %s | comment-only line %s" % x)
