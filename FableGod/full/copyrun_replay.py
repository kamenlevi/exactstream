"""copyrun_replay.py (Fable FULL, LOSSY350): how much of a code answer is a verbatim copy of the prompt, in how long runs,
and what the recorded walk spends on it.
Per window: prompt = ids[:g0], answer = ids[g0:] (g0 = the first check's anchor + 1). A copy run = a maximal stretch of the
answer equal, token for token, to a contiguous stretch of the prompt (greedy longest match, runs >= 8 tokens kept).
The first TRIG tokens of a run are not usable (a k-gram lookup needs them to fire); the rest is the usable run.
Per region (kind from relax_core.CodeScan: docstring / comment / code): tokens, tested rows, first-miss hazard.
Checks: the walk's own checks, split by where their anchor sits; and the checks a width-W copy block would need.
The npz is the project's own recorded walk (allow_pickle only for its names array). CPU, seconds.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I copyrun_replay.py > copyrun_replay.out"""
import importlib.util, math, sys
import numpy as np
from tokenizers import Tokenizer
TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", "/Users/kamen/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
_d = {}
def dec(i):
    i = int(i)
    if i not in _d: _d[i] = TOK.decode([i])
    return _d[i]
TRIG = 8
for dom in sys.argv[1:] or ["code", "mbpp"]:
    W = np.load("/Users/kamen/TrainLab/data/d02b_walk_hpd0_%s.npz" % dom, allow_pickle=True)
    gen = 0; runs = []; cov = {8: 0, 16: 0, 32: 0, 64: 0, 128: 0}
    kind_tok = {}; kind_copy = {}; rows = {"copy": [0, 0], "noncopy": [0, 0]}
    chk_total = 0; chk_copy = 0; tok_by_anchor = {"copy": 0, "noncopy": 0}; chk_by_anchor = {"copy": 0, "noncopy": 0}
    need = {16: 0.0, 32: 0.0, 64: 0.0, 128: 0.0}; usable = 0; per_answer = []
    for wi in range(len(W["names"])):
        ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]; dt = W[f"{wi}_dtok"]; tt = W[f"{wi}_ttok"]
        if ck.shape[0] == 0: continue
        g0 = int(ck[0, 2]) + 1; P = ids[:g0]; A = ids[g0:]
        pos = {}
        for i, t in enumerate(P): pos.setdefault(t, []).append(i)
        incopy = [0] * len(A); usable_m = [False] * len(A); q = 0; my = []
        while q < len(A):
            best = 0
            for i in pos.get(A[q], ()):
                L = 0
                while q + L < len(A) and i + L < len(P) and A[q + L] == P[i + L]: L += 1
                best = max(best, L)
            if best >= 8:
                for x in range(q, q + best): incopy[x] = best
                for x in range(q + TRIG, q + best): usable_m[x] = True
                runs.append(best); my.append(best); q += best
            else: q += 1
        gen += len(A); per_answer.append((len(A), sum(my), len(my)))
        for th in cov: cov[th] += sum(1 for x in incopy if x >= th)
        usable += sum(usable_m)
        for L in my:
            for Wd in need: need[Wd] += math.ceil(max(0, L - TRIG) / (Wd - 1)) if L > TRIG else 0
        sc = rc.CodeScan(); kind = []
        for s in (dec(i) for i in ids):
            bc = bool(sc.code and sc.com); k0 = "docstring" if (sc.code and sc.doc) else ("string" if (sc.code and (sc.str_ or sc.qn)) else ("code" if sc.code else "prose")); anyc = False
            for ch in s:
                sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
            kind.append("comment" if (anyc or bc) else k0)
        for x in range(len(A)):
            k = kind[g0 + x]; kind_tok[k] = kind_tok.get(k, 0) + 1
            if incopy[x]: kind_copy[k] = kind_copy.get(k, 0) + 1
        for ci in range(ck.shape[0]):
            k, n, a = map(int, ck[ci]); d = dt[ci]; t = tt[ci]
            fm = next((j for j in range(n) if d[j] != t[j]), n)
            first = a + 1 - g0
            where = "copy" if (0 <= first < len(A) and usable_m[first]) else "noncopy"
            chk_by_anchor[where] += 1; tok_by_anchor[where] += k + 1
            for j in range(min(fm + 1, n)):
                x = a + 1 + j - g0
                if not (0 <= x < len(A)): break
                r = "copy" if usable_m[x] else "noncopy"
                rows[r][0] += 1; rows[r][1] += int(j == fm)
    runs = np.array(runs)
    print(f"== {dom}: {len(per_answer)} answers, {gen} generated tokens; copy runs >= 8 tokens: {len(runs)} ({len(runs) / len(per_answer):.1f} per answer), "
          f"median length {np.median(runs) if len(runs) else 0:.0f}, mean {runs.mean() if len(runs) else 0:.0f}, max {runs.max() if len(runs) else 0}")
    print("  generated tokens inside a verbatim prompt-copy run of length >= 8 / 16 / 32 / 64 / 128: " + " / ".join(f"{100 * cov[t] / gen:.1f} %" for t in (8, 16, 32, 64, 128)))
    print(f"  usable copy tokens (after the {TRIG}-token trigger): {usable} = {100 * usable / gen:.1f} % of generated tokens")
    print("  by region: " + "; ".join(f"{k} {100 * v / gen:.1f} % of tokens, {100 * kind_copy.get(k, 0) / v:.0f} % of them copied" for k, v in sorted(kind_tok.items(), key=lambda x: -x[1])))
    for r in ("copy", "noncopy"):
        n, m = rows[r]
        print(f"  tested rows on {r:7s} tokens: {n:6d}, first-miss hazard {m / max(1, n):.4f} ({m} misses)")
    tc = chk_by_anchor["copy"] + chk_by_anchor["noncopy"]
    print(f"  the walk's checks: {tc}; starting on a usable copy token {chk_by_anchor['copy']} ({100 * chk_by_anchor['copy'] / tc:.0f} %), keeping {tok_by_anchor['copy'] / max(1, chk_by_anchor['copy']):.2f} tokens each; "
          f"the others {chk_by_anchor['noncopy']}, keeping {tok_by_anchor['noncopy'] / max(1, chk_by_anchor['noncopy']):.2f}")
    print("  checks a copy block of width W needs for all usable copy tokens (W - 1 copied rows + the bonus): " + "; ".join(f"W {w}: {need[w]:.0f}" for w in (16, 32, 64, 128)))
