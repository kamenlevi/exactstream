"""nocomment_chain.py (Fable FULL, LOSSY350): the comment-free kept in ONE convention (first-miss depth over 15 rows).
Per depth j (0-based, 15 rows per check): the first-miss hazard over all tested rows and over code rows only (rows whose
true token is not a comment token and not a comment-ending newline; code right after a comment line is included).
chain(h) = sum over depths of the survival = expected rows before the first miss. The all-row chain must reproduce the
measured mean first-miss depth. The npz is the project's own recorded walk (allow_pickle only for its names array).
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I nocomment_chain.py > nocomment_chain.out"""
import importlib.util, sys
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
def chain(h):
    s = 0.0; surv = 1.0
    for x in h:
        surv *= 1.0 - x; s += surv
    return s
for dom in sys.argv[1:] or ["code", "mbpp"]:
    W = np.load("/Users/kamen/TrainLab/data/d02b_walk_hpd0_%s.npz" % dom, allow_pickle=True)
    R = 15
    n_all = np.zeros(R); m_all = np.zeros(R); n_code = np.zeros(R); m_code = np.zeros(R); n_oth = np.zeros(R); m_oth = np.zeros(R)
    fms = []; fms_nc = []; fms_pure = []
    for wi in range(len(W["names"])):
        ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]; dt = W[f"{wi}_dtok"]; tt = W[f"{wi}_ttok"]
        sc = rc.CodeScan(); flag = []; incom = []
        for i in ids:
            s = dec(i); before = bool(sc.code and sc.com); incom.append(before); anyc = False
            for ch in s:
                sc._ch(ch); anyc = anyc or bool(sc.code and sc.com)
            flag.append(1 if anyc else (2 if before and s.startswith("\n") else (1 if before else 0)))
        after = [False] * len(ids); live = False
        for p in range(len(ids)):
            if flag[p] == 2: live = True; continue
            if live:
                if flag[p] == 0 and "\n" in dec(ids[p]): live = False
                elif flag[p] == 0: after[p] = True
        for ci in range(ck.shape[0]):
            k, n, a = map(int, ck[ci])
            if n < R: continue
            d = dt[ci]; t = tt[ci]
            fm = next((j for j in range(R) if d[j] != t[j]), R); fms.append(fm)
            fl = [flag[a + 1 + j] if a + 1 + j < len(ids) else 0 for j in range(R)]
            if not (a + 1 < len(ids) and incom[a + 1]): fms_nc.append(fm)
            if not any(fl) and not (a + 1 < len(ids) and incom[a + 1]): fms_pure.append(fm)
            for j in range(min(fm + 1, R)):
                p = a + 1 + j
                if p >= len(ids): break
                m = int(j == fm); n_all[j] += 1; m_all[j] += m
                if flag[p] == 0:
                    n_code[j] += 1; m_code[j] += m
                    if not after[p]: n_oth[j] += 1; m_oth[j] += m
    h_all = m_all / np.maximum(n_all, 1); h_code = m_code / np.maximum(n_code, 1); h_oth = m_oth / np.maximum(n_oth, 1)
    ca, cc, co = chain(h_all), chain(h_code), chain(h_oth)
    print(f"== {dom}: {len(fms)} full-width checks; mean first-miss depth {np.mean(fms):.2f} (all-row chain {ca:.2f}); anchors not in a comment {np.mean(fms_nc):.2f} (n {len(fms_nc)}); pure-code checks {np.mean(fms_pure):.2f} (n {len(fms_pure)})")
    print("  hazard by depth, all rows : " + " ".join(f"{x:.3f}" for x in h_all))
    print("  hazard by depth, code rows: " + " ".join(f"{x:.3f}" for x in h_code))
    print(f"  chain on code-row hazards (comment rows gone, the hard lines after comments kept at their own hazard): {cc:.2f} = x{cc / ca:.3f} (+1 bonus: x{(cc + 1) / (ca + 1):.3f})")
    print(f"  chain on 'other code' hazards only (upper end, as if step-start lines were ordinary): {co:.2f} = x{co / ca:.3f}")
    # pessimistic: every code row right after a comment line takes the comment rows' own hazard at that depth (the step choice moves there)
    n_com = n_all - n_code; m_com = m_all - m_code; h_com = m_com / np.maximum(n_com, 1)
    n_aft = n_code - n_oth
    h_pess = (m_oth + n_aft * np.maximum(h_com, (m_code - m_oth) / np.maximum(n_aft, 1))) / np.maximum(n_code, 1)
    cp = chain(h_pess)
    print(f"  pessimistic chain (code rows right after a comment take the comment rows' hazard): {cp:.2f} = x{cp / ca:.3f} (+1 bonus: x{(cp + 1) / (ca + 1):.3f})")
