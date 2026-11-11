"""nocomment_bands.py (Fable FULL, LOSSY350): second cut of the HPD-0 walks for the comment-free mode.
Per tested row: is it outside the code fence (prose), a comment row, or code; its drafter cf band; first-miss hazard.
The npz files are the project's own recorded walks (TrainLab), loaded with allow_pickle for their names array only.
Run: nice -n 15 ~/BigMoE/.venv/bin/python -I nocomment_bands.py > nocomment_bands.out"""
import importlib.util, sys
import numpy as np
from tokenizers import Tokenizer
TOK = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
spec = importlib.util.spec_from_file_location("relax_core", "/Users/kamen/EvalSuite/evalsuite/relax_core.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
WALK = "/Users/kamen/TrainLab/data/d02b_walk_hpd0_%s.npz"
_d = {}
def dec(i):
    i = int(i)
    if i not in _d: _d[i] = TOK.decode([i])
    return _d[i]
BANDS = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
for dom in sys.argv[1:] or ["code", "mbpp"]:
    W = np.load(WALK % dom, allow_pickle=True); names = list(W["names"])
    kinds = ["prose", "comment", "code"]
    rows = {k: 0 for k in kinds}; miss = {k: 0 for k in kinds}; band = {k: [0] * 4 for k in kinds}
    gen = {k: 0 for k in kinds}; firstrow = {k: 0 for k in kinds}; firstmiss = {k: 0 for k in kinds}
    chk_with = 0; chk = 0; sure_all = 0; rows_all = 0
    for wi in range(len(names)):
        ids = W[f"{wi}_ids"].astype(int).tolist(); ck = W[f"{wi}_checks"]; dtok = W[f"{wi}_dtok"]; ttok = W[f"{wi}_ttok"]; dcf = W[f"{wi}_dcf"]
        sc = rc.CodeScan(); kind = []
        for i in ids:
            s = dec(i); code_b = sc.code; com_b = bool(sc.code and sc.com); anyc = False
            for ch in s:
                sc._ch(ch)
                anyc = anyc or bool(sc.code and sc.com)
            if anyc or (com_b and not s.startswith("\n")): kind.append("comment")
            elif code_b and sc.code: kind.append("code")
            elif code_b or sc.code: kind.append("code")      # the fence lines themselves
            else: kind.append("prose")
        g0 = int(ck[0, 2]) + 1 if ck.shape[0] else len(ids)
        for p in range(g0, len(ids)): gen[kind[p]] += 1
        for ci in range(ck.shape[0]):
            kept, nrows, a = int(ck[ci, 0]), int(ck[ci, 1]), int(ck[ci, 2])
            d = dtok[ci][:nrows]; t = ttok[ci][:nrows]; c = dcf[ci][:nrows]
            fm = next((j for j in range(nrows) if d[j] != t[j]), nrows)
            chk += 1; has = False
            for j in range(min(fm + 1, nrows)):
                p = a + 1 + j
                if p >= len(ids): break
                k = kind[p]; rows[k] += 1; miss[k] += int(j == fm); has = has or k == "comment"
                b = next(bi for bi, (lo, hi) in enumerate(BANDS) if lo <= float(c[j]) < hi); band[k][b] += 1
                rows_all += 1; sure_all += int(float(c[j]) >= 0.9)
            chk_with += int(has)
    tr = sum(rows.values()); tm = sum(miss.values()); tg = sum(gen.values())
    print(f"== {dom}: {chk} checks, {tr} tested rows, {tm} first misses; checks whose tested rows touch a comment {100 * chk_with / chk:.0f} %")
    for k in kinds:
        if rows[k] == 0: continue
        b = band[k]; n = sum(b)
        print(f"  {k:8s} generated {100 * gen[k] / tg:5.1f} % | tested rows {100 * rows[k] / tr:5.1f} % | hazard {miss[k] / rows[k]:.4f} | share of first misses {100 * miss[k] / tm:5.1f} % | "
              f"cf <0.5 / 0.5-0.9 / 0.9-0.99 / >=0.99: {100 * b[0] / n:.0f} / {100 * b[1] / n:.0f} / {100 * b[2] / n:.0f} / {100 * b[3] / n:.0f} %")
    sure_code = band["code"][2] + band["code"][3]
    print(f"  rows with cf >= 0.9 (the split form's strict rows): all tested rows {100 * sure_all / rows_all:.1f} %, code rows only {100 * sure_code / max(1, rows['code']):.1f} %")
