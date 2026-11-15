"""D-08 CALC information for NewIdeas (MTP Head, Wed 10-28; REPORT 'D-05 MTPHYB + D-08 CALC walk'): which calculator-
determined completion tokens of the 125B's 60 GSM8K answers does DFlash2 L1-20 miss? The detector is NewIdeas'
~/NewIdeas/scripts/d08_calc.py, its regex / norm / results copied unchanged; the completion region is fixed to the
data's meaning (dfl2d's gen = the absolute index where generation starts = the prompt length; the detector as written
takes the LAST gen tokens, so it is also run that way for comparison). DFlash2 hits / first misses come from the walk
(data/d02b_walk_math.npz: per check [accepted k, checkable m, anchor a]): tokens a+1..a+k are hits, a+k+1 is the first
miss when k < m; the first completion token is given (the prefill's), never drafted.
Usage: [BigMoE venv] python sticky/d08_calc_walk.py -> data/d08_calc_walk.txt"""
import glob, os, re
import numpy as np
from tokenizers import Tokenizer

tk = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
NUM = r"\(?\s*\d[\d,]*(?:\.\d+)?\s*\)?"
EXPR = re.compile(r"((?:" + NUM + r"\s*[-+*/]\s*)+" + NUM + r")\s*=\s*\$?\s*$")


def norm(t):
    for a, b in ((r"\times", "*"), (r"\cdot", "*"), ("×", "*"), ("÷", "/"), (r"\div", "/"), (" x ", " * "), ("{", ""), ("}", "")):
        t = t.replace(a, b)
    return t[-80:]


def results(v):
    out = []
    if abs(v - round(v)) < 1e-9:
        i = int(round(v))
        out += [str(i), f"{i:,}"]
    else:
        out.append(f"{v:.4f}".rstrip("0").rstrip("."))
        out.append(f"{v:.2f}")
    return out


def calc_positions(ids, start):
    """NewIdeas' scan from index start to the end; returns the calculator-determined token indices"""
    n, i, pos = len(ids), start, []
    while i < n:
        m = EXPR.search(norm(tk.decode(ids[start:i])))
        if not m:
            i += 1
            continue
        expr = m.group(1).replace(",", "")
        try:
            v = eval(expr, {"__builtins__": {}}, {}) if re.fullmatch(r"[\d\s.+\-*/()]+", expr) else None
        except Exception:
            v = None
        if v is None:
            i += 1
            continue
        cont = tk.decode(ids[i:i + 12]).lstrip()
        hit = next((r for r in results(float(v)) if cont.startswith(r)), None)
        if hit is None:
            i += 1
            continue
        k = 1
        while k < 12 and len(tk.decode(ids[i:i + k]).lstrip()) < len(hit):
            k += 1
        pos += range(i, i + k)
        i += k
    return pos


lines = []
def say(s):
    print(s, flush=True); lines.append(s)

W = np.load(os.path.expanduser("~/TrainLab/data/d02b_walk_math.npz"))
names = list(W["names"])
tot_asw = det_asw = tot = 0
cls = {"hit": 0, "miss": 0, "given": 0}
misses_all = misses_calc = 0
depth_of_calc_miss = []
for s, nm in enumerate(names):
    ids, off, gen, checks = W[f"{s}_ids"].tolist(), int(W[f"{s}_off"]), int(W[f"{s}_gen"]), W[f"{s}_checks"]
    z = np.load(f"/Users/kamen/Leap/data/dfl2d_math/{nm}")
    assert z["ids"].tolist() == ids and int(z["gen"]) == gen
    n = len(ids)
    tot_asw += gen                                         # the detector as written: last gen tokens, denominator gen
    det_asw += len(calc_positions(ids, n - gen))
    c0 = gen - off                                         # the completion: ids[c0:]
    tot += n - c0
    calc = set(calc_positions(ids, c0))
    hit, miss = set(), {}
    for k, m, a in checks.tolist():
        r = a - off
        hit.update(range(r + 1, r + k + 1))
        if k < m:
            miss[r + k + 1] = k + 1                        # first miss at depth k + 1
    misses_all += len(miss)
    for p in calc:
        if p in hit:
            cls["hit"] += 1
        elif p in miss:
            cls["miss"] += 1
            misses_calc += 1
            depth_of_calc_miss.append(miss[p])
        else:
            cls["given"] += 1
ncalc = sum(cls.values())
say(f"D-08 CALC x DFlash2 L1-20 walk (60 GSM8K answers of the 125B, block 16, official 1-step loop):")
say(f"  detector as written (last gen tokens, gen = prompt length): {det_asw} calculator tokens / {tot_asw} = "
    f"{100 * det_asw / tot_asw:.2f} % (NewIdeas' 3.1 % figure)")
say(f"  completion region fixed (ids[gen - off:]): {ncalc} calculator tokens / {tot} completion tokens = "
    f"{100 * ncalc / tot:.2f} %")
say(f"  of those: DFlash2 hits {cls['hit']} ({100 * cls['hit'] / max(ncalc, 1):.1f} %), first misses {cls['miss']} "
    f"({100 * cls['miss'] / max(ncalc, 1):.1f} %), given (first completion token) {cls['given']}")
say(f"  first misses on calculator tokens: {misses_calc} of all {misses_all} first misses = "
    f"{100 * misses_calc / max(misses_all, 1):.1f} %; depth of those misses: median "
    f"{np.median(depth_of_calc_miss) if depth_of_calc_miss else float('nan'):.0f}, "
    f"mean {np.mean(depth_of_calc_miss) if depth_of_calc_miss else float('nan'):.1f}")
say(f"  NewIdeas' follow-up line (>= 40 % of calculator tokens missed by DFlash2): "
    f"{'PASS' if cls['miss'] >= 0.4 * ncalc else 'FAIL'} ({100 * cls['miss'] / max(ncalc, 1):.1f} %)")
open(os.path.expanduser("~/TrainLab/data/d08_calc_walk.txt"), "w").write("\n".join(lines) + "\n")
