"""Fit12 (Tue 3 Nov 14:5x): the SHARED-TEXT speed of two T3 arms (CPU only). Free-running non-exact arms diverge, and
after that each decodes its own text (one stopped at EOS on HumanEval/91 at token 177, skipping 80 hard tokens), so
summed tokens / summed time compares different texts. Here, per prompt, the cut = the first token where the two arms'
outputs differ (from {TAG}_{key}_{arm}_toks.json; without them, the smaller of their 'first divergence' from ex in the
run's text output). For each arm: its checks until its committed tokens reach the cut, timed by the sum of its phase
durations ({TAG}_{key}_{arm}_ph.json); tok/s = those tokens / that time, summed over prompts.
Usage: t3_shared.py DIR TAG ARM_A ARM_B [ARM ...]  (ratios vs ARM_A; the cut is A vs each other arm)"""
import glob, json, os, re, sys

D, TAG, A = sys.argv[1], sys.argv[2], sys.argv[3]
OTHERS = sys.argv[4:]
txt = open(os.path.join(D, TAG + ".txt")).read()
keys = sorted({os.path.basename(f)[len(TAG) + 1:-len(f"_{A}_ph.json")] for f in glob.glob(os.path.join(D, f"{TAG}_*_{A}_ph.json"))})


def div_from_ex(key, arm):
    m = re.search(re.escape(key.replace("_", "/", 2)) + rf" {arm} theta .*?first divergence (\w+)\)", txt)
    return None if m is None or m.group(1) == "None" else int(m.group(1))


def cut(key, a, b):
    fa, fb = (os.path.join(D, f"{TAG}_{key}_{x}_toks.json") for x in (a, b))
    if os.path.exists(fa) and os.path.exists(fb):
        ta, tb = json.load(open(fa)), json.load(open(fb))
        return next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    da, db = div_from_ex(key, a), div_from_ex(key, b)
    return min(v for v in (da, db, 10 ** 9) if v is not None)


def timed(key, arm, c):
    L = json.load(open(os.path.join(D, f"{TAG}_{key}_{arm}.json")))
    ph = json.load(open(os.path.join(D, f"{TAG}_{key}_{arm}_ph.json")))
    tok = n = 0
    for S, path, k, cf in L:
        if tok >= c:
            break
        tok += k; n += 1
    return tok, sum(p[6] - p[2] for p in ph if p[1] < n), n


for B in OTHERS:
    ta = tb = sa = sb = na = nb = 0
    per = []
    for key in keys:
        c = cut(key, A, B)
        a, b = timed(key, A, c), timed(key, B, c)
        ta += a[0]; sa += a[1]; na += a[2]; tb += b[0]; sb += b[1]; nb += b[2]
        per.append(f"{key.split('_')[-1]} cut {c}: x{(b[0] / b[1]) / (a[0] / a[1]):.2f}")
    print(f"{TAG} shared text {B} vs {A}: {A} {ta} tok / {sa:.2f} s = {ta / sa:.1f} tok/s (kept {ta / max(1, na):.2f}) | {B} {tb} tok / "
          f"{sb:.2f} s = {tb / sb:.1f} tok/s (kept {tb / max(1, nb):.2f}) | x{(tb / sb) / (ta / sa):.3f} | " + ", ".join(per))
