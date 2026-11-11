# LUNATIC round 2, Route P kill test: is the 125B's continuation robust to a perturbed prefix?
# Two exact references (REF2-T vs REF2-CT) decode the same 222 prompts; where they diverge at a near-tie, each text
# continues from a different prefix. Use B's continuation as the draft source for A's continuation (suffix-match
# lookup, like prompt-lookup / SAM drafting) and measure the acceptance A would get: rho for Parareal decoding.
import json, sys
from collections import defaultdict

def load(p):
    d = {}
    for ln in open(p):
        r = json.loads(ln)
        d[r['key']] = (r['set'], r['completion_ids'])
    return d

A = load(sys.argv[1]); B = load(sys.argv[2])
keys = sorted(set(A) & set(B))

def sim_draft(a, b, start, nmax=64, ngram=(8, 4, 2, 1)):
    """walk a[start:]; at each step draft up to nmax tokens of b following the longest n-gram suffix match of the
    emitted text (search b near the aligned offset first). returns (accepted draft tokens, emitted tokens, checks)."""
    # index b by n-gram -> positions
    idx = {n: defaultdict(list) for n in ngram}
    for n in ngram:
        for j in range(len(b) - n + 1):
            idx[n][tuple(b[j:j + n])].append(j + n)
    i, acc, checks, last = start, 0, 0, None
    while i < len(a):
        checks += 1
        draft = []
        for n in ngram:
            if i - n < 0:
                continue
            pos = idx[n].get(tuple(a[i - n:i]))
            if pos:
                # prefer the match nearest the previous alignment (monotone walk), else the first
                p = min(pos, key=lambda q: abs(q - (last if last is not None else i)))
                draft = b[p:p + nmax]
                last = p
                break
        m = 0
        while m < len(draft) and i + m < len(a) and draft[m] == a[i + m]:
            m += 1
        acc += m
        i += m + 1          # accepted drafts + the 125B's own token
        if last is not None:
            last += m + 1
    return acc, len(a) - start, checks

agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # set -> [pairs, diverged, acc, emitted, checks, prefix_tokens]
for k in keys:
    s, a = A[k]; _, b = B[k]
    g = agg[s]; g[0] += 1
    m = 0
    while m < min(len(a), len(b)) and a[m] == b[m]:
        m += 1
    if m == len(a) == len(b):
        continue
    g[1] += 1; g[5] += m
    acc, em, ch = sim_draft(a, b, m + 1)   # after the first divergent token (A's token at m is the 125B's own)
    g[2] += acc; g[3] += em; g[4] += ch

print("pairs: REF2-T vs REF2-CT, same prompts; continuation after the first divergence, drafted from the other text")
tot = [0] * 6
for s in sorted(agg):
    g = agg[s]; tot = [x + y for x, y in zip(tot, g)]
    rho = g[2] / g[3] if g[3] else float('nan')
    kpc = g[3] / g[4] if g[4] else float('nan')
    print(f"{s:10s} pairs {g[0]:3d} diverged {g[1]:3d} (mean shared prefix {g[5] / max(1, g[1]):6.1f} tok) | "
          f"tokens after divergence {g[3]:6d} | drafted-and-accepted {rho * 100:5.1f} % | emitted per check {kpc:5.1f}")
rho = tot[2] / tot[3]; kpc = tot[3] / tot[4]
print(f"{'ALL':10s} pairs {tot[0]:3d} diverged {tot[1]:3d} | tokens after divergence {tot[3]:6d} | "
      f"drafted-and-accepted {rho * 100:5.1f} % | emitted per check {kpc:5.1f}")
