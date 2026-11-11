"""Fit12 (Thu 5 Nov 19:5x; 12 GB plan lever 1, the exact-pass diet): where T3R2's t80ck1u time goes by check mode, from the
run's own logs (/Users/kamen/Fit12/data/scr/t3r2/r47_*_t80ck1u{,_ph}.json; 16 HumanEval prompts, Wed 4 Nov quiet lock).
Per check record: [rows, mode, kept, cf]; mode 0 = the cf gate sent the check straight to the exact check (first draft's
cf < 0.9), 1 = pass 1 certified and committed, 2 = pass 1 + pass 2 (exact) from the first uncertified row. Per phase:
[name, check, t0, a, bytes, misses, t1, a, bytes, misses] (cumulative counters). CPU, seconds. Usage: python -I -B t3_exact_split.py"""
import glob, json, os, collections
D = "/Users/kamen/Fit12/data/scr/t3r2"
T = collections.defaultdict(float); K = collections.Counter(); N = collections.Counter(); B = collections.defaultdict(float)
P = collections.defaultdict(float); cfs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(D, "r47_humaneval_*_t80ck1u.json"))):
    recs = json.load(open(f))
    ph = json.load(open(f[:-5] + "_ph.json"))
    per = collections.defaultdict(lambda: collections.defaultdict(float)); byt = collections.defaultdict(float)
    for p in ph:
        name, c, t0, b0, t1, b1 = p[0], p[1], p[2], p[4], p[6], p[8]
        per[c][name] += t1 - t0
        byt[c] += b1 - b0
    for c, (rows, mode, kept, cf) in enumerate(recs):
        N[mode] += 1; K[mode] += kept; B[mode] += byt[c]; cfs[mode].append(cf)
        for name, dt in per[c].items():
            T[mode] += dt; P[(mode, name)] += dt
tot_t, tot_k = sum(T.values()), sum(K.values())
print(f"16 prompts, {sum(N.values())} checks, {tot_k} kept tokens, {tot_t:.2f} s in phases -> {tot_k / tot_t:.1f} tok/s (phases only)")
for m, lab in ((0, "cf gate -> exact check"), (1, "pass 1 certified"), (2, "pass 1 + pass 2 (exact)")):
    if not N[m]:
        continue
    parts = ", ".join(f"{n} {1e3 * P[(m, n)] / N[m]:.0f}" for n in ("draft", "p1", "commit", "exact") if P[(m, n)])
    print(f"mode {m} {lab:25s}: {N[m]:3d} checks ({100 * N[m] / sum(N.values()):4.1f} %), kept {K[m] / N[m]:5.2f} per check, "
          f"{1e3 * T[m] / N[m]:5.0f} ms per check ({parts}), {B[m] / N[m] / 1e6:5.0f} MB per check; "
          f"{100 * T[m] / tot_t:4.1f} % of time, {100 * K[m] / tot_k:4.1f} % of tokens; cf median "
          f"{sorted(cfs[m])[len(cfs[m]) // 2]:.3f}")
