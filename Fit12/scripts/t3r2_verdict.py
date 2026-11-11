"""Fit12 T3R2 read-out (NewIdeas' pre-registration, QUEUE 4dbe21a; PLAN.md 'T3R2', written before the run). CPU only.
Usage: t3r2_verdict.py [DIR (/Users/kamen/Fit12/data/scr/t3r2)] [TAG (r47)]
Verdict: the POOLED shared-text ratio t80ck1u / t80ck2 over every prompt that counts (16 unless a prompt was paused twice);
per prompt the cut = the first token where the two arms' outputs differ; each arm's checks until its committed tokens reach the
cut, timed by the sum of its phase durations (t3_shared.py's metric). Coverage = shared tokens / the arm's total tokens over the
same prompts (from *_toks.json); < 50 % for either arm = no verdict. Line: >= 1.10 PASS, <= 1.03 KILL, else BETWEEN. Beside
it: the same on T3R's original 8 prompts; INFO: t80ck1um3 / t80ck2 (pooled and original 8). Checks: t80ck2 = T3t tokens + lsha
on 156 / 42 / 68 / 91, K1U engaged, lifetime <= 11.95 GB, '] done', the PAIR lines (SSD per arm, PAUSE, re-runs, exclusions).
T3R-PF (p47.txt, if it ran): every GATE PF2 PASS, t80ck1upf / t80ck1u summed tok/s (>= 1.05 PASS, <= 1.01 KILL), t80ck2pf /
t80ck2 second; its PAIR lines (a PAUSE inside its prompts voids its timings)."""
import glob, json, os, re, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/t3r2"
TAG = sys.argv[2] if len(sys.argv) > 2 else "r47"
ORIG8 = ["156", "42", "68", "91", "16", "6", "79", "19"]
T3T = {"156": ("ba7f9b3682", "674de3d0b4"), "42": ("33bbf6856a", "eae0a8bcd9"), "68": ("c96b655e9d", "685b9746a8"),
       "91": ("40348025ac", "d6162e5e65")}   # T3t's t80ck2 (tokens sha, lsha), /Users/kamen/Fit12/data/scr/t3t/q1.txt


def keys_of(arm):
    pre, suf = f"{TAG}_", f"_{arm}_ph.json"
    return sorted(os.path.basename(f)[len(pre):-len(suf)] for f in glob.glob(os.path.join(D, pre + "*" + suf)))


def load(key, arm, x):
    return json.load(open(os.path.join(D, f"{TAG}_{key}_{arm}{x}.json")))


def cut(key, a, b):
    ta, tb = load(key, a, "_toks"), load(key, b, "_toks")
    return next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))


def timed(key, arm, c):
    tok = n = 0
    for S, path, k, cf in load(key, arm, ""):
        if tok >= c:
            break
        tok += k; n += 1
    return tok, sum(p[6] - p[2] for p in load(key, arm, "_ph") if p[1] < n)


def shared(a, b, keys):
    ta = sa = tb = sb = na = nb = 0
    per = []
    for key in keys:
        c = cut(key, a, b)
        x, y = timed(key, a, c), timed(key, b, c)
        ta += x[0]; sa += x[1]; tb += y[0]; sb += y[1]
        na += len(load(key, a, "_toks")); nb += len(load(key, b, "_toks"))
        per.append(f"{key.split('_')[-1]} cut {c}: x{(y[0] / y[1]) / (x[0] / x[1]):.2f}")
    r = (tb / sb) / (ta / sa)
    return r, ta / na, tb / nb, f"{a} {ta} tok / {sa:.2f} s = {ta / sa:.1f} tok/s | {b} {tb} tok / {sb:.2f} s = {tb / sb:.1f} tok/s", per


def line(r, ca, cb, p_, k_):
    if min(ca, cb) < 0.5:
        return "NO VERDICT (coverage < 50 %)"
    return "PASS" if r >= p_ else "KILL" if r <= k_ else "BETWEEN"


f = os.path.join(D, f"{TAG}.txt")
txt = open(f).read() if os.path.exists(f) else ""
life = [float(v) for v in re.findall(r"lifetime ([0-9.]+) GB", txt)]
errs = [l for l in txt.splitlines() if "Error" in l or "Traceback" in l]
chk = [(m.group(1), (m.group(2), m.group(3)) == T3T[m.group(1)])
       for m in re.finditer(r"HumanEval/(\d+) t80ck2 theta .*?\| tok \d+ sha (\w+) \| lsha (\w+)", txt) if m.group(1) in T3T]
k1u = re.findall(r"HumanEval/(\d+) t80ck1u theta .*?K1U: ([0-9.]+) experts per row-layer", txt)
print("\n".join(l for l in txt.splitlines() if " PAIR " in l or "PAUSE summary" in l or " TOTAL " in l or "start (SSD" in l))
print(f"{TAG}: finished {'] done' in txt}; lifetime max {max(life) if life else float('nan'):.3f} GB (<= 11.95: "
      f"{bool(life) and max(life) <= 11.95}); t80ck2 = T3t {chk} -> {bool(chk) and all(v for _, v in chk)}; K1U engaged "
      f"{bool(k1u) and all(float(v) > 0 for _, v in k1u)} ({len(k1u)} runs)" + (f"; ERRORS: {errs[-1][:160]}" if errs else ""))
pair = {}                                                # (Tue 3 Nov 22:3x) the last PAIR line per prompt (main or re-run) must
for m in re.finditer(r"\] PAIR (\S+?)(?: \(re-run\))?: .*-> (.*)$", txt, re.M):   # say 'ok': a process cut between a prompt's
    pair[m.group(1).replace("/", "_")] = m.group(2).strip()                       # files and its PAIR line (or before its
live = [k for k in keys_of("t80ck2") if k in set(keys_of("t80ck1u"))]             # re-run) cannot be checked for a PAUSE
ks = [k for k in live if pair.get(k) == "ok"]
if len(ks) < len(live):
    print(f"live files without a clean PAIR line (not counted): {[(k.split('_')[-1], pair.get(k)) for k in live if k not in ks]}")
o8 = [k for k in ks if k.split("_")[-1] in ORIG8]
print(f"prompts that count: {len(ks)} ({', '.join(k.split('_')[-1] for k in ks)}); of the original 8: {len(o8)}")
if ks:
    r, ca, cb, desc, per = shared("t80ck2", "t80ck1u", ks)
    v = line(r, ca, cb, 1.10, 1.03)
    if len(ks) < 12:                                     # (NewIdeas 19:27, QUEUE 1ba42c3, before any run) >= 12 of 16 must count
        v = f"NO VERDICT ({len(ks)} of 16 prompts counted, < 12; the ratio on the line would read {v})"
    print(f"T3R2 VERDICT (pooled, {len(ks)} prompts): t80ck1u / t80ck2 x{r:.3f}, coverage {ca:.0%} / {cb:.0%} -> "
          f"{v} | {desc} | " + ", ".join(per))
    if o8:
        r, ca, cb, desc, _ = shared("t80ck2", "t80ck1u", o8)
        print(f"  beside it, T3R's original {len(o8)}: x{r:.3f}, coverage {ca:.0%} / {cb:.0%} ({line(r, ca, cb, 1.10, 1.03)} on "
              f"the line, not the verdict)")
    km = [k for k in ks if k in set(keys_of("t80ck1um3"))]
    if km:
        r, ca, cb, desc, per = shared("t80ck2", "t80ck1um3", km)
        print(f"INFO t80ck1um3 / t80ck2 (pooled, {len(km)}): x{r:.3f}, coverage {ca:.0%} / {cb:.0%} | " + ", ".join(per))
        km8 = [k for k in km if k.split("_")[-1] in ORIG8]
        if km8:
            r, ca, cb, _, _ = shared("t80ck2", "t80ck1um3", km8)
            print(f"  INFO on the original {len(km8)}: x{r:.3f}, coverage {ca:.0%} / {cb:.0%}")

f = os.path.join(D, "p47.txt")
if os.path.exists(f):
    txt = open(f).read()
    tot = {m.group(1): float(m.group(2)) for m in re.finditer(r"TOTAL (\w+): \d+ prompts, \d+ tokens in [0-9.]+ s = ([0-9.]+) tok/s", txt)}
    print("\n".join(l for l in txt.splitlines() if " TOTAL " in l or "GATE PF2" in l or " PAIR " in l))
    gates = re.findall(r"GATE PF2 \((\w+) = (\w+), tokens \+ logits\): (PASS|FAIL)", txt)
    pf2 = re.findall(r"HumanEval/(\d+) (t80ck\w*pf) theta .*?PF2: (\d+) late ids posted in (\d+) wide calls, loader late reads "
                     r"(\d+), forced to wait (\d+)", txt)
    gok = bool(gates) and all(g[2] == "PASS" for g in gates)
    pz = "PAUSE-FLAGGED" in txt
    print(f"p47: finished {'] done' in txt}; GATE PF2 {len(gates)} lines, all PASS {gok}; PAUSE inside its prompts {pz}; PF2 "
          f"(prompt, arm, posted, calls, late reads, forced) {pf2}")
    if gok and not pz:
        for a, b, name in (("t80ck1upf", "t80ck1u", "VERDICT"), ("t80ck2pf", "t80ck2", "second verdict")):
            if a in tot and b in tot:
                x = tot[a] / tot[b]
                print(f"T3R-PF {name}: {a} {tot[a]:.2f} / {b} {tot[b]:.2f} tok/s = x{x:.3f} -> "
                      f"{'PASS' if x >= 1.05 else 'KILL' if x <= 1.01 else 'BETWEEN'}")
    else:
        print("T3R-PF: no timing verdict (" + ("GATE PF2 not all PASS" if not gok else "a PAUSE inside its prompts") + ")")
else:
    print(f"p47: no output in {D} (skipped or not run)")
