"""Fit12 (Sun 8 Nov evening; the orchestrator 19:1x: take what works from Strata's low-memory chat loop). CPU replay [P] of
verify-width policies on the 12 GB chat loop, from measured logs only. Strata speculates ONE token per pass on chat (2 rows,
1.67 tokens per pass, 115.7 ms per pass). Ours checks 15 drafts, in calls of up to 4 rows.
Inputs (read-only):
- /Users/kamen/Fit12/data/scr/chw13/chw13a_* (A' = 4-row calls): per check the committed count = L + 1, L = accepted drafts;
- /Users/kamen/Fit12/data/scr/chw14/chw14c2_* (P2' = cf0 < 0.9 -> a 2-row first call, else 4; later calls 4 rows).
Cost model, measured on the same 12 IFEval prompts at CAP 128:
- per call: the mean wall time of the measured calls of that width and position (first / later in the check), pooled over both
  runs. Each call's own time is t1 - t0 in the per-call log;
- per check: the mean draft phase (phase log) plus the mean per-check rest (decode window - calls - drafts, per run).
Policies, replayed on A''s checks (each A' check = a run of L accepted drafts, then a miss or the end):
- A': calls of 4 rows until the miss (as measured; the calibration);
- P2': the first call 2 rows if cf0 < 0.9, else 4; later calls 4 (as measured in chw14c2; a second calibration);
- C2: every check verifies ONE 2-row call (x + the first draft), then re-drafts. A run of L accepted drafts takes
  ceil((L + 1) / 2) checks, each one draft + one first-position 2-row call. This assumes a check started inside an accepted run
  keeps its first draft, as the old block did there [G];
- C2c: one draft per check, then 2-row calls until the miss: ceil((L + 1) / 2) calls of 2 rows;
- P2'c: cf0 < 0.9 -> C2c, else A' (4-row calls).
Prints tok/s [P] per policy for the decode window, the calibration against the measured whole decode, and the kept-token
distribution. Usage: python -I -B rows_price.py"""
import glob, json, math, os, re, statistics

S = "/Users/kamen/Fit12/data/scr"
import sys
SET = sys.argv[1] if len(sys.argv) > 1 else "chat"   # chat: chw13a + chw14c2; code: chw14ha + chw14h2 (16 HumanEval, CAP 256)
RUNS = {"chw13a": f"{S}/chw13", "chw14c2": f"{S}/chw14"} if SET == "chat" else {"chw14ha": f"{S}/chw14", "chw14h2": f"{S}/chw14"}
REF, CMP = list(RUNS)


def load(tag, d):
    pr = {}
    for f in sorted(glob.glob(f"{d}/{tag}_*_ex.json")):
        base = f[:-len(".json")]
        key = os.path.basename(base)[len(tag) + 1:-len("_ex")]
        ck = json.load(open(f))                         # per check: rows, path, committed, cf0
        cl = json.load(open(base + "_calls.json"))      # per call: check, ci, w, kept, t0, seq, bytes, miss, t1, seq, bytes, miss
        ph = json.load(open(base + "_ph.json"))         # per phase: name, check, (uptime, seq, bytes, misses) x 2
        pr[key] = dict(ck=ck, cl=cl, ph=ph)
    dec = {}
    for ln in open(f"{d}/{tag}.txt"):
        m = re.match(rf"^\[{tag}\] (\S+) ex theta None .*?\| tok (\d+) .*?decode ([\d.]+) s", ln)
        if m:
            dec[m.group(1).replace("/", "_")] = (int(m.group(2)), float(m.group(3)))
    return pr, dec


data = {t: load(t, d) for t, d in RUNS.items()}
cost = {}                                               # (pos, w) -> [ms]
drafts, rest = [], {}
for t, (pr, dec) in data.items():
    tot_rest, n_ck = 0.0, 0
    for k, r in pr.items():
        for c in r["cl"]:
            cost.setdefault(("first" if c[1] == 0 else "later", c[2]), []).append((c[8] - c[4]) * 1e3)
        dr = [(p[6] - p[2]) * 1e3 for p in r["ph"] if p[0] == "draft"]
        drafts += dr
        calls_ms = sum((c[8] - c[4]) * 1e3 for c in r["cl"])
        tot_rest += dec[k][1] * 1e3 - calls_ms - sum(dr)
        n_ck += len(r["ck"])
    rest[t] = tot_rest / max(1, n_ck)
C = {k: statistics.mean(v) for k, v in cost.items()}
D = statistics.mean(drafts)
R = statistics.mean(rest.values())
print("Measured per-call ms (pos, width): " + ", ".join(f"{p} w{w} {C[(p, w)]:.1f} (n {len(cost[(p, w)])})" for p, w in sorted(C)))
print(f"Draft {D:.1f} ms per check; per-check rest {', '.join(f'{t} {v:.1f}' for t, v in rest.items())} ms (used {R:.1f})")
cw = lambda p, w: C.get((p, w), C[("first", w)] if p == "later" and ("first", w) in C else float("nan"))
pr, dec = data[REF]
checks = [(c[2] - 1, c[3]) for r in pr.values() for c in r["ck"]]          # (L, cf0) per A' check
toks = sum(L + 1 for L, _ in checks)
Ls = [L for L, _ in checks]
print(f"A' checks {len(checks)}, tokens {toks}, kept per check {toks / len(checks):.3f}; P(L = 0) {Ls.count(0) / len(Ls):.3f}, "
      f"P(L >= 1) {1 - Ls.count(0) / len(Ls):.3f}, P(L >= 3) {sum(1 for L in Ls if L >= 3) / len(Ls):.3f}")


def a_prime(L, w1=4):
    rows, n, ms = L + 1, 0, D + R                       # rows needed = L + 1 (the miss row or the bonus)
    w = w1; done = 0
    while True:
        ms += cw("first" if n == 0 else "later", w); n += 1; done += w
        if done >= rows or done >= 16:
            return ms
        w = 4


def c2(L):
    k = math.ceil((L + 1) / 2)
    return k * (D + R + cw("first", 2))


def c2c(L, later=None):
    k = math.ceil((L + 1) / 2)
    return D + R + cw("first", 2) + (k - 1) * (cw("later", 2) if later is None else later)


pol = {"A'": lambda L, cf: a_prime(L, 4), "P2'": lambda L, cf: a_prime(L, 2 if cf < 0.9 else 4),
       "C2": lambda L, cf: c2(L), "C2c": lambda L, cf: c2c(L), "C2c-cons": lambda L, cf: c2c(L, cw("first", 2)), "P2'c": lambda L, cf: c2c(L) if cf < 0.9 else a_prime(L, 4)}
res = {}
for name, f in pol.items():
    ms = sum(f(L, cf) for L, cf in checks)
    res[name] = toks / (ms / 1e3)
meas = {t: sum(v[0] for v in dec_.values()) / sum(v[1] for v in dec_.values()) for t, (_, dec_) in data.items()}
print(f"[{SET}] Measured whole decode (pooled): A' ({REF}) {meas[REF]:.2f}, P2' ({CMP}) {meas[CMP]:.2f} tok/s")
for name, v in res.items():
    print(f"  {name:8}: {v:6.2f} tok/s [P] (x{v / res[chr(65) + chr(39)]:.3f} vs A' replay; x{v / res['P2' + chr(39)]:.3f} vs P2' replay)")
print("Caveats: P2''s replay uses A''s check outcomes (its tokens differ on 7 / 12 prompts); C2 and C2c re-check inside accepted "
      "runs at 2 rows, which assumes the first draft there is kept as the old block's was [G]; later-w2 calls are rare in the logs.")
