"""FCP read-out (Leap, Sat 2026-11-07): per arm, check 0 vs checks 1-3 of every prompt (fcprobe.py's windows), then the
pre-registered lines. usage: python fcp_read.py [arms...]  (default fcpa fcpb) -> stdout (fcp_run.sh tees it)."""
import collections
import json
import os
import statistics as S
import sys

D = os.path.expanduser("~/Leap/data/fcp")
ARMS = sys.argv[1:] or ["fcpa", "fcpb"]
med = lambda xs: S.median(xs) if xs else float("nan")


def load(arm):
    try:
        return json.load(open(f"{D}/{arm}_fcp.json"))
    except (OSError, ValueError) as e:
        print(f"{arm}: no probe file ({e})")
        return None


def window(r):
    a, b, e = r["c1"], r["cb"], r["ce"]
    w = dict(wall=1e3 * (e["t"] - a["t"]), build=1e3 * (b["t"] - a["t"]), wait=1e3 * (e["t"] - b["t"]),
             main=1e3 * (e["tt"] - a["tt"]), proc=1e3 * (e["proc"] - a["proc"]))
    w["other"] = w["proc"] - w["main"]
    for k in a["ev"]:
        w["ev." + k] = e["ev"][k] - a["ev"][k]
    for k in ("zero_fill", "pageins", "faults", "decompressions", "purges", "swapins", "compressions"):
        w["vm." + k] = e["vm"][k] - a["vm"][k]
    w["vm.wire_mb"] = (e["vm"]["wire"] - a["vm"]["wire"]) * 16384 / 1e6
    sc = (e["ru_self"] - a["ru_self"]) / max(1e-9, w["proc"])          # proc_pid_rusage units per ms of this process
    w["mtl"] = sum(v - a["mtl"].get(p, 0) for p, v in e["mtl"].items() if v >= 0) / sc if sc > 0 else float("nan")
    w["mtl_n"] = len(e["mtl"])
    if a.get("mlx") and e.get("mlx"):
        w["mlx_cache0_mb"], w["mlx_act0_mb"] = a["mlx"][1] / 1e6, a["mlx"][0] / 1e6
    th = {}
    for h, (ns, nm) in e.get("thr", {}).items():
        d = ns - a["thr"].get(h, [0, nm])[0]
        if d > 0:
            th[nm or h] = th.get(nm or h, 0) + d
    w["thr"] = th
    return w


for arm in ARMS:
    P = load(arm)
    if P is None:
        continue
    W = collections.defaultdict(dict)
    for r in P["rec"]:
        W[r["run"]][r["chk"]] = window(r)
    runs = sorted(W)
    print(f"== {arm} (NOCLEAR {P['noclear']}): {len(P['rec'])} windows, {len(runs)} prompts (run 0..2 = warm-ups if 6 runs)")
    rows = [k for k in ("wall", "build", "wait", "main", "other", "mtl", "mtl_n", "ev.msg_sent", "ev.sys_mach", "ev.sys_unix", "ev.csw",
                        "ev.faults", "vm.zero_fill", "vm.faults", "vm.pageins", "vm.decompressions", "vm.purges",
                        "vm.wire_mb", "mlx_cache0_mb", "mlx_act0_mb")]
    last = runs[-3:]                                         # the measured prompts (after the warm-ups)
    print(f"  measured runs {last}; per check: median over them")
    print("  " + "metric".ljust(18) + "".join(f"chk{c}".rjust(11) for c in range(4)) + "   c0 - med(1-3)")
    for k in rows:
        v = [med([W[r][c][k] for r in last if c in W[r] and k in W[r][c]]) for c in range(4)]
        print("  " + k.ljust(18) + "".join(f"{x:11.1f}" for x in v) + f"   {v[0] - med(v[1:]):+10.1f}")
    # threads: CPU per thread name in check 0 vs checks 1-3 (units as proc_threadinfo reports them; ns expected)
    agg = collections.defaultdict(lambda: [[], []])
    for r in last:
        for c in range(4):
            for nm, ns in (W[r].get(c, {}).get("thr") or {}).items():
                agg[nm][0 if c == 0 else 1].append(ns / 1e6)
    top = sorted(agg.items(), key=lambda kv: -(sum(kv[1][0]) / max(1, len(last))))[:6]
    print("  threads (CPU ms, mean per prompt: check 0 | mean of checks 1-3): " +
          "; ".join(f"{nm[:28]} {sum(a) / len(last):.1f} | {sum(b) / max(1, 3 * len(last)):.1f}" for nm, (a, b) in top))
    # timed calls
    C = collections.defaultdict(lambda: collections.defaultdict(list))
    for run, chk, ph, kind, t0, wall, cpu, caller in P["calls"]:
        if run in last and chk is not None and 0 <= chk < 4:
            C[(kind, ph)][chk].append((wall, cpu, caller))
    print("  calls (per prompt mean: n, wall ms, CPU ms) check 0 vs mean of checks 1-3:")
    for (kind, ph), byc in sorted(C.items(), key=lambda kv: -sum(w for w, _, _ in kv[1].get(0, []))):
        z = byc.get(0, []); o = [x for c in (1, 2, 3) for x in byc.get(c, [])]
        n = len(last)
        print(f"    {kind:20s} {ph:5s} chk0 n {len(z) / n:6.1f} wall {sum(w for w, _, _ in z) / n:8.1f} cpu {sum(c for _, c, _ in z) / n:7.1f}"
              f"   | chk1-3 n {len(o) / (3 * n):6.1f} wall {sum(w for w, _, _ in o) / (3 * n):8.1f} cpu {sum(c for _, c, _ in o) / (3 * n):7.1f}")
    big = sorted(((w, c, k, cl, r) for (k, ph), byc in C.items() for (w, c, cl) in byc.get(0, []) for r in [ph]), reverse=True)[:6]
    print("  longest single calls in check 0: " + "; ".join(f"{k} {w:.1f} ms (cpu {c:.1f}) at {cl} [{r}]" for w, c, k, cl, r in big))
    # sampler: wall by main-thread line, build phase, check 0 vs checks 1-3
    sm = collections.defaultdict(lambda: collections.defaultdict(float))
    prev = None
    for t, run, chk, ph, key in P["samp"]:
        if prev is not None and prev[1] == run and prev[2] == chk and prev[3] == ph and run in last and 0 <= chk < 4:
            top2 = " < ".join(prev[4].split(" < ")[:2])
            sm[(0 if chk == 0 else 1, ph)][top2] += 1e3 * (t - prev[0])
        prev = (t, run, chk, ph, key)
    for ph in ("build", "wait"):
        a, b = sm[(0, ph)], sm[(1, ph)]
        ex = sorted(((a[k] / len(last) - b.get(k, 0) / (3 * len(last)), k) for k in a), reverse=True)[:8]
        print(f"  sampler {ph}: main-thread wall by line, check 0 minus mean(1-3) per prompt (ms): " +
              "; ".join(f"{d:+.1f} {k}" for d, k in ex))
    print()

print("LINES (pre-registered in fcp_run.sh's commit): NAME = one line / call holds >= 60 % of check 0's build excess; "
      "class by counters (COMPILE: MTLCompilerService CPU >= 50 % of the excess; VM: zero fills + decompressions + pageins "
      ">= 1 GB more than checks 1-3; THREAD: other threads' CPU >= 50 % of it). fcpb: PASS if check 0's wall drops >= 100 ms "
      "vs fcpa with every check's logits hash equal; KILL < 30 ms; else BETWEEN.")

for arm in ARMS:                                             # the screen's own summary per arm (decode tok/s, peak)
    try:
        T = open(f"{D}/{arm}.txt").read().splitlines()
    except OSError:
        continue
    s = [x.strip()[:150] for x in T if ": tok/s" in x and "where" not in x][-1:] + [x.strip()[:80] for x in T if "footprint now" in x][-1:]
    print(f"{arm}: " + " | ".join(s))
for pr in os.environ.get("FCP_PAIRS", "").split():          # bitwise: each check's verify-logits hash (R57_LOGHASH)
    a, b = pr.split(":")
    try:
        A = {r["key"]: r for r in json.load(open(f"{D}/r57/{a}_runs.json"))}
        B = {r["key"]: r for r in json.load(open(f"{D}/r57/{b}_runs.json"))}
    except (OSError, ValueError) as e:
        print(f"pair {pr}: missing ({e})")
        continue
    ks = [k for k in A if k in B]
    lh = [bool(A[k].get("lh")) and A[k].get("lh") == B[k].get("lh") for k in ks]
    try:                                                     # tokens from <arm>_tokens.json (runs.json has no tokens:
        TA = json.load(open(f"{D}/r57/{a}_tokens.json"))     # Sat 15:1x fix: the earlier .get("out") compared None to None)
        TB = json.load(open(f"{D}/r57/{b}_tokens.json"))
        tok = [k in TA and k in TB and TA[k] == TB[k] for k in ks]
    except (OSError, ValueError):
        tok = [False for _ in ks]
    print(f"pair {a} vs {b}: logits hash equal on every check in {sum(lh)} of {len(ks)} prompts "
          f"({sum(len(A[k].get('lh') or []) for k in ks)} checks); tokens equal in {sum(tok)} of {len(ks)}")

for arm in ARMS:                                             # PLAN 220: the partial-commit split (FCP_COMMIT arms)
    P = load(arm)
    if not P or not P.get("crec"):
        continue
    runs = sorted({r for r, *_ in P["crec"]})
    keep = set(runs[-int(os.environ.get("FCP_NPROMPTS", "6")):])
    G = collections.defaultdict(list)
    for run, chk, kind, ck, t0, wall, cpu, sync in P["crec"]:
        if run in keep and ck:
            G[(kind, "partial" if ck[0] < ck[1] else "full")].append((wall, cpu, sync))
    print(f"== {arm}: commit split (main thread, ms; medians; runs {sorted(keep)}; CSYNC {P.get('csync')})")
    for kind in ("commit", "lz_pre", "mtplx_commit", "layer_fwd", "lz_post"):
        for pf in ("partial", "full"):
            v = G.get((kind, pf), [])
            if v:
                sy = [x for _, _, x in v if x is not None]
                print(f"    {kind:13s} {pf:7s} n {len(v):4d}  wall {med([w for w, _, _ in v]):7.2f}  cpu {med([c for _, c, _ in v]):7.2f}"
                      + (f"  sync after {med(sy):7.2f}" if sy else ""))
    try:
        C = json.load(open(f"{D}/r57/{arm}_checks.json"))
        pe, dr = collections.defaultdict(list), collections.defaultdict(list)
        for c in C:
            n = len(c["ck"])
            for i in range(1, n):
                pf = "partial" if c["kept"][i] < c["rows"][i] else "full"
                pe[pf].append(c["ck"][i] - c["bld"][i] - c["wt"][i])
                if i + 1 < n:
                    dr[pf].append(c["drf"][i + 1])
        tot = len(pe["partial"]) + len(pe["full"])
        print(f"    checks.json: partial {len(pe['partial'])} / {tot}; post-eval ms partial {med(pe['partial']):.2f} full "
              f"{med(pe['full']):.2f}; next draft ms after partial {med(dr['partial']):.2f} after full {med(dr['full']):.2f}")
    except (OSError, ValueError, KeyError) as e:
        print(f"    checks.json: {e}")
