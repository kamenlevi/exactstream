"""M-13 MISSFLOOR (MoonShots, Sun 2026-11-01; SSD-miss wall, a measurement for cache sizing): how low can misses go with a
bigger expert cache alone?
Data (read only): BigMoE's cap45 live captures (~/BigMoE/data/cap45/*_gate_0.npz; the exact REF2 loop, real DFlash2 16-row
checks, 6 prompts run in sequence; res0 = each prompt's residency at loop start, C 160). Each check = 48 layer calls of 16
rows + the commit replay of layer 1 (kept rows; all hits, ignored). Per layer the check needs the union of its rows' experts.
(a) Replays per layer from res0: LRU and decayed LFU (BigMoE's ext rule: each call scales the increment by 1 / 0.99, needed
    experts += it, victims = the lowest score outside the call's needs). At C 160 they must land within 15 % of the logged
    per-check medians (970 / 552 / 624 / 870 / 601 / 666); then C 224 / 288 / 384 (from res0, the cache grows up to C).
(b) Floor with an unlimited cache: experts first used in the decode and absent from res0. Steady state = the checks in each
    answer's second half.
Reported per check and per kept token, and as SSD MB per kept token at 2.16 MB (MSB-3 record) and 2.77 MB (4-bit).
Lines (QUEUE M-13, written first): PASS = the steady-state floor <= 14 misses per kept token on the code prompts; KILL = > 28;
BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python missfloor45.py (CPU, one core, ~1 min) -> stdout + data/missfloor45.txt"""
import glob, os, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, R = 48, 16
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")                  # the capture run's order (cap45_run.txt)
LOGGED = {"humaneval_HumanEval_68": 970, "math500_test_prealgebra_1640": 552, "humaneval_HumanEval_91": 624,
          "math500_test_algebra_694": 870, "math500_test_algebra_1282": 601, "gsm8k_708": 666}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
t0 = time.time()
P = {}
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"need": {}, "kept": None})
        if s not in order:
            order.append(s)
        if l in c["need"]:
            c["kept"] = r                                            # the commit replay: kept rows
        else:
            c["need"][l] = frozenset(np.unique(ids[roff[n]:roff[n] + r]).tolist())
    cks = [checks[s] for s in order if len(checks[s]["need"]) == NL and checks[s]["kept"] is not None]
    res0 = z["res0"]                                                 # [48, 512] slot or -1
    P[name] = (cks, [frozenset(np.nonzero(res0[l] >= 0)[0].tolist()) for l in range(NL)])
say(f"M-13 MISSFLOOR: cap45, {sum(len(v[0]) for v in P.values())} live checks over {len(P)} prompts; res0 holds "
    f"{np.mean([len(x) for v in P.values() for x in v[1]]):.0f} experts per layer at loop start")


def replay(cks, start, C, policy):
    """misses per check (sum over layers) from the residency at loop start"""
    out = []
    state = []
    for l in range(NL):
        st = {e: 0.0 for e in start[l]}                              # LRU: last-use stamp; LFU: decayed score
        state.append(st)
    w = 1.0
    for t, c in enumerate(cks):
        w /= 0.99
        m = 0
        for l in range(NL):
            st, need = state[l], c["need"][l]
            miss = [e for e in need if e not in st]
            m += len(miss)
            for e in need:
                if policy == "lru":
                    st[e] = t + 1
                else:
                    st[e] = st.get(e, 0.0) + w
            if len(st) > C:
                cand = sorted((v, e) for e, v in st.items() if e not in need)
                for v, e in cand[:len(st) - C]:
                    del st[e]
        out.append(m)
    return np.array(out)


say("(a) replay misses per check (median) vs the logged median at C 160:")
say("  prompt                          kept  logged   LRU 160  LFU 160 |  LFU 224  LFU 288  LFU 384")
ok = True
res = {}
for name in ORDER:
    cks, start = P[name]
    kept = np.array([c["kept"] for c in cks])
    row = {}
    for C in (160, 224, 288, 384):
        for pol in (("lru", "lfu") if C == 160 else ("lfu",)):
            row[(pol, C)] = replay(cks, start, C, pol)
    res[name] = (kept, row)
    lfu160 = np.median(row[("lfu", 160)])
    lru160 = np.median(row[("lru", 160)])
    ok &= abs(min(lfu160, lru160, key=lambda v: abs(v - LOGGED[name])) / LOGGED[name] - 1) <= 0.15
    say(f"  {name[:30]:30s} {kept.mean():5.2f} {LOGGED[name]:7d} {lru160:9.0f} {lfu160:8.0f} | " +
        " ".join(f"{np.median(row[('lfu', C)]):8.0f}" for C in (224, 288, 384)))
say(f"  calibration (one policy within 15 % of the logged median on every prompt): {'OK' if ok else 'NOT OK'}")
say("(b) floor with an unlimited cache (first use in the decode, absent from res0), per kept token:")
code_floor = []
for name in ORDER:
    cks, start = P[name]
    kept = np.array([c["kept"] for c in cks], float)
    seen = [set(start[l]) for l in range(NL)]
    new = np.zeros(len(cks))
    for t, c in enumerate(cks):
        for l in range(NL):
            n = c["need"][l] - seen[l]
            new[t] += len(n)
            seen[l] |= n
    h = len(cks) // 2
    whole = new.sum() / kept.sum()
    steady = new[h:].sum() / kept[h:].sum()
    lfu288 = res[name][1][("lfu", 288)]
    say(f"  {name[:30]:30s} whole answer {whole:6.1f}, second half {steady:6.1f} misses per kept token "
        f"({steady * 2.16:5.0f} MB at 2.16 MB); LFU C 288 second half {lfu288[h:].sum() / kept[h:].sum():6.1f}; "
        f"{len(cks)} checks")
    if "humaneval" in name:
        code_floor.append(steady)
say("(c) who the floor's first uses are: share inside the global popularity top-K per layer (counted on the OTHER five "
    "prompts' checks, leave-one-prompt-out), second half:")
for name in ORDER:
    pop = np.zeros((NL, 512))
    for other in ORDER:
        if other != name:
            for c in P[other][0]:
                for l in range(NL):
                    pop[l, list(c["need"][l])] += 1
    rank = np.argsort(np.argsort(-pop, axis=1), axis=1)            # 0 = most used
    cks, start = P[name]
    seen = [set(start[l]) for l in range(NL)]
    h = len(cks) // 2
    tot, inK = 0, {224: 0, 288: 0, 384: 0}
    for t, c in enumerate(cks):
        for l in range(NL):
            n = c["need"][l] - seen[l]
            seen[l] |= n
            if t >= h:
                tot += len(n)
                for K in inK:
                    inK[K] += sum(1 for e in n if rank[l, e] < K)
    say(f"  {name[:30]:30s} " + ", ".join(f"top-{K} {100 * v / max(tot, 1):4.0f} %" for K, v in inK.items()) +
        f" of {tot} first uses")
cf = float(np.mean([v for v in code_floor[1:]]))                   # HumanEval/68: 5 checks captured, dropped
call = "PASS" if cf <= 14 else ("KILL" if cf > 28 else "BETWEEN")
say(f"Lines: steady-state floor on the code prompt (HumanEval/91; /68 has 5 checks) {cf:.1f} misses per kept token (PASS <= 14, KILL > 28) -> {call}"
    f"{'' if ok else ' (replay calibration NOT OK: read (a) with care)'}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/missfloor45.txt"), "w").write("\n".join(lines) + "\n")
