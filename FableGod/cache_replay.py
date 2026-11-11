#!/usr/bin/env python3 -I
"""cache_replay.py (FableGod, mission 20). CPU, no numpy. Expert-cache policies replayed on DAY9's routing capture
(decision_map/out/routing/*.npz: rank-sorted top-10 expert ids per position and layer, prompt + answer) with the real
check boundaries and kept counts of the HPD-0 walks (per check: accepted k, checkable drafts m, anchor A), per domain
(12 windows each, replayed in order as one session; the cache persists across windows).
Prompt rows are read position by position (exact top-10); each answer check reads the union of its rows' experts.
Policies per layer at C slots: LRU; LRU + prompt protection (experts routed to by the prompt are evicted last for the
first N checks of the answer); LRU + a pinned hot set (the P most-used experts per layer from the prompt, or from a global
prior = the other windows); both; Belady's MIN (evict the resident expert used farthest in the future: the ceiling).
Start-up: misses in the first 1 / 2 / 3 checks with and without pre-loading the prompt's experts (by prompt frequency)
at prefill end. Two-pass check (quality-defined): pass 1 reads top-k1 for every row, rows certified when their own p1
under that cut >= theta (the map's topk2 / topk5 runs), pass 2 reads ranks k1+1..10 for uncertified rows up to and
including the first exact miss; also the 125B-sure oracle (top-2 on sure rows). Output: misses per kept token and MB
per check (2.2 MB per expert record at the 45 GB tiers)."""
import zipfile, struct, array, json, os, re, math, sys, time
from collections import OrderedDict, defaultdict, Counter
OUT = os.path.expanduser("~/FableGod/decision_map/out"); TD = os.path.expanduser("~/TrainLab/data")
MB = 2.2
def npz(path):
    z = zipfile.ZipFile(path); out = {}
    for i in z.infolist():
        d = z.read(i.filename); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
        descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
        shape = tuple(int(x) for x in re.findall(r"\d+", re.search(r"'shape': \(([^)]*)\)", hdr).group(1)))
        name = i.filename[:-4]
        if descr.startswith("<U"):
            w = int(descr[2:]); n = shape[0] if shape else 1
            out[name] = [body[j * w * 4:(j + 1) * w * 4].decode("utf-32-le").rstrip("\x00") for j in range(n)]
        elif descr == "<f2": n = len(body) // 2; out[name] = (list(struct.unpack(f"<{n}e", body)), shape)
        else:
            a = array.array({"<i2": "h", "<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d"}[descr]); a.frombytes(body); out[name] = (a, shape)
    return out
base = json.load(open(f"{OUT}/baseline.json"))
P2 = json.load(open(f"{OUT}/topk2.json")); P5 = json.load(open(f"{OUT}/topk5.json"))
rows = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))); CLS = defaultdict(dict)
for r in rows: CLS[r["nm"]][r["j"]] = r["cls"]
DOMS = {"code": "code", "ifeval": "ifeval", "multi": "multi"}
def load_domain(dom):
    walk = npz(f"{TD}/d02b_walk_hpd0_{dom}.npz"); names = walk["names"]
    wins = []
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom: continue
        z = zipfile.ZipFile(f"{OUT}/routing/{nm}"); ids = npz(f"{OUT}/routing/{nm}")["ids"][0]
        a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a
        checks, (nc, _) = walk[f"{i}_checks"]
        ck = []
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            last = min(A + m, n1 - 1)
            ck.append((A, last, k + 1))                      # rows A..last, kept = accepted + bonus
        cert2 = [math.exp(lp[0]) for lp in P2[nm]["lp"]]; cert5 = [math.exp(lp[0]) for lp in P5[nm]["lp"]]
        wins.append(dict(nm=nm, ids=ids, a=a, n1=n1, checks=ck, cert2=cert2, cert5=cert5, cls=CLS[nm]))
    return wins
def top(ids, p, l, k=10): o = (p * 48 + l) * 10; return ids[o:o + k]

def row_experts(w, p, l, scheme, first_miss):
    """expert ids read for row p at layer l under a scheme."""
    j = p - w["a"]
    if scheme == "exact" or j < 0 or j >= len(w["cert2"]): return top(w["ids"], p, l, 10)
    if scheme == "oracle": return top(w["ids"], p, l, 2 if w["cls"].get(j) == "sure" else 10)
    k1, theta, cert = (2, 0.99, w["cert2"]) if scheme == "two-pass top-2" else (5, 0.9, w["cert5"])
    if cert[j] >= theta or p > first_miss: return top(w["ids"], p, l, k1)
    return top(w["ids"], p, l, 10)

def replay(wins, C, policy, scheme="exact", N=30, P=0, prior=None, preload=False, startup=None):
    """returns (misses on answer checks, kept tokens, checks, misses per check list)"""
    cache = [OrderedDict() for _ in range(48)]; pinned = [set() for _ in range(48)]
    miss = 0; kept = 0; nchk = 0; first3 = [0, 0, 0]; per_check = []
    # Belady needs the access stream per layer: build it per window lazily (the future within the session)
    for w in wins:
        ids, a, n1 = w["ids"], w["a"], w["n1"]
        # the access stream: steps = prompt positions, then checks (as sets)
        steps = [[top(ids, p, l, 10) for l in range(48)] for p in range(a)]
        first_miss_pos = [A + (k - 1) for (A, last, k) in w["checks"]]   # the first exact miss row (k - 1 = accepted count)
        for (A, last, k), fm in zip(w["checks"], first_miss_pos):
            steps.append([set().union(*[row_experts(w, p, l, scheme, fm) for p in range(A, last + 1)]) for l in range(48)])
        nxt = None
        if policy == "belady":
            nxt = [dict() for _ in range(len(steps))]            # per step, per layer: expert -> next step index
            last_use = [dict() for _ in range(48)]
            for s in range(len(steps) - 1, -1, -1):
                for l in range(48):
                    d = {}
                    for e in steps[s][l]:
                        d[e] = last_use[l].get(e, 10 ** 9); last_use[l][e] = s
                    nxt[s][l] = d
        # prompt counts for protection / pinning
        pc = [Counter() for _ in range(48)]
        for p in range(a):
            for l in range(48): pc[l].update(top(ids, p, l, 10))
        if policy in ("pin", "both"):
            src = prior if prior is not None else pc
            pinned = [set(e for e, _ in src[l].most_common(P)) for l in range(48)]
        prot_until = [dict() for _ in range(48)]
        for s, st in enumerate(steps):
            is_check = s >= a
            if is_check and s == a:
                if preload:
                    for l in range(48):
                        c = cache[l]
                        for e, _ in reversed(pc[l].most_common(C)):
                            if e in c: c.move_to_end(e)
                            else:
                                c[e] = 1
                                while len(c) > C:
                                    victim = next(iter(c)); c.pop(victim)
                if policy in ("prot", "both"):
                    for l in range(48):
                        for e in pc[l]: prot_until[l][e] = a + N
            m_here = 0
            for l in range(48):
                c = cache[l]
                for e in st[l]:
                    if e in c: c.move_to_end(e); continue
                    if is_check: m_here += 1
                    c[e] = 1
                    if len(c) > C:
                        if policy == "belady":
                            fut = nxt[s][l]; far = None; farv = -1
                            for r in c:
                                v = fut.get(r)
                                if v is None:                    # not used at this step: look it up from this step's table is enough only for used ones
                                    v = next_use_lookup(nxt, s, l, r)
                                if v > farv: far, farv = r, v
                            c.pop(far)
                        else:
                            victim = None
                            if policy in ("prot", "both", "pin"):
                                for r in c:
                                    if r in pinned[l]: continue
                                    if policy in ("prot", "both") and prot_until[l].get(r, -1) > s: continue
                                    victim = r; break
                                if victim is None:
                                    for r in c:
                                        if r not in pinned[l]: victim = r; break
                            if victim is None: victim = next(iter(c))
                            c.pop(victim)
            if is_check:
                kept += w["checks"][s - a][2]; nchk += 1; per_check.append(m_here); miss += m_here
                ci = s - a
                if ci < 3: first3[ci] += m_here
    return miss, kept, nchk, first3, per_check

NEXT_CACHE = {}
def next_use_lookup(nxt, s, l, e):
    """next step >= s+1 at which expert e is used at layer l (scan forward; memoised per (s, l, e) is too big, so scan)."""
    for t in range(s + 1, len(nxt)):
        v = nxt[t][l].get(e)
        if v is not None: return t
    return 10 ** 9

def pct(n, d): return 100.0 * n / d if d else float("nan")
CS = [51, 160, 256, 320]
t0 = time.time()
results = {}
for dom in DOMS:
    wins = load_domain(dom)
    # global prior: expert counts over the other windows of the domain (prompt + answer), per layer
    cnt_all = [Counter() for _ in range(48)]
    for w in wins:
        for p in range(w["n1"]):
            for l in range(48): cnt_all[l].update(top(w["ids"], p, l, 10))
    print(f"\n### {dom}: {len(wins)} windows, {sum(len(w['checks']) for w in wins)} checks, {sum(k for w in wins for _, _, k in w['checks'])} kept tokens, prompt rows {sum(w['a'] for w in wins)} ({time.time() - t0:.0f} s)")
    print("| C | LRU | +prot N=10 | +prot N=30 | +prot N=100 | +pin C/4 (prompt) | +pin C/4 (prior) | prot 30 + pin C/4 | Belady MIN |")
    print("|---|---|---|---|---|---|---|---|---|")
    for C in CS:
        cells = []
        for pol, kw in (("lru", {}), ("prot", {"N": 10}), ("prot", {"N": 30}), ("prot", {"N": 100}), ("pin", {"P": C // 4}),
                        ("pin", {"P": C // 4, "prior": cnt_all}), ("both", {"N": 30, "P": C // 4}), ("belady", {})):
            if pol == "belady" and C > 160 and dom != "code": cells.append("—"); continue
            m, k, n, f3, pc_ = replay(wins, C, pol, **kw)
            results[(dom, C, pol, tuple(sorted(kw.items())) if "prior" not in kw else "prior")] = (m, k, n)
            cells.append(f"{m / k:.1f}")
        print(f"| {C} | " + " | ".join(cells) + f" |  ({time.time() - t0:.0f} s)")
    # start-up: misses in the first 1 / 2 / 3 checks per window, LRU vs preload
    for C in (51, 160, 256):
        a_ = replay(wins, C, "lru"); b_ = replay(wins, C, "lru", preload=True)
        print(f"start-up at C {C}: misses in the first 1 / 2 / 3 checks per window: LRU {a_[3][0] / len(wins):.0f} / {(a_[3][0] + a_[3][1]) / len(wins):.0f} / {sum(a_[3]) / len(wins):.0f}; pre-loaded prompt experts {b_[3][0] / len(wins):.0f} / {(b_[3][0] + b_[3][1]) / len(wins):.0f} / {sum(b_[3]) / len(wins):.0f}; steady misses per check (LRU) {a_[0] / a_[2]:.0f}")
    # two-pass at C 51 and 160 (and 256 for reference), LRU and prot 30
    print("| two-pass scheme | C | policy | misses per check | MB per check | misses per kept | SSD cap tok/s (13 GB/s) |")
    print("|---|---|---|---|---|---|---|")
    for scheme in ("exact", "two-pass top-5", "two-pass top-2", "oracle"):
        for C in (51, 160, 256):
            for pol, kw in (("lru", {}), ("prot", {"N": 30})):
                m, k, n, f3, pc_ = replay(wins, C, pol, scheme=scheme, **kw)
                print(f"| {scheme} | {C} | {pol}{' 30' if pol == 'prot' else ''} | {m / n:.0f} | {m / n * MB:.0f} | {m / k:.1f} | {13000 / (m / k * MB):.0f} |")
print(f"\ndone in {time.time() - t0:.0f} s")
