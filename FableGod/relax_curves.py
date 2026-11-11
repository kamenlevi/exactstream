#!/usr/bin/env python3 -I
"""relax_curves.py (FableGod, mission 16). CPU, no numpy. Acceptance curves for relaxed (quality-defined) acceptance rules,
from existing data: every drafted position DFlash2 L1-20 reached in the HPD-0 walks (the draft token d and the drafter's
selector softmax cf ~ q(d)) joined with the 125B's own distribution at the verifying row (decision_map/out/baseline.json
top-8 for the 12 code windows; ~/TrainLab/data/d02a/tf_*.npz top-32 for GSM8K / IFEval / multi). Rules:
  EXACT            accept iff d == argmax
  R1(delta)        accept iff log p(d) >= log p1 - delta            (nats)
  R2(k, x)         accept iff d in top-k and p(d) >= x * p1
  R3(T)            expected acceptance min(1, p_T(d) / q(d)), p_T renormalised over the stored top-k, q(d) = cf
  U-R2(k, x)       R2 only where the 125B is unsure (p1 < 0.99); sure rows stay exact
  TYP(eps, alpha)  typical acceptance: p(d) >= min(eps, alpha * exp(-H)), H = entropy over the stored top-k
Per rule and domain: acceptance per tested position (all / by class / by depth band), first-order deviation from greedy
(accepted and d != argmax), and accepted drafts per check at 16 and 32 rows from the per-depth acceptance profile
(depths 16-31 extrapolated with the measured decay from depths 11 to 15). Caveat: teacher-forced data; after a deviation
the engine's path leaves the stored one, so all numbers are first order."""
import zipfile, struct, array, json, os, math, re
from collections import defaultdict
TD = os.path.expanduser("~/TrainLab/data"); OUT = os.path.expanduser("~/FableGod/decision_map/out")

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

def dist_tables(dom):
    """name -> (row0, [(tk list, lp list) per row]); row index p = the position predicting token p+1."""
    D = {}
    if dom == "code":
        base = json.load(open(f"{OUT}/baseline.json"))
        for nm, b in base.items():
            if b["dom"] == "code": D[nm] = (b["a"], list(zip(b["tk"], b["lp"])))
    else:
        pref = {"ifeval": "tf_ifeval", "multi": "tf_multi", "math": "tf_gsm"}[dom]
        for f in os.listdir(f"{TD}/d02a"):
            if not f.startswith(pref): continue
            z = npz(f"{TD}/d02a/{f}"); tk, (n, k) = z["tk"]; lp, _ = z["lp"]
            D[f] = (0, [(list(tk[r * k:(r + 1) * k]), lp[r * k:(r + 1) * k]) for r in range(n)])
    return D

DOMS = ["code", "math", "ifeval", "multi"]
POS = []                                                    # tested positions with the 125B's distribution
for dom in DOMS:
    W = npz(f"{TD}/d02b_walk_hpd0_{dom}.npz"); names = W["names"]; D = dist_tables(dom); found = 0
    for i, nm in enumerate(names):
        key = nm if nm in D else (nm + ".npz" if nm + ".npz" in D else None)
        if key is None: continue
        found += 1; row0, rows = D[key]
        ids = W[f"{i}_ids"][0]; checks, (nc, _) = W[f"{i}_checks"]; dcf = W[f"{i}_dcf"][0]; dtok = W[f"{i}_dtok"][0]
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            for j in range(min(k + 1, m)):
                p = A + j; r = p - row0
                if p + 1 >= len(ids) or not (0 <= r < len(rows)): continue
                tk, lp = rows[r]; d = dtok[c * 15 + j]
                p1, p2 = math.exp(lp[0]), math.exp(lp[1])
                cls = "sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid")
                lpd = lp[tk.index(d)] if d in tk else None
                POS.append(dict(dom=dom, depth=j + 1, cls=cls, right=(d == tk[0]), lp0=lp[0], lpd=lpd, lps=lp, cf=dcf[c * 15 + j], tgt=ids[p + 1]))
    print(f"{dom}: windows {found} / {len(names)}; positions {sum(1 for q in POS if q['dom'] == dom)}")

def rule_accept(q, rule):
    kind = rule[0]
    if kind == "EXACT": return 1.0 if q["right"] else 0.0
    if q["lpd"] is None: return 0.0
    p1 = math.exp(q["lp0"]); pd = math.exp(q["lpd"])
    if kind == "R1": return 1.0 if q["lpd"] >= q["lp0"] - rule[1] else 0.0
    if kind == "R2":
        k, x = rule[1], rule[2]; lps = q["lps"]
        rank = sum(1 for v in lps if v > q["lpd"])             # 0 = argmax
        return 1.0 if (rank < k and pd >= x * p1) else 0.0
    if kind == "U":                                           # R2 only where the 125B is unsure
        if p1 >= 0.99: return 1.0 if q["right"] else 0.0
        return rule_accept(q, ("R2", rule[1], rule[2]))
    if kind == "R3":
        T = rule[1]; cf = q["cf"]
        if cf != cf or cf <= 0: return 1.0 if q["right"] else 0.0
        Z = sum(math.exp(v / T) for v in q["lps"]); pT = math.exp(q["lpd"] / T) / Z
        return min(1.0, pT / cf)
    if kind == "TYP":
        eps, alpha = rule[1], rule[2]; ps = [math.exp(v) for v in q["lps"]]; Z = sum(ps); ps = [v / Z for v in ps]
        H = -sum(v * math.log(v) for v in ps if v > 0)
        return 1.0 if pd >= min(eps, alpha * math.exp(-H)) else 0.0
    raise ValueError(rule)

RULES = [("EXACT",), ("R1", 0.1), ("R1", 0.25), ("R1", 0.5), ("R1", 0.69), ("R1", 1.0), ("R1", 1.39), ("R1", 2.0),   # 0.69 = ratio 0.5, 1.39 = R-61N's rho 0.25
         ("R2", 2, 0.5), ("R2", 2, 0.25), ("R2", 3, 0.25), ("R2", 3, 0.1), ("R2", 5, 0.1),
         ("U", 2, 0.5), ("U", 2, 0.25), ("U", 3, 0.25), ("U", 5, 0.1),
         ("R3", 0.6), ("R3", 1.0), ("TYP", 0.09, 0.3), ("TYP", 0.3, 0.5)]
def rname(r): return r[0] + ("" if len(r) == 1 else "(" + ", ".join(str(v) for v in r[1:]) + ")")
BANDS = [(1, 3), (4, 7), (8, 11), (12, 15)]
def pct(n, d): return 100.0 * n / d if d else float("nan")

def kept_from_profile(acc_by_depth, B):
    """accepted drafts per check at B rows from per-depth acceptance a_1..a_15; depths > 15 continue the measured
    decay from depths 11 -> 15 (geometric per depth, capped at no growth); the bonus token is not counted."""
    a = dict(acc_by_depth)
    if 11 in a and 15 in a and a[11] > 0: g = min(1.0, (a[15] / a[11]) ** 0.25)
    else: g = 1.0
    last = None; tot = 0.0; surv = 1.0
    for j in range(1, B):
        if j in a and a[j] == a[j]: aj = a[j]; last = aj                     # measured depth
        elif j <= 15 and last is not None: aj = last                        # a depth the chain never reached: carry the last rate
        else: last = (last if last is not None else 0.0) * g; aj = last     # beyond 15: continue the measured decay
        surv *= aj; tot += surv
    return tot

for dom in DOMS:
    P = [q for q in POS if q["dom"] == dom]
    print(f"\n### {dom}: acceptance per tested position (%, all / sure / tie / mid), first-order deviation from greedy (% of positions, all / sure / tie / mid), accepted drafts per check at 16 / 32 rows (today's exact walk: {kept_from_profile({d: pct(sum(q['right'] for q in P if q['depth'] == d), sum(1 for q in P if q['depth'] == d)) / 100 for d in range(1, 16)}, 16):.2f} at 16)")
    print("| rule | accept all | sure | tie | mid | deviation all | sure | tie | mid | accepted at depths 1-3 / 4-7 / 8-11 / 12-15 | drafts kept per check 16 / 32 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for rule in RULES:
        acc = {q_id: rule_accept(q, rule) for q_id, q in enumerate(P)}
        def A(sel): S = [acc[i] for i, q in enumerate(P) if sel(q)]; return pct(sum(S), len(S))
        def Dv(sel): S = [(acc[i] if not q["right"] else 0.0) for i, q in enumerate(P) if sel(q)]; return pct(sum(S), len(S))
        prof = {d: A(lambda q, d=d: q["depth"] == d) / 100 for d in range(1, 16)}
        bands = " / ".join(f"{A(lambda q, lo=lo, hi=hi: lo <= q['depth'] <= hi):.1f}" for lo, hi in BANDS)
        print(f"| {rname(rule)} | {A(lambda q: True):.1f} | {A(lambda q: q['cls'] == 'sure'):.1f} | {A(lambda q: q['cls'] == 'tie'):.1f} | {A(lambda q: q['cls'] == 'mid'):.1f} | "
              f"{Dv(lambda q: True):.2f} | {Dv(lambda q: q['cls'] == 'sure'):.2f} | {Dv(lambda q: q['cls'] == 'tie'):.1f} | {Dv(lambda q: q['cls'] == 'mid'):.1f} | {bands} | "
              f"{kept_from_profile(prof, 16):.2f} / {kept_from_profile(prof, 32):.2f} |")
    n_in = sum(1 for q in P if q["lpd"] is not None); print(f"(drafts inside the stored top-k: {pct(n_in, len(P)):.1f} % of positions; wrong drafts inside it: {pct(sum(1 for q in P if q['lpd'] is not None and not q['right']), sum(1 for q in P if not q['right'])):.0f} % of misses)")
json.dump([{k: v for k, v in q.items() if k != "lps"} for q in POS], open(os.path.expanduser("~/FableGod/relax_positions.json"), "w"))
print("\nwrote ~/FableGod/relax_positions.json")
