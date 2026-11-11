#!/usr/bin/env python3 -I
"""depth_wall.py (FableGod, mission 23). CPU, no numpy. Takes DFlash2 L1-20's depth decay apart on the real walks:
16 rows (HPD-0 = base16) and 32 rows (MATCH's base32), code first, then math at 16. Per tested position (every draft up
to the first miss): depth, right, the drafter's cf, the draft and target tokens, the 125B's class (12 map windows via
baseline.json), the copy state (in-window match >= 8 on the true tokens), the target's kind, the tokens since the last
newline, and whether the block's anchor follows a newline. Then, per depth band: accuracy, the mix of positions
(class, copy, cf band), accuracy WITHIN cf bands (stale taps vs rising uncertainty), misses by kind, and the
line-structure effect."""
import zipfile, struct, array, json, os, re, math
from collections import defaultdict, Counter
TD = os.path.expanduser("~/TrainLab/data"); OUT = os.path.expanduser("~/FableGod/decision_map/out")
def npz(path):
    z = zipfile.ZipFile(path); out = {}
    for i in z.infolist():
        d = z.read(i.filename); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
        descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
        shape = tuple(int(x) for x in re.findall(r"\d+", re.search(r"'shape': \(([^)]*)\)", hdr).group(1)))
        nm = i.filename[:-4]
        if descr.startswith("<U"):
            w = int(descr[2:]); n = shape[0] if shape else 1
            out[nm] = [body[j * w * 4:(j + 1) * w * 4].decode("utf-32-le").rstrip("\x00") for j in range(n)]
        elif descr == "<f2": n = len(body) // 2; out[nm] = (list(struct.unpack(f"<{n}e", body)), shape)
        else:
            a = array.array({"<i2": "h", "<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d"}[descr]); a.frombytes(body); out[nm] = (a, shape)
    return out
vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json"))); id2tok = {v: k for k, v in vocab.items()}
NL = {i for i, s in id2tok.items() if s and set(s) == {"Ċ"}}
def kind(tid):
    s = id2tok.get(tid, "?"); core = s.replace("Ġ", " ").replace("Ċ", "\n")
    if core.strip() == "": return "ws"
    if re.fullmatch(r"[ ]?[0-9]+", core): return "num"
    if re.fullmatch(r"[ ]?[A-Za-z_][A-Za-z0-9_]*", core): return "word" if core.startswith(" ") else "cont"
    if re.fullmatch(r"[ ]?[^\w\s]+", core): return "punct"
    return "other"
base = json.load(open(f"{OUT}/baseline.json"))
CL = {}
for nm, b in base.items():
    cls = []
    for lp in b["lp"]:
        p1, p2 = math.exp(lp[0]), math.exp(lp[1]); cls.append("sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid"))
    CL[nm] = (b["a"], cls)
def cfband(c): return ">=.999" if c >= 0.999 else ".99-.999" if c >= 0.99 else ".9-.99" if c >= 0.9 else "<.9"
def pct(n, d): return 100.0 * n / d if d else float("nan")

def positions(walkfile, nrows):
    W = npz(f"{TD}/{walkfile}"); P = []
    for i, nm in enumerate(W["names"]):
        if f"{i}_dcf" not in W: continue
        ids = list(W[f"{i}_ids"][0]); checks, (nc, _) = W[f"{i}_checks"]; dcf = W[f"{i}_dcf"][0]; dtok = W[f"{i}_dtok"][0]
        per = len(dcf) // max(nc, 1)                           # drafts stored per check (15 or 31)
        a_cls = CL.get(nm)
        # copy state and line distance on the true tokens
        local = [dict() for _ in range(9)]; copy = [False] * len(ids); dist = [0] * len(ids); since = 10 ** 6
        for p in range(len(ids)):
            L = 0
            for k in range(8, 0, -1):
                if p - k + 1 < 0: continue
                if tuple(ids[p - k + 1:p + 1]) in local[k]: L = k; break
            copy[p] = L >= 8; since = 0 if ids[p] in NL else since + 1; dist[p] = since
            for k in range(1, 9):
                if p - k + 1 < 0: break
                local[k][tuple(ids[p - k + 1:p + 1])] = p
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            anchor_nl = A > 0 and ids[A] in NL
            for j in range(min(k + 1, m)):
                p = A + j
                if p + 1 >= len(ids): break
                cls = None
                if a_cls and 0 <= p - a_cls[0] < len(a_cls[1]): cls = a_cls[1][p - a_cls[0]]
                P.append(dict(d=j + 1, right=j < k, cf=dcf[c * per + j], cls=cls, copy=copy[p], kind=kind(ids[p + 1]), dist=dist[p], anl=anchor_nl))
    return P

def report(P, bands, title):
    print(f"\n### {title}: {len(P)} tested positions")
    print("| depth | n | right % | sure / mid / tie share (of classed) | right % sure / mid | copy share | right % in / out of copies | cf<.9 share | right % at cf >=.999 / .99-.999 / .9-.99 / <.9 | misses by kind word / punct / ws / num / cont | right % dist 0-3 / 4-8 / 9+ | right % anchor at line start / not |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lo, hi in bands:
        S = [q for q in P if lo <= q["d"] <= hi]
        if not S: continue
        def acc(sel): T = [q for q in S if sel(q)]; return pct(sum(q["right"] for q in T), len(T))
        C = [q for q in S if q["cls"]]
        sh = lambda c: pct(sum(1 for q in C if q["cls"] == c), len(C))
        miss = Counter(q["kind"] for q in S if not q["right"]); tm = sum(miss.values())
        print(f"| {lo}-{hi} | {len(S)} | {acc(lambda q: True):.1f} | {sh('sure'):.0f} / {sh('mid'):.0f} / {sh('tie'):.0f} | {acc(lambda q: q['cls'] == 'sure'):.1f} / {acc(lambda q: q['cls'] == 'mid'):.1f} | {pct(sum(q['copy'] for q in S), len(S)):.0f} % | {acc(lambda q: q['copy']):.1f} / {acc(lambda q: not q['copy']):.1f} | {pct(sum(q['cf'] < 0.9 for q in S), len(S)):.0f} % | "
              f"{acc(lambda q: q['cf'] >= 0.999):.1f} / {acc(lambda q: 0.99 <= q['cf'] < 0.999):.1f} / {acc(lambda q: 0.9 <= q['cf'] < 0.99):.1f} / {acc(lambda q: q['cf'] < 0.9):.1f} | "
              f"{pct(miss['word'], tm):.0f} / {pct(miss['punct'], tm):.0f} / {pct(miss['ws'], tm):.0f} / {pct(miss['num'], tm):.0f} / {pct(miss['cont'], tm):.0f} | "
              f"{acc(lambda q: q['dist'] <= 3):.1f} / {acc(lambda q: 4 <= q['dist'] <= 8):.1f} / {acc(lambda q: q['dist'] >= 9):.1f} | {acc(lambda q: q['anl']):.1f} / {acc(lambda q: not q['anl']):.1f} |")
    # the decomposition of the deep drop: accuracy if the shallow cf mix held
    sh_, dp_ = [q for q in P if bands[0][0] <= q["d"] <= bands[0][1]], [q for q in P if bands[-1][0] <= q["d"] <= bands[-1][1]]
    if sh_ and dp_:
        bands_cf = [">=.999", ".99-.999", ".9-.99", "<.9"]
        mix_sh = {b: pct(sum(1 for q in sh_ if cfband(q["cf"]) == b), len(sh_)) / 100 for b in bands_cf}
        acc_dp = {b: pct(sum(q["right"] for q in dp_ if cfband(q["cf"]) == b), sum(1 for q in dp_ if cfband(q["cf"]) == b)) / 100 for b in bands_cf}
        acc_sh = {b: pct(sum(q["right"] for q in sh_ if cfband(q["cf"]) == b), sum(1 for q in sh_ if cfband(q["cf"]) == b)) / 100 for b in bands_cf}
        mix_dp = {b: pct(sum(1 for q in dp_ if cfband(q["cf"]) == b), len(dp_)) / 100 for b in bands_cf}
        a_sh = sum(mix_sh[b] * acc_sh[b] for b in bands_cf); a_dp = sum(mix_dp[b] * acc_dp[b] for b in bands_cf)
        a_mix = sum(mix_dp[b] * acc_sh[b] for b in bands_cf); a_stale = sum(mix_sh[b] * acc_dp[b] for b in bands_cf)
        print(f"decomposition shallow band {bands[0]} -> deep band {bands[-1]}: accuracy {100 * a_sh:.1f} -> {100 * a_dp:.1f}; with the deep cf MIX but shallow per-band accuracy {100 * a_mix:.1f} (the drafter's rising uncertainty); with the shallow mix but deep per-band accuracy {100 * a_stale:.1f} (stale taps / miscalibration). cf-band shares shallow -> deep: " + ", ".join(f"{b} {100 * mix_sh[b]:.0f}->{100 * mix_dp[b]:.0f} %" for b in bands_cf))

P16 = positions("d02b_walk_hpd0_code.npz", 16); report(P16, [(1, 4), (5, 8), (9, 12), (13, 15)], "code, 16 rows (HPD-0 = base16)")
try:
    P32 = positions("d02b_walk_m_base32_code.npz", 32); report(P32, [(1, 8), (9, 16), (17, 24), (25, 31)], "code, 32 rows (MATCH base32)")
except Exception as e: print("32-row walk not readable:", e)
PM = positions("d02b_walk_hpd0_math.npz", 16); report(PM, [(1, 4), (5, 8), (9, 12), (13, 15)], "math, 16 rows")
