#!/usr/bin/env python3 -I
"""drafter_anatomy.py (FableGod, mission 15). CPU, no numpy (parses .npz by hand). Where DFlash2 L1-20 misses, per
drafted position (not per first miss): the HPD-0 walks (~/TrainLab/data/d02b_walk_hpd0_<dom>.npz: per check (k, m, A),
the 15 drafted tokens and their cf) joined with the 125B's own distribution at the row that verifies each draft
(decision_map/out/baseline.json for the 12 code windows; ~/TrainLab/data/d02a/tf_<set>_*.npz for IFEval / multi /
GSM8K: top-32 log-probs per position). Class: sure p1 >= 0.99, tie p1 - p2 <= 0.1, mid otherwise. Token kind from the
model's vocab.json (byte-level BPE: 'Ġ' = space, 'Ċ' = newline)."""
import zipfile, struct, array, json, os, math, re
from collections import defaultdict, Counter
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
        elif descr == "<f2":
            n = len(body) // 2; out[name] = (list(struct.unpack(f"<{n}e", body)), shape)
        else:
            a = array.array({"<i2": "h", "<i4": "i", "<i8": "q", "<f4": "f", "<f8": "d"}[descr]); a.frombytes(body); out[name] = (a, shape)
    return out

vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json")))
id2tok = {v: k for k, v in vocab.items()}
def kind(tid):
    s = id2tok.get(tid, "?")
    if s == "?": return "other"
    core = s.replace("Ġ", " ").replace("Ċ", "\n")
    if core.strip() == "": return "whitespace / newline / indent"
    if re.fullmatch(r"[ ]?[0-9]+", core): return "number"
    if re.fullmatch(r"[ ]?[A-Za-z_][A-Za-z0-9_]*", core): return "word, after a space" if core.startswith(" ") else "word continuation"
    if re.fullmatch(r"[ ]?[^\w\s]+", core): return "punctuation / operator"
    return "other (mixed, non-ascii)"

def classes_from_lp(lp_rows):
    out = []
    for lp in lp_rows:
        p1, p2 = math.exp(lp[0]), math.exp(lp[1])
        out.append("sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid"))
    return out

# the 125B's class per position (row i = the distribution predicting token i+1), per window name
def load_classes(dom):
    C = {}
    if dom == "code":
        base = json.load(open(f"{OUT}/baseline.json"))
        for nm, b in base.items():
            if b["dom"] != "code": continue
            C[nm] = (b["a"], classes_from_lp(b["lp"]))             # answer rows start at a
    else:
        pref = {"ifeval": "tf_ifeval", "multi": "tf_multi", "math": "tf_gsm", "agent": "agent"}[dom]
        for f in os.listdir(f"{TD}/d02a"):
            if not f.startswith(pref): continue
            z = npz(f"{TD}/d02a/{f}"); lp, (n, k) = z["lp"]; gen = z["gen"][0][0]; nid = z["ids"][1][0]
            rows = [lp[r * k:(r + 1) * k] for r in range(n)]
            a = nid - gen                                                  # answer rows start where the prompt ends
            C[f] = (0, classes_from_lp(rows), a)                           # rows cover every position here
    return C

DOMS = ["code", "ifeval", "multi", "math", "agent"]
WALK = {"agent": "d02b_walk_d02a_agent.npz"}                       # agent: the L1-20 walk on the d02a agent windows
BANDS = [(1, 3), (4, 7), (8, 11), (12, 15)]
rows_all = []
for dom in DOMS:
    W = npz(f"{TD}/{WALK.get(dom, f'd02b_walk_hpd0_{dom}.npz')}"); names = W["names"]; C = load_classes(dom)
    found = 0
    for i, nm in enumerate(names):
        key = nm if nm in C else (nm + ".npz" if nm + ".npz" in C else None)
        if key is None: continue
        found += 1
        ids = W[f"{i}_ids"][0]; checks, (nc, _) = W[f"{i}_checks"]
        dcf = W[f"{i}_dcf"][0] if f"{i}_dcf" in W else None; dtok = W[f"{i}_dtok"][0] if f"{i}_dtok" in W else None   # older walks store no cf
        ent = C[key]
        if dom == "code": a, cls = ent; off = a; cl = lambda p: cls[p - off] if 0 <= p - off < len(cls) else None
        else: _, cls, a = ent; cl = lambda p: cls[p] if 0 <= p < len(cls) else None
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            for j in range(min(k + 1, m)):                                  # drafts 0..k-1 right, draft k wrong (if k < m)
                p = A + j
                if p + 1 >= len(ids): break
                right = j < k; c0 = cl(p)
                if c0 is None: continue
                rows_all.append(dict(dom=dom, nm=nm, depth=j + 1, right=right, cls=c0, cf=(dcf[c * 15 + j] if dcf else float("nan")),
                                     tgt=ids[p + 1], draft=(dtok[c * 15 + j] if dtok else -1)))
    print(f"{dom}: windows with classes {found} / {len(names)}; tested draft positions so far {len(rows_all)}")

def pct(n, d): return 100.0 * n / d if d else float("nan")
print("\n### 1. positions tested per class and the drafter's accuracy there, by depth band (right %, n)")
print("| dom | class | share of tested positions | depth 1-3 | 4-7 | 8-11 | 12-15 | all |")
print("|---|---|---|---|---|---|---|---|")
for dom in DOMS:
    R = [r for r in rows_all if r["dom"] == dom]
    for cls in ("sure", "tie", "mid"):
        S = [r for r in R if r["cls"] == cls]
        cells = [f"{dom}", cls, f"{pct(len(S), len(R)):.1f} %"]
        for lo, hi in BANDS:
            B = [r for r in S if lo <= r["depth"] <= hi]
            cells.append(f"{pct(sum(r['right'] for r in B), len(B)):.1f} ({len(B)})")
        cells.append(f"{pct(sum(r['right'] for r in S), len(S)):.1f} ({len(S)})")
        print("| " + " | ".join(cells) + " |")
    cells = [dom, "all", "100 %"]
    for lo, hi in BANDS:
        B = [r for r in R if lo <= r["depth"] <= hi]
        cells.append(f"{pct(sum(r['right'] for r in B), len(B)):.1f} ({len(B)})")
    cells.append(f"{pct(sum(r['right'] for r in R), len(R)):.1f} ({len(R)})")
    print("| " + " | ".join(cells) + " |")

print("\n### 2. miss composition and the reachable gap (per domain): misses by class; per-token agreement now, with sure positions perfect, with sure + mid perfect (ties stay at the measured rate)")
print("| dom | misses sure / tie / mid (share of misses) | a now | a if sure perfect | a if sure + mid perfect | a if everything but ties perfect and ties at 50 % |")
print("|---|---|---|---|---|---|")
for dom in DOMS:
    R = [r for r in rows_all if r["dom"] == dom]; n = len(R)
    miss = Counter(r["cls"] for r in R if not r["right"]); tot = sum(miss.values())
    share = {c: len([r for r in R if r["cls"] == c]) / n for c in ("sure", "tie", "mid")}
    acc = {c: pct(sum(r["right"] for r in R if r["cls"] == c), len([r for r in R if r["cls"] == c])) / 100 for c in ("sure", "tie", "mid")}
    a_now = sum(share[c] * acc[c] for c in share)
    a_sure = share["sure"] * 1.0 + share["tie"] * acc["tie"] + share["mid"] * acc["mid"]
    a_sm = share["sure"] + share["mid"] + share["tie"] * acc["tie"]
    a_tie50 = share["sure"] + share["mid"] + share["tie"] * 0.5
    print(f"| {dom} | {pct(miss['sure'], tot):.0f} / {pct(miss['tie'], tot):.0f} / {pct(miss['mid'], tot):.0f} % of {tot} | {a_now:.3f} | {a_sure:.3f} | {a_sm:.3f} | {a_tie50:.3f} |")

print("\n### 3. the drafter's confidence when it is wrong, by class (median cf; share of misses at cf >= 0.99)")
for dom in DOMS:
    R = [r for r in rows_all if r["dom"] == dom and not r["right"]]
    cells = []
    for cls in ("sure", "tie", "mid"):
        S = sorted(r["cf"] for r in R if r["cls"] == cls and r["cf"] == r["cf"])
        if not S: continue
        cells.append(f"{cls}: median cf {S[len(S) // 2]:.3f}, {pct(sum(1 for v in S if v >= 0.99), len(S)):.0f} % at cf >= 0.99 (n {len(S)})")
    print(f"- {dom}: " + "; ".join(cells))

print("\n### 4. code: the TARGET token's kind at tested positions: share of positions, miss rate, share of misses (and of sure-row misses)")
R = [r for r in rows_all if r["dom"] == "code"]
K = Counter(kind(r["tgt"]) for r in R); M = Counter(kind(r["tgt"]) for r in R if not r["right"]); MS = Counter(kind(r["tgt"]) for r in R if not r["right"] and r["cls"] == "sure")
print("| kind | share of positions | miss rate | share of misses | share of sure-row misses |")
print("|---|---|---|---|---|")
for k_, n_ in K.most_common():
    print(f"| {k_} | {pct(n_, len(R)):.1f} % | {pct(M[k_], n_):.1f} % | {pct(M[k_], sum(M.values())):.0f} % | {pct(MS[k_], max(sum(MS.values()), 1)):.0f} % |")
print("\n### 5. code: sure-row misses by depth: what the drafter put instead (kind of draft vs kind of target), first 12")
n = 0
for r in R:
    if not r["right"] and r["cls"] == "sure":
        print(f"- depth {r['depth']:2d} cf {r['cf']:.3f}: target {id2tok.get(r['tgt'], '?')!r} ({kind(r['tgt'])}) vs draft {id2tok.get(r['draft'], '?')!r}")
        n += 1
        if n >= 12: break
json.dump(rows_all, open(os.path.expanduser("~/FableGod/drafter_anatomy.json"), "w"))
print("\nwrote ~/FableGod/drafter_anatomy.json")
