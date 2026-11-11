#!/usr/bin/env python3 -I
"""knowledge_probe.py (FableGod, mission 24). CPU, no numpy. On the 12 code map windows (the 125B's class and top-8 per
row) joined with DFlash2's HPD-0 walk (draft, cf per tested position): candidate signals where the drafter fails.
  S1 the session's longest in-window match (any length; the token that followed it last time), reported by length;
  S2 the token at the same column of the previous line (code structure);
  S3 the native MTP chain's draft at the same position (d02b_walk_d17e_code.npz, if its windows coincide);
per class (mid; sure outside copies; sure inside copies; tie): coverage, accuracy, the drafter's accuracy on the same
rows, the union, and the signal's precision where it disagrees with the drafter. Then ten mid-row misses decoded."""
import zipfile, struct, array, json, os, re, math
from collections import defaultdict
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
def dec(t): return id2tok.get(t, "?").replace("Ġ", " ").replace("Ċ", "\\n")
base = json.load(open(f"{OUT}/baseline.json"))
def walk_drafts(path):
    W = npz(path); D = {}
    for i, nm in enumerate(W["names"]):
        if f"{i}_dtok" not in W: continue
        checks, (nc, _) = W[f"{i}_checks"]; dtok = W[f"{i}_dtok"][0]; dcf = W[f"{i}_dcf"][0] if f"{i}_dcf" in W else None
        per = len(dtok) // max(nc, 1); g = {}
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            for j in range(min(k + 1, m)): g[A + j] = (dtok[c * per + j], dcf[c * per + j] if dcf else float("nan"), j < k)
        D[nm] = g
    return D
H = walk_drafts(f"{TD}/d02b_walk_hpd0_code.npz")
M = {}
for f in ("d02b_walk_d17e_code.npz", "d02b_walk_d02a_orig_code.npz"):
    if os.path.exists(f"{TD}/{f}"):
        try: M = walk_drafts(f"{TD}/{f}"); M = {k: v for k, v in M.items() if k in base};
        except Exception as e: M = {}
        if M: print("MTP-chain walk used:", f, "windows in common:", len(M)); break
def pct(n, d): return 100.0 * n / d if d else float("nan")
stats = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # (signal, class) -> [n, signal right, drafter right, either, disagree, signal right when disagree]
dump = []
for nm, b in base.items():
    if b["dom"] != "code" or nm not in H: continue
    walkW = npz(f"{TD}/d02b_walk_hpd0_code.npz"); idx = walkW["names"].index(nm); ids = list(walkW[f"{idx}_ids"][0]); a = b["a"]
    local = [dict() for _ in range(17)]
    for j, (tk, lp) in enumerate(zip(b["tk"], b["lp"])):
        p = a + j
        if p not in H[nm] or p + 1 >= len(ids):
            for k in range(1, 17):
                if p - k + 1 >= 0: local[k][tuple(ids[p - k + 1:p + 1])] = p
            continue
        p1, p2 = math.exp(lp[0]), math.exp(lp[1]); cls = "sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid")
        truth = ids[p + 1]; draft, cf, right = H[nm][p]
        # S1 longest match
        L, pred1 = 0, None
        for k in range(16, 0, -1):
            if p - k + 1 < 0: continue
            key = tuple(ids[p - k + 1:p + 1])
            if key in local[k]: L, pred1 = k, ids[local[k][key] + 1]; break
        copy = L >= 8
        klass = "mid" if cls == "mid" else ("tie" if cls == "tie" else ("sure-copy" if copy else "sure-nocopy"))
        sigs = [("S1 match 1-3", pred1 if 1 <= L <= 3 else None), ("S1 match 4-7", pred1 if 4 <= L <= 7 else None), ("S1 match 8+", pred1 if L >= 8 else None)]
        # S2 same column of the previous line
        cur_nl = max((r for r in range(p, -1, -1) if ids[r] in NL), default=None)
        if cur_nl is not None:
            prev_nl = max((r for r in range(cur_nl - 1, -1, -1) if ids[r] in NL), default=None)
            if prev_nl is not None:
                q = prev_nl + (p + 1 - cur_nl)
                sigs.append(("S2 same column, previous line", ids[q] if prev_nl < q < cur_nl else None))
        if nm in M and p in M[nm]: sigs.append(("S3 MTP chain draft", M[nm][p][0]))
        for name, pred in sigs:
            if pred is None: continue
            s = stats[(name, klass)]; s[0] += 1; s[1] += (pred == truth); s[2] += right; s[3] += (right or pred == truth)
            if pred != draft: s[4] += 1; s[5] += (pred == truth)
        stats[("rows", klass)][0] += 1; stats[("rows", klass)][2] += right
        if cls == "mid" and not right and len(dump) < 10:
            dump.append((nm, "".join(dec(t) for t in ids[max(0, p - 11):p + 1]), dec(draft), cf, dec(truth), [(dec(t), round(math.exp(l), 2)) for t, l in zip(tk[:3], lp[:3])]))
        for k in range(1, 17):
            if p - k + 1 >= 0: local[k][tuple(ids[p - k + 1:p + 1])] = p
print("| signal | class | coverage (rows with an opinion / class rows) | signal right % | drafter right % on those rows | either % | disagrees with the drafter % of its rows | signal right when it disagrees % |")
print("|---|---|---|---|---|---|---|---|")
for klass in ("mid", "sure-nocopy", "sure-copy", "tie"):
    n_all = stats[("rows", klass)][0]
    print(f"| drafter alone | {klass} | {n_all} rows | — | {pct(stats[('rows', klass)][2], n_all):.1f} | — | — | — |")
    for name in ("S1 match 1-3", "S1 match 4-7", "S1 match 8+", "S2 same column, previous line", "S3 MTP chain draft"):
        s = stats.get((name, klass))
        if not s or not s[0]: continue
        print(f"| {name} | {klass} | {s[0]} / {n_all} ({pct(s[0], n_all):.0f} %) | {pct(s[1], s[0]):.1f} | {pct(s[2], s[0]):.1f} | {pct(s[3], s[0]):.1f} | {pct(s[4], s[0]):.0f} % | {pct(s[5], s[4]):.1f} |")
print("\nten mid-row misses (context ... | draft (cf) -> truth | the 125B's top-3):")
for nm, ctx, d, cf, t, top in dump:
    print(f"- {ctx!r} | {d!r} ({cf:.2f}) -> {t!r} | {top}")
