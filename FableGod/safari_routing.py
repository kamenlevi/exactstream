#!/usr/bin/env python3 -I
"""safari_routing.py (FableGod, mission 19). CPU, no numpy. On DAY9's routing capture (decision_map/out/routing/*.npz:
rank-sorted top-10 expert ids per position and layer), the map's baseline (classes, target tokens), floor_rows.json
(flips per row under the cuts) and vocab.json (token kinds):
  H1 expert reuse: routing at a position inside a copied span (longest in-window match >= 8, source s) vs the routing at
     the source; the same for a repeated token id with little context (match < 4); consecutive positions (sticky);
  H4 the working set: do the prompt's experts (and the first 100 answer rows') cover the answer's later expert slots?
  H5 token kinds: flip % under top-2 / top-5 / reuse 12-35 by the target token's kind, and for drafter-unsure rows;
  H2 sure runs: lengths, the target kind at the row that ends a run, P(sure | inside a copy)."""
import zipfile, struct, array, json, os, re
from collections import defaultdict, Counter
OUT = os.path.expanduser("~/FableGod/decision_map/out")
vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json"))); id2tok = {v: k for k, v in vocab.items()}
def kind(tid):
    s = id2tok.get(tid, "?"); core = s.replace("Ġ", " ").replace("Ċ", "\n")
    if core.strip() == "": return "whitespace"
    if re.fullmatch(r"[ ]?[0-9]+", core): return "number"
    if re.fullmatch(r"[ ]?[A-Za-z_][A-Za-z0-9_]*", core): return "word+space" if core.startswith(" ") else "word-cont"
    if re.fullmatch(r"[ ]?[^\w\s]+", core): return "punct"
    return "other"
def load_npy(z, name):
    d = z.read(name); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
    descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
    a = array.array({"<i2": "h", "<i4": "i", "<i8": "q"}[descr]); a.frombytes(body); return a
base = json.load(open(f"{OUT}/baseline.json")); rows = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))
FR = defaultdict(dict)
for r in rows: FR[r["nm"]][r["j"]] = r
GROUPS = [("0-7", 0, 7), ("8-35", 8, 35), ("36-47", 36, 47)]
def pct(n, d): return 100.0 * n / d if d else float("nan")
W = {}
for f in sorted(os.listdir(f"{OUT}/routing")):
    if f not in base: continue
    z = zipfile.ZipFile(f"{OUT}/routing/{f}"); W[f] = dict(dom=base[f]["dom"], a=base[f]["a"], ids=load_npy(z, "ids.npy"), tok=list(load_npy(z, "tokens.npy")))
def top(ids, p, l, k=10): o = (p * 48 + l) * 10; return ids[o:o + k]

# ---- H1 and H4 and H2 per domain ---------------------------------------------------------------------------------
for dom in ("code", "ifeval", "multi"):
    reuse = defaultdict(lambda: [0, 0.0, 0, 0])             # (kind, group) -> [n, sum overlap10, top1 equal, top2 set equal]
    cov = defaultdict(lambda: [0, 0])                       # (what, rank band) -> [slots, covered]
    uniq = [0, 0, 0, 0]                                     # sum over windows/layers of |prompt set|, |answer set|, |answer first100 set|, windows*layers
    runs = []; endkind = Counter(); allkind = Counter(); sure_in = [0, 0]; sure_out = [0, 0]
    for nm, w in W.items():
        if w["dom"] != dom: continue
        ids, tok, a = w["ids"], w["tok"], w["a"]; n1 = len(tok) - 1
        # in-window match tables (built as we scan), the last position of each token id
        local = [dict() for _ in range(17)]; last_tok = {}
        cls = {j: FR[nm][j]["cls"] for j in FR[nm]}
        for p in range(n1):
            if p >= a:
                # longest match ending at p
                L, s = 0, None
                for k in range(16, 0, -1):
                    if p - k + 1 < 0: continue
                    key = tuple(tok[p - k + 1:p + 1])
                    if key in local[k]: L, s = k, local[k][key]; break
                same = last_tok.get(tok[p])
                which = None
                if L >= 8: which, src = "copy>=8", s
                elif 4 <= L <= 7: which, src = "match4-7", s
                elif same is not None: which, src = "same token, match<4", same
                if which:
                    for g, lo, hi in GROUPS + [("all", 0, 47)]:
                        for l in range(lo, hi + 1):
                            A, B = top(ids, p, l), top(ids, src, l)
                            sa, sb = set(A), set(B); r = reuse[(which, g)]
                            r[0] += 1; r[1] += len(sa & sb) / 10; r[2] += (A[0] == B[0]); r[3] += (set(A[:2]) == set(B[:2]))
                if p > 0:
                    for g, lo, hi in GROUPS + [("all", 0, 47)]:
                        for l in range(lo, hi + 1):
                            A, B = top(ids, p, l), top(ids, p - 1, l); r = reuse[("consecutive", g)]
                            r[0] += 1; r[1] += len(set(A) & set(B)) / 10; r[2] += (A[0] == B[0]); r[3] += (set(A[:2]) == set(B[:2]))
                j = p - a
                if j in cls:
                    c = cls[j]; (sure_in if L >= 8 else sure_out)[0] += 1; (sure_in if L >= 8 else sure_out)[1] += (c == "sure")
            for k in range(1, 17):
                if p - k + 1 < 0: break
                local[k][tuple(tok[p - k + 1:p + 1])] = p
            last_tok[tok[p]] = p
        # H4 working sets per layer
        for l in range(48):
            P = set(); F = set(); Aset = set()
            for p in range(0, a): P.update(top(ids, p, l))
            for p in range(a, min(a + 100, n1)): F.update(top(ids, p, l))
            for p in range(a, n1):
                T = top(ids, p, l); Aset.update(T)
                for rk, e in enumerate(T):
                    band = "ranks 0-1" if rk < 2 else "ranks 2-9"
                    c = cov[("prompt covers answer", band)]; c[0] += 1; c[1] += (e in P)
                    if p >= a + 100:
                        c = cov[("first 100 answer rows cover rows 100+", band)]; c[0] += 1; c[1] += (e in F)
                        c = cov[("prompt + first 100 cover rows 100+", band)]; c[0] += 1; c[1] += (e in F or e in P)
            uniq[0] += len(P); uniq[1] += len(Aset); uniq[2] += len(F); uniq[3] += 1
        # H2 runs from the classes
        seq = [cls[j] for j in sorted(cls)]; tks = [base[nm]["tk"][j][0] for j in sorted(cls)]
        run = 0
        for j, c in enumerate(seq):
            allkind[kind(tks[j])] += 1
            if c == "sure": run += 1
            else:
                if run: runs.append(run); endkind[kind(tks[j])] += 1
                run = 0
        if run: runs.append(run)
    print(f"\n### {dom}: H1 expert reuse (overlap of the top-10 set, top-1 equal, top-2 set equal; % by layer group)")
    print("| relation | 0-7 | 8-35 | 36-47 | all |")
    print("|---|---|---|---|---|")
    for which in ("copy>=8", "match4-7", "same token, match<4", "consecutive"):
        cells = []
        for g, _, _ in GROUPS + [("all", 0, 47)]:
            r = reuse.get((which, g))
            cells.append(f"{pct(r[1], r[0]):.0f} / {pct(r[2], r[0]):.0f} / {pct(r[3], r[0]):.0f} (n {r[0] // (48 if g == 'all' else (8 if g == '0-7' else 28 if g == '8-35' else 12))})" if r and r[0] else "—")
        print(f"| {which} | " + " | ".join(cells) + " |")
    print(f"H4 working set ({dom}): unique experts per layer: prompt {uniq[0] / uniq[3]:.0f}, answer {uniq[1] / uniq[3]:.0f}, first 100 answer rows {uniq[2] / uniq[3]:.0f}; coverage of answer slots: "
          + "; ".join(f"{k[0]} [{k[1]}] {pct(v[1], v[0]):.1f} %" for k, v in sorted(cov.items())))
    runs.sort(); n = len(runs)
    print(f"H2 sure runs ({dom}): n {n}, mean {sum(runs) / max(n, 1):.1f}, median {runs[n // 2] if n else 0}, p90 {runs[int(0.9 * n)] if n else 0}, max {runs[-1] if n else 0}; P(sure | inside copy >= 8) {pct(sure_in[1], sure_in[0]):.1f} % (n {sure_in[0]}) vs outside {pct(sure_out[1], sure_out[0]):.1f} % (n {sure_out[0]})")
    tot_e = sum(endkind.values()); tot_a = sum(allkind.values())
    print("  target kind at the row that ends a sure run vs all rows: " + "; ".join(f"{k} {pct(endkind[k], tot_e):.0f} % vs {pct(allkind[k], tot_a):.0f} %" for k in ("whitespace", "punct", "word+space", "word-cont", "number", "other")))

# ---- H5 token kinds under cuts (all three domains pooled, and code) -----------------------------------------------
print("\n### H5 token kinds: flip % under top-2 / top-5 / reuse 12-35 by the target token's kind; drafter-unsure rows (cf < 0.99) under top-2; and 125B-mid rows under top-2")
for dom in ("code", "all"):
    T = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0])      # kind -> [n, f2, f5, fr, n_unsure, f2_unsure, n_mid, f2_mid]
    for r in rows:
        if dom != "all" and r["dom"] != dom: continue
        kd = kind(base[r["nm"]]["tk"][r["j"]][0]); t = T[kd]
        t[0] += 1; t[1] += r["flip"]["topk2"]; t[2] += r["flip"]["topk5"]; t[3] += r["flip"]["moe_reuse_prev_L12_35"]
        if r["cf"] is None or r["cf"] < 0.99: t[4] += 1; t[5] += r["flip"]["topk2"]
        if r["cls"] == "mid": t[6] += 1; t[7] += r["flip"]["topk2"]
    print(f"| {dom}: kind | rows | top-2 | top-5 | reuse 12-35 | drafter-unsure rows: top-2 (n) | 125B-mid rows: top-2 (n) |")
    print("|---|---|---|---|---|---|---|")
    for kd, t in sorted(T.items(), key=lambda kv: -kv[1][0]):
        print(f"| {kd} | {t[0]} | {pct(t[1], t[0]):.2f} | {pct(t[2], t[0]):.2f} | {pct(t[3], t[0]):.2f} | {pct(t[5], t[4]):.1f} ({t[4]}) | {pct(t[7], t[6]):.1f} ({t[6]}) |")
