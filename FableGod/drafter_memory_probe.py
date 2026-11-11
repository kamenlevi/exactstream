#!/usr/bin/env python3 -I
"""drafter_memory_probe.py (FableGod, mission 17). CPU, no numpy. On DFlash2 L1-20's tested positions (HPD-0 walks),
how much of its error is predictable from history and structure without any new weights:
  (a) an in-context MATCH model (PAQ / branch-predictor style): the longest suffix of the tokens so far that occurred
      earlier in the same window (prompt + generated so far); predict the token that followed the most recent occurrence;
  (b) a cross-window DATASTORE of the 125B's own answers in the other windows of the domain (leave-one-out);
  (c) a CORRECTION CACHE (a misprediction table): at a miss, has the same (last 3 tokens, draft) already been corrected
      earlier in the window, and to the same token? plus its false-positive rate;
  (d) the drafter's own probability of the true token at its misses (tcf): the room for any re-ranking;
  (e) code only: an INDENTATION rule (the previous line's indent, +4 after a ':' line) vs the drafter on indent tokens."""
import zipfile, struct, array, json, os, re
from collections import defaultdict, Counter
TD = os.path.expanduser("~/TrainLab/data")
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
vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json"))); id2tok = {v: k for k, v in vocab.items()}
NEWLINE = {i for i, s in id2tok.items() if s and set(s) == {"Ċ"}}
INDENT = {i: len(s) for i, s in id2tok.items() if s and set(s) == {"Ġ"}}
WIDTH2ID = {w: i for i, w in INDENT.items()}
KMAX = 16
def pct(n, d): return 100.0 * n / d if d else float("nan")

for dom in ("code", "math", "ifeval", "multi"):
    W = npz(f"{TD}/d02b_walk_hpd0_{dom}.npz"); names = W["names"]
    wins = []
    for i, nm in enumerate(names):
        ids = list(W[f"{i}_ids"][0]); gen = W[f"{i}_gen"][0][0]; a = len(ids) - gen
        checks, (nc, _) = W[f"{i}_checks"]; dcf = W[f"{i}_dcf"][0]; dtok = W[f"{i}_dtok"][0]; tcf = W[f"{i}_tcf"][0] if f"{i}_tcf" in W else None
        tested = {}                                          # position p -> (draft, right, cf, tcf, depth)
        for c in range(nc):
            k, m, A = checks[c * 3], checks[c * 3 + 1], checks[c * 3 + 2]
            for j in range(min(k + 1, m)):
                p = A + j
                if p + 1 >= len(ids): continue
                tested[p] = (dtok[c * 15 + j], j < k, dcf[c * 15 + j], (tcf[c * 15 + j] if tcf else None), j + 1)
        wins.append((nm, ids, a, tested))
    # cross-window datastore: answers of the other windows, keyed by the last k tokens -> next token (most recent wins)
    def build_store(skip):
        S = [dict() for _ in range(KMAX + 1)]
        for w, (nm, ids, a, _) in enumerate(wins):
            if w == skip: continue
            for q in range(a, len(ids) - 1):
                for k in range(1, KMAX + 1):
                    if q - k + 1 < 0: break
                    S[k][tuple(ids[q - k + 1:q + 1])] = ids[q + 1]
        return S
    stats = defaultdict(lambda: [0, 0, 0, 0])             # key -> [n, match right, drafter right, either]
    rule = {L: [0, 0] for L in (4, 6, 8)}                  # "use the match when its length >= L, else the drafter": [n, right]
    depth_acc = defaultdict(lambda: [0, 0, 0])            # depth -> [n, drafter right, rule-L8 right]
    corr = [0, 0, 0, 0]                                     # misses, repeat-key seen earlier as a miss, same correction, false positives (key seen, different truth)
    tcfs = []
    ind = [0, 0, 0, 0, 0]                                   # indent positions, rule right, drafter right, drafter misses, rule right at drafter misses
    n_tested = 0
    for w, (nm, ids, a, tested) in enumerate(wins):
        store = build_store(w) if dom == "code" or len(wins) <= 30 else None
        local = [dict() for _ in range(KMAX + 1)]           # in-context tables, built as we scan
        seen_miss = {}                                       # (ctx3, draft) -> truth at an earlier miss
        for q in range(len(ids) - 1):
            # predictions at position q (predicting ids[q+1]) from what is known up to q
            if q in tested:
                n_tested += 1
                d, right, cf, tc, depth = tested[q]; truth = ids[q + 1]
                # (a) in-context longest match
                Lm, pm = 0, None
                for k in range(KMAX, 0, -1):
                    if q - k + 1 < 0: continue
                    key = tuple(ids[q - k + 1:q + 1])
                    if key in local[k]: Lm, pm = k, local[k][key]; break
                # (b) datastore
                Ls, ps = 0, None
                if store is not None:
                    for k in range(KMAX, 0, -1):
                        if q - k + 1 < 0: continue
                        key = tuple(ids[q - k + 1:q + 1])
                        if key in store[k]: Ls, ps = k, store[k][key]; break
                for tag, L, pr in (("ctx", Lm, pm), ("store", Ls, ps)):
                    band = "none" if L == 0 else ("1-3" if L <= 3 else "4-5" if L <= 5 else "6-7" if L <= 7 else "8+")
                    s = stats[(tag, band)]; s[0] += 1; s[1] += (pr == truth); s[2] += right; s[3] += (right or pr == truth)
                    s2 = stats[(tag, "all")]; s2[0] += 1; s2[1] += (pr == truth); s2[2] += right; s2[3] += (right or pr == truth)
                    if not right:
                        s3 = stats[(tag, "at drafter misses")]; s3[0] += 1; s3[1] += (pr == truth); s3[3] += (pr == truth)
                for L in rule:
                    use = Lm >= L
                    rule[L][0] += 1; rule[L][1] += ((pm == truth) if use else right)
                da = depth_acc[depth]; da[0] += 1; da[1] += right; da[2] += ((pm == truth) if Lm >= 8 else right)
                # (c) correction cache
                if not right:
                    corr[0] += 1; key = (tuple(ids[max(0, q - 2):q + 1]), d)
                    if key in seen_miss:
                        corr[1] += 1
                        if seen_miss[key] == truth: corr[2] += 1
                        else: corr[3] += 1
                    seen_miss[key] = truth
                    if tc is not None: tcfs.append(tc)
                # (e) indentation rule
                if dom == "code" and ids[q] in NEWLINE and truth in INDENT:
                    prev_nl = max((r for r in range(q - 1, -1, -1) if ids[r] in NEWLINE), default=None)
                    if prev_nl is not None and prev_nl + 1 < q:
                        prev_ind = INDENT.get(ids[prev_nl + 1], 0)
                        last = next((ids[r] for r in range(q - 1, prev_nl, -1) if ids[r] not in NEWLINE and ids[r] not in INDENT), None)
                        width = prev_ind + (4 if last is not None and id2tok.get(last, "").endswith(":") else 0)
                        pred = WIDTH2ID.get(width)
                        ind[0] += 1; ind[1] += (pred == truth); ind[2] += right; ind[3] += (not right); ind[4] += ((pred == truth) and not right)
            # update in-context tables with position q -> ids[q+1]
            for k in range(1, KMAX + 1):
                if q - k + 1 < 0: break
                local[k][tuple(ids[q - k + 1:q + 1])] = ids[q + 1]
    print(f"\n### {dom}: {n_tested} tested positions, {len(wins)} windows")
    print("| model | match length | share of positions | match right % | drafter right % | either % |")
    print("|---|---|---|---|---|---|")
    for tag in ("ctx", "store"):
        for band in ("none", "1-3", "4-5", "6-7", "8+", "all", "at drafter misses"):
            s = stats.get((tag, band))
            if not s or s[0] == 0: continue
            print(f"| {tag} | {band} | {pct(s[0], n_tested):.1f} % | {pct(s[1], s[0]):.1f} | {pct(s[2], s[0]):.1f} | {pct(s[3], s[0]):.1f} |")
    print("rule 'match if length >= L else drafter': " + "; ".join(f"L {L}: {pct(r, n):.2f} %" for L, (n, r) in rule.items()) + f"; drafter alone {pct(stats[('ctx', 'all')][2], n_tested):.2f} %")
    def kept(prof, B):
        a = dict(prof); g = min(1.0, (a[15] / a[11]) ** 0.25) if a.get(11) and a.get(15) else 1.0
        tot = surv = 1.0; tot = 0.0; last = None
        for j in range(1, B):
            if j in a and a[j] == a[j]: aj = a[j]; last = aj
            elif j <= 15 and last is not None: aj = last
            else: last = (last if last is not None else 0.0) * g; aj = last
            surv *= aj; tot += surv
        return tot
    pd = {d: v[1] / v[0] for d, v in depth_acc.items() if v[0]}; pr = {d: v[2] / v[0] for d, v in depth_acc.items() if v[0]}
    print("per-depth acceptance drafter -> rule L8 (depths 1-15): " + " ".join(f"{100 * pd.get(d, float('nan')):.0f}>{100 * pr.get(d, float('nan')):.0f}" for d in range(1, 16)))
    print(f"accepted drafts per check (profile model): drafter {kept(pd, 16):.2f} / {kept(pd, 32):.2f} at 16 / 32 rows; rule L8 {kept(pr, 16):.2f} / {kept(pr, 32):.2f}; ratio {kept(pr, 16) / kept(pd, 16):.3f} / {kept(pr, 32) / kept(pd, 32):.3f}")
    print(f"correction cache: misses {corr[0]}, repeat (same last-3 + draft seen at an earlier miss) {corr[1]} ({pct(corr[1], corr[0]):.1f} %), same correction {corr[2]} ({pct(corr[2], corr[0]):.1f} % of misses), different truth {corr[3]}")
    if tcfs:
        t = sorted(tcfs); print(f"tcf at misses (drafter's prob of the true token): median {t[len(t) // 2]:.3f}; share >= 0.1: {pct(sum(v >= 0.1 for v in t), len(t)):.0f} %, >= 0.2: {pct(sum(v >= 0.2 for v in t), len(t)):.0f} %, >= 0.3: {pct(sum(v >= 0.3 for v in t), len(t)):.0f} %")
    if dom == "code" and ind[0]:
        print(f"indentation rule on code: {ind[0]} indent-after-newline positions: rule right {pct(ind[1], ind[0]):.1f} %, drafter right {pct(ind[2], ind[0]):.1f} %; drafter misses there {ind[3]}, of which the rule gets {ind[4]}")
