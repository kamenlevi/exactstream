#!/usr/bin/env python3 -I
"""selfcert_probe.py (FableGod, mission 21). CPU, no numpy. Can the 125B certify its own rows mid-pass? Per tap layer
(5 / 15 / 25 / 35 / 45: the mean of the 4 HC streams, fp16, stored per position in ~/TrainLab/data/d02a/tf_<set>_*.npz
with each row's top-32 log-probs), a logistic probe on the row's own state predicting "the row ends sure" (p1 >= 0.99),
trained on the first part of the windows and judged on held-out windows; coverage at <= 0.1 % (and 0.5 %) false
certificates, against the drafter's cf (from the HPD-0 walk) and the oracle (the sure share); a fusion probe
(layer score + cf + copy state + the draft's token kind). Code runs the same way once EARLYTAP's taps exist."""
import zipfile, struct, array, json, os, re, math, random, sys, time
from operator import mul
TD = os.path.expanduser("~/TrainLab/data"); random.seed(1)
TAPS = [5, 15, 25, 35, 45]; D = 2560
def npz_raw(path, names):
    z = zipfile.ZipFile(path); out = {}
    for i in z.infolist():
        nm = i.filename[:-4]
        if nm not in names: continue
        d = z.read(i.filename); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
        descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
        shape = tuple(int(x) for x in re.findall(r"\d+", re.search(r"'shape': \(([^)]*)\)", hdr).group(1)))
        out[nm] = (descr, shape, body)
    return out
def walk_cf(dom):
    """row position -> drafter cf (the draft the row verifies), per window name (dm_gate's rule), plus the draft token."""
    z = zipfile.ZipFile(f"{TD}/d02b_walk_hpd0_{dom}.npz"); raw = {}
    for i in z.infolist():
        nm = i.filename[:-4]; d = z.read(i.filename); hl = struct.unpack("<H", d[8:10])[0]; hdr = d[10:10 + hl].decode("latin1"); body = d[10 + hl:]
        descr = re.search(r"'descr': '([^']+)'", hdr).group(1)
        if descr.startswith("<U"):
            w = int(descr[2:]); n = int(re.search(r"'shape': \((\d+)", hdr).group(1)); raw[nm] = [body[j * w * 4:(j + 1) * w * 4].decode("utf-32-le").rstrip("\x00") for j in range(n)]
        else:
            a = array.array({"<i4": "i", "<i8": "q", "<f4": "f"}[descr]); a.frombytes(body); raw[nm] = a
    G = {}
    for i, nm in enumerate(raw["names"]):
        ck = raw[f"{i}_checks"]; dcf = raw[f"{i}_dcf"]; dtok = raw[f"{i}_dtok"]; g = {}
        for c in range(len(ck) // 3):
            k, m, A = ck[c * 3], ck[c * 3 + 1], ck[c * 3 + 2]
            for j in range(min(k + 1, m)): g[A + j] = (dcf[c * 15 + j], dtok[c * 15 + j])
        G[nm] = g
    return G
vocab = json.load(open(os.path.expanduser("~/BigMoE/models/bare/vocab.json"))); id2tok = {v: k for k, v in vocab.items()}
KINDS = ["whitespace", "number", "word+space", "word-cont", "punct", "other"]
def kind(tid):
    s = id2tok.get(tid, "?"); core = s.replace("Ġ", " ").replace("Ċ", "\n")
    if core.strip() == "": return 0
    if re.fullmatch(r"[ ]?[0-9]+", core): return 1
    if re.fullmatch(r"[ ]?[A-Za-z_][A-Za-z0-9_]*", core): return 2 if core.startswith(" ") else 3
    if re.fullmatch(r"[ ]?[^\w\s]+", core): return 4
    return 5

def load(dom, pref, n_train_windows, max_rows_per_window=120):
    files = sorted(f for f in os.listdir(f"{TD}/d02a") if f.startswith(pref)); G = walk_cf(dom)
    train, test = [], []
    for wi, f in enumerate(files):
        r = npz_raw(f"{TD}/d02a/{f}", {"f", "lp", "ids", "gen"})
        n, five, dd = r["f"][1]; lpn, k = r["lp"][1]
        ids = array.array("i"); ids.frombytes(r["ids"][2]); gen = struct.unpack("<q", r["gen"][2][:8])[0]; a = len(ids) - gen
        g = G.get(f, G.get(f[:-4], {}))
        rows_here = [p for p in range(a, min(n, len(ids) - 1)) if p in g]
        random.shuffle(rows_here); rows_here = rows_here[:max_rows_per_window]
        fb, lpb = r["f"][2], r["lp"][2]
        # copy state: longest in-window match >= 8 ending at p
        local = [dict() for _ in range(9)]; copy = {}
        for p in range(len(ids) - 1):
            L = 0
            for kk in range(8, 0, -1):
                if p - kk + 1 < 0: continue
                key = tuple(ids[p - kk + 1:p + 1])
                if key in local[kk]: L = kk; break
            copy[p] = L >= 8
            for kk in range(1, 9):
                if p - kk + 1 < 0: break
                local[kk][tuple(ids[p - kk + 1:p + 1])] = p
        for p in rows_here:
            lp0 = struct.unpack_from("<e", lpb, p * k * 2)[0]; sure = math.exp(lp0) >= 0.99
            x = array.array("f", struct.unpack_from(f"<{five * dd}e", fb, p * five * dd * 2))
            cf, dtok = g[p]
            (train if wi < n_train_windows else test).append((x, sure, cf, 1 if copy[p] else 0, kind(dtok)))
    return train, test, len(files)

def train_logistic(rows, layer_idx, epochs=6, lr=0.02, l2=1e-4):
    off = layer_idx * D; n = len(rows)
    mu = [0.0] * D; sd = [0.0] * D
    for x, *_ in rows[:800]:
        for i in range(D): mu[i] += x[off + i]
    mu = [m / min(n, 800) for m in mu]
    for x, *_ in rows[:800]:
        for i in range(D): sd[i] += (x[off + i] - mu[i]) ** 2
    sd = [math.sqrt(s / min(n, 800)) + 1e-3 for s in sd]
    def feat(x): return [(x[off + i] - mu[i]) / sd[i] for i in range(D)]
    w = [0.0] * D; b = 0.0; wa = [0.0] * D; ba = 0.0; steps = 0
    order = list(range(n))
    for ep in range(epochs):
        random.shuffle(order)
        for idx in order:
            x, y, *_ = rows[idx]; z = feat(x); s = sum(map(mul, w, z)) + b; p = 1 / (1 + math.exp(-max(-30, min(30, s))))
            gcoef = (p - (1.0 if y else 0.0)) * lr
            w = [wi - gcoef * zi - lr * l2 * wi for wi, zi in zip(w, z)]; b -= gcoef
            steps += 1
            if ep >= epochs - 2:
                wa = [a_ + wi for a_, wi in zip(wa, w)]; ba += b
        lr *= 0.7
    cnt = max(1, sum(1 for ep in range(epochs) if ep >= epochs - 2) * n)
    wa = [a_ / cnt for a_ in wa]; ba /= cnt
    return lambda x: sum(map(mul, wa, feat(x))) + ba

def coverage(scores, labels, fp_rate):
    """largest prefix of rows sorted by score (desc) whose share of not-sure rows <= fp_rate: coverage of all rows."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i]); bad = 0; best = 0
    for i, idx in enumerate(order, 1):
        bad += (not labels[idx])
        if bad <= fp_rate * i: best = i
    return 100.0 * best / len(scores)

def run(dom, pref, n_train):
    t0 = time.time(); train, test, nf = load(dom, pref, n_train)
    print(f"\n### {dom}: {nf} windows ({n_train} train / {nf - n_train} held out); rows with a drafter cf: train {len(train)}, test {len(test)}; sure share test {100 * sum(r[1] for r in test) / len(test):.1f} % (the oracle); load {time.time() - t0:.0f} s")
    ytest = [r[1] for r in test]
    cfs = [r[2] for r in test]
    print(f"baselines on the test rows: drafter cf coverage at <= 0.1 % false {coverage(cfs, ytest, 0.001):.1f} %, at <= 0.5 % {coverage(cfs, ytest, 0.005):.1f} %, at <= 2 % {coverage(cfs, ytest, 0.02):.1f} %")
    print("| probe | coverage at <= 0.1 % false | <= 0.5 % | <= 2 % | test rows scored sure-first (AUC-like: share of (sure, not-sure) pairs ordered right) |")
    print("|---|---|---|---|---|")
    scores_by_layer = {}
    for li, L in enumerate(TAPS):
        t1 = time.time(); f = train_logistic(train, li); sc = [f(r[0]) for r in test]; scores_by_layer[L] = sc
        pos = [s for s, y in zip(sc, ytest) if y]; neg = [s for s, y in zip(sc, ytest) if not y]
        neg_s = sorted(neg); auc = sum(sum(1 for v in neg_s if v < s) for s in pos[:400]) / max(1, min(len(pos), 400) * len(neg_s))
        print(f"| layer {L} state | {coverage(sc, ytest, 0.001):.1f} % | {coverage(sc, ytest, 0.005):.1f} % | {coverage(sc, ytest, 0.02):.1f} % | {100 * auc:.1f} % ({time.time() - t1:.0f} s) |")
    # fusion: layer-35 score + cf + copy + draft kind (trained on the train rows' in-sample scores)
    f35 = train_logistic(train, 3)
    def fvec(r, s35): return [s35, r[2], 1.0 if r[2] >= 0.999 else 0.0, float(r[3])] + [1.0 if r[4] == kk else 0.0 for kk in range(6)]
    tr = [(fvec(r, f35(r[0])), r[1]) for r in train]; te = [(fvec(r, s), r[1]) for r, s in zip(test, scores_by_layer[35])]
    w = [0.0] * 10; b = 0.0; lr = 0.05
    for ep in range(30):
        random.shuffle(tr)
        for z, y in tr:
            s = sum(map(mul, w, z)) + b; p = 1 / (1 + math.exp(-max(-30, min(30, s)))); g = (p - (1.0 if y else 0.0)) * lr
            w = [wi - g * zi for wi, zi in zip(w, z)]; b -= g
        lr *= 0.9
    sc = [sum(map(mul, w, z)) + b for z, _ in te]
    print(f"fusion (layer-35 score + cf + copy state + draft kind): coverage at <= 0.1 % false {coverage(sc, ytest, 0.001):.1f} %, <= 0.5 % {coverage(sc, ytest, 0.005):.1f} %, <= 2 % {coverage(sc, ytest, 0.02):.1f} %")
    sc2 = [r[2] + (1.0 if r[3] else 0.0) for r in test]
    print(f"cf + copy state alone (no layer): <= 0.1 % {coverage(sc2, ytest, 0.001):.1f} %, <= 0.5 % {coverage(sc2, ytest, 0.005):.1f} %")

if __name__ == "__main__":
    which = sys.argv[1:] or ["math", "ifeval"]
    if "math" in which: run("math", "tf_gsm", 45)
    if "ifeval" in which: run("ifeval", "tf_ifeval", 22)
    if "multi" in which: run("multi", "tf_multi", 18)
