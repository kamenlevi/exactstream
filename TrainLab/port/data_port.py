"""PORT data (MTP Head, Wed 11-04 20:2x): the MLX trainer's windows, groups and batches (sticky/d09_train.py
load_windows / groups_of / make_batch), the same numpy code with the same RNG calls in the same order, so the PyTorch
trainer sees bitwise-identical batches. Two window sources: the extractor's npz files (as the MLX trainer reads them)
and packed shards (port/pack_shards.py: memory-mapped, for Kaggle datasets)."""
import glob, json, os
import numpy as np

W = 2048


def _window(key, split, h, ids, asst, tk, lp):
    asst = asst.astype(bool)
    n = tk.shape[0]
    match = np.zeros(n + 1, bool)
    match[1:] = tk[:, 0] == ids[1:n + 1]
    anchors = [a for a in range(n) if asst[a] and asst[a + 1]]
    return dict(key=key, split=split, h=h, ids=ids, asst=asst, tk=tk, lp=lp, match=match, anchors=anchors, n=n)


def load_windows_npz(data_dir):
    wins = []
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "*.npz"))):
        z = np.load(p)
        wins.append(_window(os.path.basename(p)[:-4], str(z["split"]), z["h"], z["ids"], z["asst"], z["tk"], z["lp"]))
    return wins


def load_windows_shards(shard_dir):
    """packed shards (pack_shards.py): index.json + per-shard flat files, memory-mapped (h is read on demand)"""
    d = os.path.expanduser(shard_dir)
    idx = json.load(open(os.path.join(d, "index.json")))
    D = idx["hidden"]
    maps = {}
    wins = []
    for e in idx["windows"]:
        s = e["shard"]
        if s not in maps:
            f = lambda name, dt: np.memmap(os.path.join(d, f"shard{s:04d}.{name}"), dtype=dt, mode="r")
            maps[s] = dict(h=f("h", np.uint16), ids=f("ids", np.int32), asst=f("asst", np.uint8), tk=f("tk", np.int32),
                           lp=f("lp", np.float16))
        m = maps[s]
        n, r0, t0 = e["n"], e["row0"], e["tok0"]
        h = m["h"][r0 * D:(r0 + n) * D].reshape(n, D)
        tk = m["tk"][r0 * 32:(r0 + n) * 32].reshape(n, 32)
        lp = m["lp"][r0 * 32:(r0 + n) * 32].reshape(n, 32)
        ids = np.asarray(m["ids"][t0:t0 + n + 1])
        asst = np.asarray(m["asst"][t0:t0 + n + 1])
        wins.append(_window(e["key"], e["split"], h, ids, asst, np.asarray(tk), np.asarray(lp)))
    return wins


def load_windows(src):
    src = os.path.expanduser(src)
    return load_windows_shards(src) if os.path.exists(os.path.join(src, "index.json")) else load_windows_npz(src)


def groups_of(win, A, span=None):
    """the MLX trainer's grouping: up to A consecutive anchors within `span` positions (2A, as the trainer; a sparser
    anchor set needs 2A / frac to fill its groups, since every group computes A anchor slots)"""
    span = span or 2 * A
    out, cur = [], []
    for a in win["anchors"]:
        if cur and (len(cur) == A or a - cur[0] >= span):
            out.append((win, cur)); cur = []
        cur.append(a)
    if cur:
        out.append((win, cur))
    return out


def make_batch(items, A, BS, MASK, pmask=0.0, rng=None):
    """numpy arrays of one micro-batch (the MLX trainer's make_batch; rng = its PM_RNG, used only when pmask > 0)"""
    KD = BS - 1
    S = len(items)
    D = items[0][0]["h"].shape[1]
    U = -(-max(1, max(g[-1] - max(0, g[0] - (W - 1)) for _, g in items)) // 128) * 128   # padded rows are masked
    hc = np.zeros((S, U, D), np.uint16); upos = np.full((S, U), -1, np.int32)
    anc = np.zeros((S, A), np.int32); tok = np.full((S, A, BS), MASK, np.int32)
    B = S * A
    tkl = np.zeros((B, KD, 32), np.int32); lpl = np.zeros((B, KD, 32), np.float32)
    y = np.zeros((B, KD), np.int32); pred = np.zeros((B, KD), np.int32); w = np.zeros((B, KD), np.float32)
    for s, (win, g) in enumerate(items):
        lo, hi = max(0, g[0] - (W - 1)), g[-1]
        hc[s, :hi - lo] = win["h"][lo:hi]; upos[s, :hi - lo] = np.arange(lo, hi)
        for k in range(A):
            a = g[k] if k < len(g) else g[0]
            anc[s, k] = a; tok[s, k, 0] = win["ids"][a]
            if k >= len(g):
                continue
            b = s * A + k
            for j in range(1, KD + 1):
                r = a + j - 1
                if r > win["n"] - 1 or not win["asst"][a + j] or (j > 1 and not win["match"][a + j - 1]):
                    break
                tkl[b, j - 1] = win["tk"][r]; lpl[b, j - 1] = win["lp"][r]; y[b, j - 1] = win["tk"][r, 0]
                pred[b, j - 1] = win["ids"][a] if j == 1 else win["tk"][r - 1, 0]
                w[b, j - 1] = 1.0
            if pmask > 0 and rng.random() < pmask:
                valid = np.nonzero(w[b] > 0)[0]
                if valid.size > 1:
                    show = valid[rng.random(valid.size) < rng.uniform(0.1, 0.6)]
                    tok[s, k, show + 1] = y[b, show]
                    w[b, show] = 0.0
    return dict(hc=hc, upos=upos, anc=anc, tok=tok, tkl=tkl, lpl=lpl, y=y, pred=pred, w=w)


def to_torch(b, device):
    """numpy batch -> torch (hc: bf16 bits -> fp32, exact)"""
    import torch
    hc = (b["hc"].astype(np.uint32) << 16).view(np.float32)
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)
    return dict(hc=t(hc, torch.float32), upos=t(b["upos"], torch.long), anc=t(b["anc"], torch.long),
                tok=t(b["tok"], torch.long), tkl=t(b["tkl"], torch.long), lpl=t(b["lpl"], torch.float32),
                y=t(b["y"], torch.long), pred=t(b["pred"], torch.long), w=t(b["w"], torch.float32))


def subsample_anchors(wins, frac, seed=1):
    """keep each train window's anchor with probability frac (the Kaggle cost lever: compute ~ anchors; DFlash's own
    recipe samples anchors). Seeded per window key, so independent of window order; frac >= 1 changes nothing (no RNG
    calls: the batches stay bitwise the MLX trainer's)."""
    if frac >= 1.0:
        return wins
    import zlib
    for w_ in wins:
        if w_["split"] != "train" or not w_["anchors"]:
            continue
        r = np.random.default_rng([seed, zlib.crc32(w_["key"].encode())])
        keep = r.random(len(w_["anchors"])) < frac
        w_["anchors"] = [a for a, k in zip(w_["anchors"], keep) if k]
    return wins


def split_groups(wins, A, val_n=24, seed=0, anchor_frac=1.0):
    """train / val groups and the trainer's rng, consumed exactly as the MLX trainer does before its first step"""
    wins = subsample_anchors(wins, anchor_frac)
    span = int(round(2 * A / min(1.0, anchor_frac)))
    tr = [g for w_ in wins if w_["split"] == "train" for g in groups_of(w_, A, span)]
    va = [g for w_ in wins if w_["split"] == "val" for g in groups_of(w_, A)]
    rng = np.random.default_rng(seed)
    va = [va[i] for i in sorted(rng.choice(len(va), min(len(va), val_n), replace=False))] if va else []
    return tr, va, rng


def synthetic_windows(n_win=12, vocab=4096, hidden=256, seed=0, val=2):
    """random windows in the extractor's format (prompt then assistant tokens; the 'greedy' label equals the text on
    ~90 % of assistant tokens), for the trainer's mechanics tests"""
    g = np.random.default_rng(seed)
    wins = []
    for i in range(n_win):
        n = int(g.integers(300, 900))
        ids = g.integers(0, vocab - 2, n + 1).astype(np.int32)
        gen = int(g.integers(50, n // 2))
        asst = np.zeros(n + 1, np.uint8); asst[gen:] = 1
        tk = g.integers(0, vocab - 2, (n, 32)).astype(np.int32)
        keep = g.random(n) < 0.9
        tk[keep, 0] = ids[1:][keep]
        lp = np.sort(g.normal(-3, 2, (n, 32)).astype(np.float16), axis=1)[:, ::-1].copy()
        h = (g.normal(0, 1, (n, hidden)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        wins.append(_window(f"syn{i:03d}", "val" if i < val else "train", h, ids, asst, tk, lp))
    return wins
