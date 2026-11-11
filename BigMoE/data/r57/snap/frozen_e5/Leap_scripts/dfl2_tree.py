"""Leap tree bound (Sat 2026-10-24, DFL2D_PREREG.md part 1): DFlash2's candidate lattice at EVERY anchor of L2's code /
math sequences (features from dfl2_extract.py), then replays with exact greedy verify and tokens per pass counted
correctly (k accepted drafts + the target's bonus / correction token whenever a token follows):
- chain-8: the selector's official greedy path (select() on the same hidden states, as propose());
- tree-16 / tree-32: rows = anchor + 15 / 31 nodes, best-first by cumulative selector probability (EAGLE-2 style);
  a node's probability given its parent = softmax over its 16 candidates of unary + edge(parent token, candidate);
- ceiling: any tree over the top-16 candidates (the true token is among them at every depth so far).
Cache handling as dfl2_eval.every_anchor (one committed row per call), so every policy sees the same drafter context.
Lattice per anchor: top-16 candidates and unary scores per depth; the edge scores from the anchor (depth 1) and between
the 16 x 16 candidate pairs of consecutive depths, computed exactly as CandidateSelector.select.
Usage: [BigMoE venv] dfl2_tree.py DIR [DIR ...]   -> ~/Leap/data/tree_result.txt"""
import glob, heapq, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_third_party"))
import dflash_model_mlx as D

class Target(nn.Module):
    """only the 125B's input embedding and LM head (4-bit affine, group 64), which draft.bind() needs"""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)

def load_target():
    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    t = Target()
    t.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                   + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    mx.eval(t.parameters())
    return t

draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(load_target())
SEL, MASK, BS = draft.candidate_selector, int(draft.config.mask_token_id), 8

def lattices(f, ids, off, gen):
    cache = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, out = start, []
    for a in range(gen, off + len(ids) - 1):
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        block = mx.array([[int(ids[a - off])] + [MASK] * (BS - 1)])
        hid = draft.hidden_states(block, ctx, cache, logits_start=1)                  # (1, 7, D), as propose()
        logits = draft.compute_logits(hid)
        path, cand, _ = SEL.select(hid, logits, block[:, 0], 0.0)
        un = mx.take_along_axis(logits, cand, axis=-1)[0]                              # (7, 16)
        hp = SEL.hidden_projection(hid)                                                # (1, 7, R)
        e1 = mx.sum(SEL.predecessor_codebook(block[:, 0])[:, None] * hp[:, 0, None]
                    * SEL.successor_codebook(cand[:, 0]), axis=-1)[0]                  # (16,): from the anchor
        E = mx.sum(SEL.predecessor_codebook(cand[0, :-1])[:, :, None, :] * hp[0, 1:, None, None, :]
                   * SEL.successor_codebook(cand[0, 1:])[:, None, :, :], axis=-1)      # (6, 16 pred, 16 cand)
        mx.eval(path, cand, un, e1, E)
        fed = a
        out.append((np.array(path[0]), np.array(cand[0]), np.array(un.astype(mx.float32)),
                    np.array(e1.astype(mx.float32)), np.array(E.astype(mx.float32))))
    return out

def logsm(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())

def tree_paths(lat, N):
    """best-first tree of N nodes (paths of candidate indices) by cumulative log-probability"""
    _, cand, un, e1, E = lat
    heap = [(-lp, (i,)) for i, lp in enumerate(logsm(un[0] + e1))]
    heapq.heapify(heap)
    tree = set()
    while heap and len(tree) < N:
        nlp, p = heapq.heappop(heap)
        tree.add(p)
        d = len(p)
        if d < 7:
            for i, lp in enumerate(logsm(un[d] + E[d - 1][p[-1]])):
                heapq.heappush(heap, (nlp - lp, p + (i,)))
    return tree

def accepted(policy, lat, true):
    path, cand = lat[0], lat[1]
    if policy == "chain":
        k = 0
        while k < len(true) and path[k] == true[k]:
            k += 1
        return k
    idx = []                                  # index of the true token among each depth's candidates (until missing)
    for j, t in enumerate(true):
        w = np.nonzero(cand[j] == t)[0]
        if not len(w):
            break
        idx.append(int(w[0]))
    if policy == "ceiling":
        return len(idx)
    if policy == "lchain":                    # greedy through the lattice: must match the official path
        k, prev = 0, None
        for j in range(len(true)):
            s = lat[2][j] + (lat[3] if j == 0 else lat[4][j - 1][prev])
            i = int(np.argmax(s))
            if j >= len(idx) or i != idx[j]:
                break
            k, prev = k + 1, i
        return k
    tree = tree_paths(lat, policy)
    k = 0
    while k < len(idx) and tuple(idx[:k + 1]) in tree:
        k += 1
    return k

POL = ["chain", "lchain", 15, 31, "ceiling"]
NAME = {"chain": "chain-8", "lchain": "chain-8 (lattice)", 15: "tree-16", 31: "tree-32", "ceiling": "top-16 ceiling"}
ROWS = {"chain": 8, "lchain": 8, 15: 16, 31: 32, "ceiling": None}
res, agree = {}, {}
t0 = time.time()
for d in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(os.path.expanduser(d), "*.npz"))):
        key = os.path.basename(p)[:-4]
        cls = "_".join(key.split("_")[:2])
        z = np.load(p)
        f, ids, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
        L = lattices(f, ids, off, gen)
        r = res.setdefault(cls, {q: [0, 0, 0] for q in POL})           # tokens, passes, old-counting tokens
        last = off + len(ids) - 1
        for q in POL:
            a = gen
            while a < last:
                lat = L[a - gen]
                true = ids[a - off + 1:a - off + BS]
                k = accepted(q, lat, true)
                r[q][0] += min(k + 1, last - a); r[q][1] += 1
                r[q][2] += min(k + 1, len(true)) if k == len(true) else k + 1
                a += k + 1
        g = agree.setdefault(cls, [0, 0])
        for a in range(gen, last):
            true = ids[a - off + 1:a - off + BS]
            if len(true) == 7:
                g[0] += int(accepted("chain", L[a - gen], true) == accepted("lchain", L[a - gen], true)); g[1] += 1
        print(f"[tree] {key}: {last - gen} anchors, {time.time() - t0:.0f}s", flush=True)
lines = []
for cls, r in sorted(res.items()):
    s = " | ".join(f"{NAME[q]} {r[q][0] / r[q][1]:.2f}" + (f" ({r[q][0] / r[q][1] / ROWS[q]:.2f}/row)" if ROWS[q] else "")
                   for q in POL)
    lines.append(f"{cls:10s} {s} | chain-8 old counting {r['chain'][2] / r['chain'][1]:.2f} | lattice greedy = "
                 f"official path on {agree[cls][0] / max(1, agree[cls][1]):.3f} of anchors")
    print(lines[-1], flush=True)
open(os.path.expanduser("~/Leap/data/tree_result.txt"), "w").write("\n".join(lines) + "\n")
print(f"[tree] done in {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
