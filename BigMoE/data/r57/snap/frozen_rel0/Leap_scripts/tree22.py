"""L1-22 screen (Leap, Sat 2026-10-24): let the target pick. If a small picker cannot choose within DFlash2's lattice
(L1-21; published per-slot correctors gain ~0-1 % over DFlash2's selector), the 125B can, by verifying a token tree in one
pass. Rows cost expert bytes (union ~ rows^0.72), so the tree must pay for its rows.
Tree per block (best-first, budget N nodes + the anchor): a node is (depth j, token c, parent). Its children are the
selector's 16 candidates at depth j+1, scored by the selector's conditional log-softmax given c (unary + first-order edge,
exactly the shipped selector's scores, so the greedy chain is the tree's top path). Expand the N best paths by summed
log-prob. Accepted = the deepest tree path that equals the 125B's greedy tokens, + 1.
Replay: the official loop on the 24 held-out HumanEval samples (as cand21.py), drafter = L1-20 (or R22_WEIGHTS);
priced at both stacks (pass = (2.1 + experts x (rows / 16)^0.72) GB / 224 GB/s + drafting).
Usage: [BigMoE venv] [R22_WEIGHTS=data/l120_ft.safetensors] tree22.py -> ~/Leap/data/tree22{tag}.txt"""
import os, sys, glob, time, heapq
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_third_party"))
import dflash_model_mlx as D

class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                 + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
del w
draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
WEIGHTS = os.environ.get("R22_WEIGHTS", "data/l120_ft.safetensors")
if WEIGHTS:
    draft.load_weights(os.path.expanduser(os.path.join("~/Leap", WEIGHTS)) if not WEIGHTS.startswith(("/", "~")) else os.path.expanduser(WEIGHTS), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
SEL = draft.candidate_selector
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def pass_ms(rows, bs, experts):
    return (2.1 + experts * (rows / 16) ** 0.72) / 224 * 1000 + (3.5 if bs <= 16 else 4.0)

def block_tables(anchor, hid):
    """per depth: the selector's 16 candidates, their unary logits, hidden projection; plus a codebook lookup"""
    logits = draft.compute_logits(hid)                                        # [1, D, V]
    cand = mx.argpartition(logits, -SEL.top_k, axis=-1)[..., -SEL.top_k:][0]   # [D, k]
    unary = mx.take_along_axis(logits[0], cand, axis=-1)
    hp = SEL.hidden_projection(hid)[0]                                        # [D, r]
    succ = SEL.successor_codebook(cand)                                       # [D, k, r]
    mx.eval(cand, unary, hp, succ)
    return np.array(cand), np.array(unary.astype(mx.float32)), hp, succ

def child_scores(prev_tok, j, hp, succ, unary):
    """the shipped selector's score of depth j's candidates after prev_tok (bf16 as in select), log-softmax"""
    pc = SEL.predecessor_codebook(mx.array([prev_tok]))                       # [1, r]
    sc = unary[j] + np.array(mx.sum(pc[:, None] * hp[j][None, None] * succ[j][None], axis=-1).astype(mx.float32))[0]
    sc = sc - sc.max()
    return sc - np.log(np.exp(sc).sum())

def build_tree(anchor, cand, unary, hp, succ, N):
    """best-first: the N highest-scoring nodes (summed conditional log-prob); returns children map {node: {tok: node}}"""
    Dn = cand.shape[0]
    nodes = [(-1, anchor, -1)]                                                # (depth, token, parent)
    kids = {0: {}}
    heap = []
    ls = child_scores(anchor, 0, hp, succ, unary)
    for i in range(cand.shape[1]):
        heapq.heappush(heap, (-ls[i], 0, int(cand[0, i]), 0))
    while heap and len(nodes) - 1 < N:
        negs, j, tok, par = heapq.heappop(heap)
        if tok in kids[par]:
            continue
        nid = len(nodes); nodes.append((j, tok, par)); kids[par][tok] = nid; kids[nid] = {}
        if j + 1 < Dn:
            ls = child_scores(tok, j + 1, hp, succ, unary)
            for i in range(cand.shape[1]):
                heapq.heappush(heap, (negs - ls[i], j + 1, int(cand[j + 1, i]), nid))
    return kids

samples = []
for path in sorted(glob.glob(os.path.expanduser("~/Leap/data/dfl2d_code/*.npz"))):
    z = np.load(path)
    samples.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))

def replay(bs, N):
    tok = blocks = 0; ms = ms8 = 0.0
    for f, ids, off, gen in samples:
        n = len(ids); cache = draft.make_cache()
        start = max(off, gen - (draft.config.sliding_window - 1))
        for c in cache:
            c.offset = start
        fed, a = start, gen
        while a < off + n - 1:
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            anchor = int(ids[a - off])
            hid = draft.hidden_states(mx.array([[anchor] + [MASK] * (bs - 1)]), ctx, cache, 1)
            fed = a
            true = np.asarray(ids[a - off + 1:a - off + bs])
            kids = build_tree(anchor, *block_tables(anchor, hid), N)
            node, k = 0, 0
            while k < len(true) and int(true[k]) in kids[node]:
                node = kids[node][int(true[k])]; k += 1
            acc = min(k + 1, off + n - 1 - a)
            tok += acc; blocks += 1
            ms += pass_ms(N + 1, bs, 5.06); ms8 += pass_ms(N + 1, bs, 5.52)
            if (trim := cache[0].offset - a) > 0:
                D._trim_recent_cache(cache, trim)
            a = a + acc
    return tok / blocks, tok / ms * 1000, tok / ms8 * 1000, blocks

say(f"L1-22 screen: best-first token trees over DFlash2's selector lattice (16 candidates per depth, the selector's own "
    f"conditional scores), drafter {WEIGHTS or 'incoai'}; 24 held-out HumanEval samples; rows = N + 1, priced")
for bs, Ns in ((16, (15, 23, 31, 47, 63)), (32, (31, 47, 63, 95))):
    for N in Ns:
        t0 = time.time()
        tpp, ts, ts8, nb = replay(bs, N)
        say(f"  block {bs:2d}, tree of {N:3d} nodes: {tpp:5.2f} tokens per pass at {N + 1} rows; at the stack {ts:4.0f} "
            f"tok/s (top-8: {ts8:4.0f}) ({nb} blocks, {time.time() - t0:.0f} s)")
say("reference (cand21_l120): chain block 16 = 9.54 at 16 rows (269 / 254 tok/s), best truncated 311 / 296")
open(os.path.expanduser(f"~/Leap/data/tree22{os.environ.get('R22_TAG', '')}.txt"), "w").write("\n".join(lines) + "\n")
