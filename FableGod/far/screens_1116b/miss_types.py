"""THINK_1116b screen (Fable FAR, Mon 16 Nov 2026): what the first misses of tree-32 paths ARE, on the real 127-node dumps.
For each anchor: the tree-32 path = the deepest on_true node among the first 32 pops (node < 32); its true next token
y_next (rank ynext_rank in the head's top-k) vs the head's top-1 t'. Classes:
  split : text(t') is a proper prefix of text(y_next)+text(y_next2), or text(y_next) is a proper prefix of text(t')
          (the same characters, tokenised differently)
  subst : the head's own best continuation under t' (its best-first chain in the 127 dump) equals the true tokens after
          y_next for L >= 4 (a one-token substitution that re-syncs)
  ins   : the head's chain under t' equals true[y_next:] (the head inserted a token)
  del   : t' == y_next2 and the chain continues true[y_next2+1:] (the head dropped a token)
CPU, < 1 GB, seconds."""
import sys, json, numpy as np
from collections import Counter
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/kamen/BigMoE/models/bare/tokenizer.json")
D = "/Users/kamen/BigMoE/data/deepdraft/ceiling/"
set_ = sys.argv[1] if len(sys.argv) > 1 else "heplus24"
z = np.load(D + f"tree_t30b_{set_}_n127k8.npz")
tk = np.load(D + f"topk_t30b_{set_}.npz")
# true sequences per file: topk rows (fi, row, y)
seq = {}
for fi in np.unique(tk["fi"]):
    m = tk["fi"] == fi; rows = tk["row"][m]; ys = tk["y"][m]
    o = np.argsort(rows); seq[int(fi)] = (rows[o], ys[o])
def true_at(fi, row):
    rows, ys = seq[fi]; i = np.searchsorted(rows, row)
    return int(ys[i]) if i < len(rows) and rows[i] == row else None
N = int(z["nodes"]); A = len(z["a_fi"])
fi_, an_, node, parent, depth, token, path_p, y, on_true = (z[k] for k in ["fi","an","node","parent","depth","token","path_p","y","on_true"])
top_ids, ynext_rank = z["top_ids"], z["ynext_rank"]
# the dump is file-grouped and node-major: sort to (fi, an, node) so every anchor is one contiguous block of N nodes
o = np.lexsort((node, an_, fi_))
fi_, an_, node, parent, depth, token, path_p, y, on_true, top_ids, ynext_rank = (x[o] for x in (fi_, an_, node, parent, depth, token, path_p, y, on_true, top_ids, ynext_rank))
assert (node.reshape(A, N) == np.arange(N)).all()
NODES = int(sys.argv[2]) if len(sys.argv) > 2 else 32
L = 4
cnt = Counter(); kept = []; p1s = {}; p1_other = []; capd = []; avail = 0; ranks = Counter(); examples = []
for a in range(A):
    s = a * N; e = s + N
    fi = int(fi_[s]); an = int(an_[s])
    ot = np.where(on_true[s:e] & (node[s:e] < NODES))[0]
    if len(ot) == 0: cnt["no_root"] += 1; continue
    end = ot[np.argmax(depth[s:e][ot])]; d = int(depth[s + end])
    kept.append(d + 1)
    # position of the end node's token in the sequence: row an + d (root = anchor row an at depth 0)
    pos = an + d                                   # token at position p is topk row p + 1
    y1 = true_at(fi, pos + 2); y2 = true_at(fi, pos + 3)
    if y1 is None: cnt["seq_end"] += 1; continue
    t1 = int(top_ids[s + end][0])
    if t1 == y1: cnt["cap_or_width"] += 1; capd.append(d); continue   # the true child exists but was not popped within NODES (cap / width)
    cnt["miss"] += 1; ranks[int(ynext_rank[s + end])] += 1
    st1, sy1 = tok.decode([t1]), tok.decode([y1]); sy12 = sy1 + (tok.decode([y2]) if y2 is not None else "")
    cls = None
    # the head's best chain under t' in the 127 dump
    ch = np.where((parent[s:e] == end) & (token[s:e] == t1))[0]
    chain = []
    if len(ch):
        cur = ch[0]
        for _ in range(L + 2):
            kids = np.where(parent[s:e] == cur)[0]
            if len(kids) == 0: break
            cur = kids[np.argmax(path_p[s:e][kids])]; chain.append(int(token[s + cur]))
    H = tok.decode([t1] + chain[:3]); T = tok.decode([t for t in [y1, y2, true_at(fi, pos + 4), true_at(fi, pos + 5)] if t is not None])
    cp = 0
    while cp < min(len(H), len(T)) and H[cp] == T[cp]: cp += 1
    if st1 != sy1 and cp > max(len(st1), len(sy1)): cls = "split"
    if len(chain) >= L:
        avail += 1
        tr_sub = [true_at(fi, pos + 3 + i) for i in range(L)]
        tr_ins = [true_at(fi, pos + 2 + i) for i in range(L)]
        if cls is None and chain[:L] == tr_sub: cls = "subst"
        elif cls is None and chain[:L] == tr_ins: cls = "ins"
    if cls is None and y2 is not None and t1 == y2 and len(chain) >= L - 1 and chain[:L-1] == [true_at(fi, pos + 4 + i) for i in range(L-1)]: cls = "del"
    cnt[cls or "other"] += 1
    p1 = float(z["top_p"][o][s + end][0]); p1s.setdefault(cls or "other", []).append(p1)
    if cls == "subst": cnt["subst_word_swap"] += int(st1.strip().isalpha() and sy1.strip().isalpha())
    if cls is None: p1_other.append(p1)
    if cls and len(examples) < 12: examples.append((cls, repr(T[:30]), repr(H[:30])))
print(f"{set_}: anchors {A}, tree-{NODES} kept mean {np.mean(kept):.2f} (bonus incl.)")
print(dict(cnt)); print("ynext rank at the miss:", sorted(ranks.items())[:9])
print(f"chains under t' with >= {L} nodes available: {avail} of {cnt['miss']}")
for ex in examples: print("  ", ex)
import numpy as _n
print("cap_or_width end depth: median", _n.median(capd), "share at depth >= 31:", _n.mean(_n.array(capd) >= 31).round(3), "share depth < 20:", _n.mean(_n.array(capd) < 20).round(3))
for k, v in p1s.items(): v = _n.array(v); print(f"head p1 at the miss [{k}] n {len(v)}: median {_n.median(v):.3f}; share p1 >= 0.9: {(v >= 0.9).mean():.3f}; share p1 < 0.6: {(v < 0.6).mean():.3f}")
