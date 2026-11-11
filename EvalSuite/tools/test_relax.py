"""CPU checks of RELAX's pure rules and of the gate's two id modes (docs/PREREG_relax_2026-11-02.md).
  .venv/bin/python pending/test_relax.py   -> one line per check, exit 0 when all pass"""
import math, os, sys
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(os.path.dirname(_here), "evalsuite"), _here):   # relax_core: evalsuite/ once wired
    if os.path.exists(os.path.join(_d, "relax_core.py")):
        sys.path.insert(0, _d)
        break
from relax_core import R, committed, gate_rows, keep_mask, kept, kld_top, prompt_grams, union_bytes, union_counts

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}")

# the gate rows
cf = [0.9995, 0.99, 0.5] + [0.0] * 12
g = gate_rows(cf, list(range(100, 108)), list(range(200, 215)), 0.999, None)
check("cf gate 0.999: only row 0", g == [1] + [0] * 15)
g = gate_rows(cf, list(range(100, 108)), list(range(200, 215)), 0.99, None)
check("cf gate 0.99: rows 0-1 (cf >= thr)", g == [1, 1] + [0] * 14)
check("row 15 never gated", gate_rows([1.0] * 15, [0] * 8, [0] * 15, 0.5, {(0,) * 8})[15] == 0)
prompt = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
grams = prompt_grams(prompt)
tail = [1, 2, 3, 5, 6, 7, 8, 9]                       # row 0's last 8 inputs: 1 2 3 5 6 7 8 9 (not in the prompt)
drafts = [10, 11, 12, 99] + [0] * 11                  # row 1: 2 3 5 6 7 8 9 10 (no); row 3: 5..12 (yes); row 4: 6..13? no (99)
g = gate_rows([0.0] * 15, tail, drafts, None, grams)
check("copy gate: row 3 only (5..12 occurs in the prompt)", g == [0, 0, 0, 2] + [0] * 12)
g = gate_rows([0.0, 0.0, 0.0, 0.9999] + [0.0] * 11, tail, drafts, 0.999, grams)
check("cf gate wins over copy on the same row", g[3] == 1 and g.count(2) == 0)
check("prompt grams: 3 eight-grams of a 10-token prompt", len(grams) == 3)
# keep mask
m = keep_mask([1, 0, 2] + [0] * 13)
check("keep mask: gated rows keep ranks 0-1", m is not None and m[0].tolist() == [True] * 2 + [False] * 8
      and m[1].all() and m[2].sum() == 2)
check("keep mask: None without gated rows", keep_mask([0] * 16) is None)
# acceptance
dr = [1, 2, 3, 4] + [9] * 11
am = [1, 2, 7, 4] + [8] * 12
dlm = [0.0, 0.0, -0.4, 0.0, -0.6] + [-1.0] * 10
check("exact k = 2, R1(0.5) k = 4", kept(dr, am, dlm, 0.5) == (2, 4))
check("no rule: k = exact k", kept(dr, am, dlm, None) == (2, 2))
check("R1 keeps every exact draft", all(kept(dr, am, [0.0] * 15, d)[1] == 15 for d in (0.0, 0.25, 0.5)))
check("committed: drafts then row k's argmax", committed(dr, am, 4, {999}, 100) == [1, 2, 3, 4, 8])
check("committed: cut after an EOS", committed([1, 999, 3] + [0] * 12, [1, 999, 3] + [0] * 13, 3, {999}, 100) == [1, 999])
check("committed: cut to the room left", committed(dr, am, 4, {999}, 2) == [1, 2])
# union bytes
ids = np.array([[[1, 2, 3], [1, 4, 5]], [[7, 7, 7], [7, 8, 9]]])        # 2 layers, 2 rows, K 3
ub, us = union_counts(ids, 2)
check("union counts: base 5 / 3, sidecar (ranks < 2) 3 / 2", ub.tolist() == [5, 3] and us.tolist() == [3, 2])
check("union bytes", union_bytes(ids, 2, [10, 10], [1, 1]) == (5 * 10 + 3 * 10 + 3 + 2, 8, 5))
# KLD with G known on its top ids
rng = np.random.default_rng(1)
for trial in range(3):
    V = 500
    le = rng.normal(size=V) * 3; lg_ = le + rng.normal(size=V) * 0.3
    lpe = le - np.log(np.exp(le).sum()); lpg = lg_ - np.log(np.exp(lg_).sum())
    exact = float((np.exp(lpe) * (lpe - lpg)).sum())
    top = np.argsort(-lpg)[:64]
    approx = kld_top(lpe[top], lpg[top])
    check(f"KLD top-64 vs exact (trial {trial}): {approx:.4f} vs {exact:.4f}", abs(approx - exact) <= 0.25 * exact + 1e-3)
check("KLD of equal rows = 0", abs(kld_top(np.log([0.5, 0.3]), np.log([0.5, 0.3]))) < 1e-12)

# the relaxed rules and the per-row rules
from relax_core import R1, R2, rule_ok, kept_rows, CodeScan, comment_draft_ok, comment_rows
import math as _m
check("R1: within 0.5 nat", rule_ok(R1, -0.49, 5) and not rule_ok(R1, -0.51, 0))
check("R2: top-2 and >= 0.25 p1", rule_ok(R2, _m.log(0.26), 1) and not rule_ok(R2, _m.log(0.24), 1)
      and not rule_ok(R2, -0.1, 2))
dr = [1, 2, 3, 4] + [9] * 11
am = [1, 2, 7, 4] + [8] * 12
dlm = [0.0, 0.0, -0.4, 0.0, -0.6] + [-1.0] * 10
drk = [0, 0, 3, 0, 1] + [5] * 10
check("kept_rows: exact rows only", kept_rows(dr, am, dlm, drk, [()] * 16) == (2, 2))
check("kept_rows: R1 on every row", kept_rows(dr, am, dlm, drk, [(R1,)] * 16) == (2, 4))
check("kept_rows: R2 fails row 2 (rank 3)", kept_rows(dr, am, dlm, drk, [(R2,)] * 16) == (2, 2))
check("kept_rows: R1 or R2", kept_rows(dr, am, dlm, drk, [(R1, R2)] * 16) == (2, 5))
check("kept_rows: a rule on row 2 only", kept_rows(dr, am, dlm, drk, [(), (), (R1,)] + [()] * 13) == (2, 4))

def regions(text, toks):
    """the comment rule on each token of a tokenized answer, the argmax taken equal to the token"""
    s, pre = CodeScan(), []
    for t in toks:
        pre.append((s.in_comment(), s.ctext))
        s.feed(t)
    cm = set(comment_rows(pre, toks, toks))
    return [i in cm for i in range(len(toks))]
ans = ["```python", "\n", "import os", "\n", "\n", "def f(x):", "\n", "    #", " add", " one", "\n", "    y = x", " #", " type",
       ":", " int", "\n", "    s = '#", " not", "'", " #", " real", "\n", '    """', "doc", " # no", '"""', "\n", "    return y",
       "\n", "```", "\n", "# prose", " after"]
rl = regions("".join(ans), ans)
want = {8, 9, 21}                                     # " add", " one" (line 3 comment), " real" (after a closed string)
got = {i for i, x in enumerate(rl) if x}
check(f"scanner: relaxed tokens {sorted(got)} == {sorted(want)}", got == want)
ans2 = ["```python", "\n", "#", " first", "\n", "# second", " line", "\n", "x = 1", " # third", " line", "\n"]
got2 = {i for i, x in enumerate(regions("".join(ans2), ans2)) if x}
check(f"scanner: the fence's first two lines stay exact, line 3 relaxes ({sorted(got2)})", got2 == {10})
s_ = CodeScan(); s_.feed("```python\nimport a\n\n    # ")
check("comment rule: a draft forming a tool prefix, or an argmax that would, stays exact",
      comment_rows([(s_.in_comment(), s_.ctext)] * 3, ["type", "noqa", "add"], ["x", "y", "type"]) == [])
check("comment_draft_ok: newline first / code after a newline are not relaxed",
      not comment_draft_ok("\n    x") and not comment_draft_ok(" a\nb") and comment_draft_ok(" a\n") and comment_draft_ok(" a"))

# the gate's id modes on a toy fused combine (MLX on the CPU): route == mask bitwise
try:
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    E, H, Kt = 32, 8, 10
    W = mx.random.normal((E, H, H), key=mx.random.key(0)).astype(mx.bfloat16)
    x = mx.random.normal((1, R, H), key=mx.random.key(1)).astype(mx.bfloat16)
    inds = mx.argsort(mx.random.uniform(shape=(1, R, E), key=mx.random.key(2)), axis=-1)[..., :Kt].astype(mx.uint32)
    sc = mx.softmax(mx.random.normal((1, R, Kt), key=mx.random.key(3)), axis=-1).astype(mx.bfloat16)
    keep = mx.array(keep_mask([1, 0, 2, 1] + [0] * 12))

    def combine(ids, s):          # each (row, rank): W[id] @ x_row, weighted by its score, summed over ranks
        y = mx.einsum("brkij,brj->brki", W[ids], x)
        return (y * s[..., None]).sum(axis=-2)

    s_m = mx.where(keep, sc, mx.zeros_like(sc))
    y_mask = combine(inds, s_m)
    y_route = combine(mx.where(keep, inds, inds[..., :1]), s_m)
    y_full = combine(inds, sc)
    mx.eval(y_mask, y_route, y_full)
    same = np.array_equal(np.array(y_mask.astype(mx.float32)), np.array(y_route.astype(mx.float32)))
    check("toy combine: route == mask bitwise", same)
    rows_same = [np.array_equal(np.array(y_mask[0, j].astype(mx.float32)), np.array(y_full[0, j].astype(mx.float32)))
                 for j in range(R)]
    check("toy combine: ungated rows unchanged, gated rows changed", all(rows_same[j] for j in range(4, R))
          and not any(rows_same[j] for j in (0, 2, 3)))
    u_mask = len(np.unique(np.array(inds)))
    u_route = len(np.unique(np.array(mx.where(keep, inds, inds[..., :1]))))
    check(f"route posts fewer distinct ids ({u_route} vs {u_mask})", u_route <= u_mask)
    # Q1-UNION: the graph's mask == the numpy reference (Breakthrough's dm_q1u_masks rule) on random checks
    from relax_core import gate_keep_mx, q1u_keep
    rng_ = np.random.default_rng(7)
    same_u = same_q = superset = no_new = True
    for trial in range(200):
        ids_ = np.stack([rng_.choice(64, 10, replace=False) for _ in range(R)]).astype(np.uint32)   # rank-sorted rows
        sure_ = rng_.random(R) < 0.6
        sure_[15] = False
        ku = np.array(gate_keep_mx(mx.array(ids_)[None], mx.array(sure_), True))
        kq = np.array(gate_keep_mx(mx.array(ids_)[None], mx.array(sure_), False))
        same_u &= np.array_equal(ku, q1u_keep(ids_, sure_))
        same_q &= np.array_equal(kq, keep_mask([int(x) for x in sure_]) if sure_.any() else np.ones((R, 10), bool))
        superset &= bool((ku | ~kq).all())                    # every slot Q1 keeps, Q1-UNION keeps
        posted = set(np.where(ku, ids_, ids_[:, :1]).ravel().tolist())
        union_q1 = set(np.where(kq, ids_, ids_[:, :1]).ravel().tolist())
        no_new &= posted == union_q1                          # the loader posts Q1's union: no new expert, same bytes
    check("Q1-UNION mask in the graph == the numpy reference (200 random checks)", same_u)
    check("Q1 mask in the graph == keep_mask (200 random checks)", same_q)
    check("Q1-UNION keeps every slot Q1 keeps", superset)
    check("Q1-UNION posts exactly Q1's expert union (no new expert, same bytes)", no_new)
except ImportError:
    print("skip MLX checks (no mlx)")
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
