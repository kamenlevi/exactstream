"""CPU test for sticky/dynk.py (R-26 DYNK-Q):
1. block_dynk = the suite's block (the STATE.sorted path of block() in ~/EvalSuite/evalsuite/model.py, read from its
   source) line for line, apart from the two R-26 lines;
2. on a small fake MoE layer (bf16, 512 experts, top-10, gate-like logits for 400 rows; per-expert outputs from a table),
   block_dynk at tau 0 = the suite's own block code (exec'd from its source) bit for bit; at tau 0.06, rows that keep
   all 10 are bit for bit the suite's, rows that drop = the suite's sorted ids with the dropped experts' weight 0 and
   the kept ones renormalized (independent float64 sum over the expert table; bf16 tolerance); both kinds of row occur;
3. dynk_scores: the drop decision = v3's sorted scores at ranks 8 / 9 < tau (numpy), stats = the numpy count, tau 1 =
   ranks 8-9 zeroed everywhere (= top-8 renormalized);
4. R-31 uniq_scores on correlated 16-row checks = a numpy copy of MoonShots' uniq24.py mass rule (r0 4, theta 0.06 /
   0.08 / 0.12, renormalized and not): drop masks, scores, stats; block_dynk with the UNIQ rule at theta 0 = the suite's.
usage: [BigMoE venv] python tests/test_dynk.py"""
import os, re, sys, textwrap
from types import SimpleNamespace
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sticky"))
import dynk  # noqa: E402

mx.set_default_device(mx.cpu)
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}")


# 1. the suite's sorted block, from its source
src = open(os.path.expanduser("~/EvalSuite/evalsuite/model.py")).read()
a = src.index("        gates = mx.softmax(self.gate(x), axis=-1, precise=True)\n        k = self.top_k")
b = src.index("\n", src.index("        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)", a))
suite_lines = [ln.strip() for ln in src[a:b].split("\n") if ln.strip()]
import inspect  # noqa: E402
mine = [ln.strip() for ln in inspect.getsource(dynk.block_dynk).split("\n")[1:]
        if ln.strip() and "# R-26" not in ln and "# R-31" not in ln]
check(f"block_dynk = the suite's sorted block line for line apart from the R-26 lines ({len(suite_lines)} lines)",
      mine == suite_lines and len(suite_lines) >= 10)
ns = {"mx": mx}
exec("def suite_sorted(self, x):\n" + textwrap.indent(textwrap.dedent(src[a:b]), "    "), ns)
suite_sorted = ns["suite_sorted"]

# 2. a fake layer
g = np.random.default_rng(3)
R, E, D, TAU = 400, 512, 16, 0.06
Wg = mx.array(g.normal(0, 1.6 / np.sqrt(32), (32, E)).astype(np.float32)).astype(mx.bfloat16)
Wo = mx.array(g.normal(0, 1, (E, D)).astype(np.float32)).astype(mx.bfloat16)
X = mx.array((g.normal(0, 1, (1, R, 32)) + g.normal(0, 0.5, (1, R, 1))).astype(np.float32)).astype(mx.bfloat16)
STATE = SimpleNamespace(capture=None)
ns["STATE"] = STATE
lay = SimpleNamespace(top_k=10, norm_topk_prob=True, gate=lambda x: x @ Wg,
                      switch_mlp=SimpleNamespace(_bm={"L": 0}), shared_expert_gate=lambda x: mx.zeros(x.shape[:-1] + (1,), x.dtype),
                      shared_expert=lambda x: mx.zeros(x.shape[:-1] + (D,), x.dtype))
lay.switch_mlp = type("SW", (), {"_bm": {"L": 0}, "__call__": lambda self, x, inds: Wo[inds]})()
y_suite = suite_sorted(lay, X)
acc = []
y0 = dynk.block_dynk(lay, X, 0.0, STATE, acc)
y6 = dynk.block_dynk(lay, X, TAU, STATE, acc)
mx.eval(y_suite, y0, y6)
check("tau 0: block_dynk = the suite's block bit for bit", np.array_equal(np.array(y0.astype(mx.float32)), np.array(y_suite.astype(mx.float32))))
gates = mx.softmax(X @ Wg, axis=-1, precise=True)                  # the suite's sorted ids and scores, step by step
inds = mx.argpartition(gates, kth=-10, axis=-1)[..., -10:]
sc = mx.take_along_axis(gates, inds, axis=-1)
sc = sc / sc.sum(axis=-1, keepdims=True)
o_ = mx.argsort(-sc, axis=-1)
ids, sc = np.array(mx.take_along_axis(inds, o_, axis=-1))[0], np.array(mx.take_along_axis(sc, o_, axis=-1).astype(mx.float32))[0]
drop = np.zeros((R, 10), bool)
drop[:, 8:] = sc[:, 8:] < TAU
dr = drop.any(1)
ys, y6n = np.array(y_suite.astype(mx.float32))[0], np.array(y6.astype(mx.float32))[0]
Won = np.array(Wo.astype(mx.float32)).astype(np.float64)
want = np.stack([(Won[ids[r][~drop[r]]] * (sc[r][~drop[r]] / sc[r][~drop[r]].sum())[:, None]).sum(0) for r in range(R)])
err = np.abs(y6n - want).max(1) / (np.abs(want).max(1) + 1e-9)
check(f"tau {TAU}: rows that keep all 10 ({(~dr).sum()}) = the suite's bit for bit; rows that drop ({dr.sum()}: rank 9 only "
      f"{(drop[:, 9] & ~drop[:, 8]).sum()}, ranks 8 + 9 {drop[:, 8].sum()}) = the kept experts renormalized (max relative "
      f"error {err[dr].max():.4f}, bf16)", np.array_equal(y6n[~dr], ys[~dr]) and err[dr].max() < 0.02 and (~dr).sum() > 20
      and dr.sum() > 20 and drop[:, 8].sum() > 5 and (drop[:, 9] & ~drop[:, 8]).sum() > 5
      and np.abs(y6n[dr] - ys[dr]).max() > 0)

# 3. dynk_scores
s_in = mx.array(sc[None]).astype(mx.bfloat16)
s6, st = dynk.dynk_scores(s_in, TAU)
s6n = np.array(s6.astype(mx.float32))[0]
s1, _ = dynk.dynk_scores(s_in, 1.0)
s1n = np.array(s1.astype(mx.float32))[0]
scb = np.array(s_in.astype(mx.float32))[0].astype(np.float64)
t8 = scb[:, :8] / scb[:, :8].sum(1, keepdims=True)
stn = np.array(st)
check(f"dynk_scores: zeros exactly where v3's rank 8 / 9 < tau, rows without a drop untouched, stats {stn.round(3).tolist()} "
      f"= numpy; tau 1 = top-8 renormalized", ((s6n == 0) == drop).all() and np.array_equal(s6n[~dr], scb[~dr].astype(np.float32))
      and stn[0] == R and stn[1] == drop[:, 9].sum() and stn[2] == drop[:, 8].sum() and abs(stn[3] - (scb * drop).sum()) < 1e-3
      and (s1n[:, 8:] == 0).all() and np.allclose(s1n[:, :8], t8, rtol=1e-2, atol=1e-4))
# 4. R-31 uniq_scores: 16-row checks of correlated rows (shared topic), against a numpy copy of uniq24.py's mass rule
g4 = np.random.default_rng(5)
bad, nd, nrows, tot_drop, tot_mass, mx_err = [], 0, 0, 0, 0.0, 0.0
for it_ in range(40):
    lg16 = mx.array((g4.normal(0, 1.2, (1, 16, E)) + 1.5 * g4.normal(0, 1, (1, 1, E))).astype(np.float32)).astype(mx.bfloat16)
    gt = mx.softmax(lg16, axis=-1, precise=True)
    ii = mx.argpartition(gt, kth=-10, axis=-1)[..., -10:]
    ss = mx.take_along_axis(gt, ii, axis=-1)
    ss = ss / ss.sum(axis=-1, keepdims=True)
    oo = mx.argsort(-ss, axis=-1)
    ii, ss = mx.take_along_axis(ii, oo, axis=-1), mx.take_along_axis(ss, oo, axis=-1)
    theta = (0.06, 0.08, 0.12)[it_ % 3]
    for renorm in (True, False):
        out, st = dynk.uniq_scores(ii, ss, theta, 4, renorm)
        on, inn, sn = np.array(out.astype(mx.float32))[0], np.array(ii)[0], np.array(ss.astype(mx.float32))[0].astype(np.float64)
        tot = {}                                                     # uniq24.py's rule, UNIQ_MODE=mass, r0 4
        for r in range(16):
            for k in range(10):
                tot[int(inn[r, k])] = tot.get(int(inn[r, k]), 0.0) + (sn[r, k] if k >= 4 else 9.0)
        dm = np.array([[k >= 4 and tot[int(inn[r, k])] < theta for k in range(10)] for r in range(16)])
        kept = np.where(dm, 0.0, sn)
        want = np.where(dm.any(1, keepdims=True), kept / kept.sum(1, keepdims=True), sn) if renorm else kept
        stn = np.array(st)
        mx_err = max(mx_err, float((np.abs(on - want) / (np.abs(want) + 1e-4)).max()))
        ok_ = ((on == 0) == (dm | (sn == 0))).all() and np.allclose(on, want, rtol=2e-2, atol=1e-4) \
            and np.array_equal(on[~dm.any(1)], np.array(ss.astype(mx.float32))[0][~dm.any(1)]) \
            and stn[0] == 16 and stn[1] == dm.any(1).sum() and stn[2] == dm.sum() and abs(stn[3] - (sn * dm).sum()) < 1e-3
        if not ok_:
            bad.append((it_, theta, renorm))
        if renorm:
            nd += dm.any(1).sum(); nrows += 16; tot_drop += dm.sum(); tot_mass += (sn * dm).sum()
check(f"uniq_scores = uniq24.py's mass rule (r0 4; theta 0.06 / 0.08 / 0.12; renormalized and not) on 40 correlated "
      f"16-row checks: drop masks, scores, stats ({len(bad)} off: {bad[:3]}); {nd} of {nrows} rows drop, "
      f"{tot_drop / nrows:.2f} uses and {100 * tot_mass / nrows:.2f} % mass per row; max relative error {mx_err:.4f} (bf16)",
      not bad and nd > 50 and nd < nrows)
X16 = X[:, :16, :]                                                  # R-31: the expert sum in 8-row chunks = in one call
yc = [dynk.block_dynk(lay, X16, r_, STATE, acc, chunk=c_) for r_ in (0.0, lambda i_, s_: dynk.uniq_scores(i_, s_, 0.08))
      for c_ in (0, 8)]
check("chunk 8: a 16-row check's output = the unchunked one bit for bit (theta 0 and UNIQ 0.08)",
      all(np.array_equal(np.array(yc[i].astype(mx.float32)), np.array(yc[i + 1].astype(mx.float32))) for i in (0, 2)))
y_u0 = dynk.block_dynk(lay, X, lambda i_, s_: dynk.uniq_scores(i_, s_, 0.0), STATE, acc)
check("block_dynk with the UNIQ rule at theta 0 = the suite's block bit for bit",
      np.array_equal(np.array(y_u0.astype(mx.float32)), np.array(y_suite.astype(mx.float32))))
print("ALL OK" if ok_all else "SOME FAILED")
