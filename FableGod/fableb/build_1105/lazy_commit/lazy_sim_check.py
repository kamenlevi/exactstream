"""lazy_sim_check.py (Fable B, build_1105): fuzz the lazy commit protocol against today's, on CPU (numpy, seconds).
Two copies of one GDN layer start equal; the same random event stream runs through today's protocol and the lazy one:
  check     snapshot; forward S rows (1..16); then full accept / partial accept / rollback + re-forward + accept / abandon
  ar        a one-row forward + commit keep 1 (REF2's AR step)
  prefill   a forward of 20-40 rows with no snapshot and no commit (the stock path: lazy flushes first)
In the lazy copy a forward takes the body w.p. 0.9 and the stock path w.p. 0.1 (and w.p. 1.0 in a second pass). Snapshots
are copies and rollbacks are clones, as MTPLX's (new array objects), so nothing can lean on identity. After every
event: outputs equal, conv state equal, true recurrent state equal, all bit for bit. Also counted: state reads /
writes / replay kernels per protocol. (The PLE-carrying layer is never lazy: its entry has 4 leaves.)
Run: python -I lazy_sim_check.py [events per stream] [streams]   (default 400 x 300; exit 0 = PASS)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script's folder out
import lazy_protocol as LP

F = LP.F
N_EV = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_ST = int(sys.argv[2]) if len(sys.argv) > 2 else 300
eq = lambda x, y: x.shape == y.shape and x.tobytes() == y.tobytes()


def stream(seed, p_body=0.9):
    rng = np.random.default_rng(seed)
    conv0 = rng.standard_normal((3, LP.C)).astype(F)
    st0 = rng.standard_normal((LP.H, LP.DV, LP.DK)).astype(F)
    A, B = LP.Entry(conv0.copy(), st0.copy()), LP.Entry(conv0.copy(), st0.copy())
    ca = dict(reads=0, writes=0, steps=0, replays=0, flushes=0)
    cb = dict(reads=0, writes=0, steps=0, replays=0, flushes=0)
    kinds = dict(full=0, partial=0, rollback=0, abandon=0, ar=0, prefill=0, stockpath=0)
    rows = lambda n: (rng.standard_normal((n, LP.C)).astype(F), rng.standard_normal((n, LP.H)).astype(F),
                      rng.standard_normal((n, LP.H)).astype(F))

    def same(tag, ya=None, yb=None):
        ok = (ya is None or eq(ya, yb)) and eq(A[0], B[0]) and eq(A[1], LP.peek_state(B))
        if not ok:
            raise AssertionError("seed %d p_body %s: %s differs" % (seed, p_body, tag))

    for _ in range(N_EV):
        r = rng.random()
        if r < 0.08:                                           # prefill chunk
            x, a, b = rows(int(rng.integers(20, 41)))
            ya = LP.stock_forward(A, x, a, b, ca)
            yb = LP.lazy_forward(B, x, a, b, cb, body=False)
            kinds["prefill"] += 1
            same("prefill", ya, yb)
            continue
        if r < 0.20:                                           # AR step
            x, a, b = rows(1)
            sa, sb = LP.snapshot(A), LP.snapshot(B)
            body = rng.random() < p_body
            ya = LP.stock_forward(A, x, a, b, ca)
            yb = LP.lazy_forward(B, x, a, b, cb, body=body)
            LP.stock_commit(A, sa, 1, 1, counters=ca)
            LP.lazy_commit(B, sb, 1, 1, counters=cb)
            kinds["ar"] += 1
            same("ar", ya, yb)
            continue
        S = int(rng.integers(1, 17))
        x, a, b = rows(S)
        sa, sb = LP.snapshot(A), LP.snapshot(B)
        body = rng.random() < p_body
        kinds["stockpath"] += 0 if body else 1
        ya = LP.stock_forward(A, x, a, b, ca)
        yb = LP.lazy_forward(B, x, a, b, cb, body=body)
        same_y = eq(ya, yb)
        r2 = rng.random()
        if r2 < 0.10:                                          # abandoned: rolled back, nothing committed
            LP.rollback(A, sa); LP.rollback(B, sb)
            kinds["abandon"] += 1
        else:
            if r2 < 0.25:                                      # r57's "rep": rolled back, the same rows again
                LP.rollback(A, sa); LP.rollback(B, sb)
                sa, sb = LP.snapshot(A), LP.snapshot(B)
                body = rng.random() < p_body
                ya2 = LP.stock_forward(A, x, a, b, ca)
                yb2 = LP.lazy_forward(B, x, a, b, cb, body=body)
                same_y = same_y and eq(ya2, yb2) and eq(ya, ya2)
                kinds["rollback"] += 1
            keep = S if rng.random() < 0.25 else int(rng.integers(1, S + 1))
            kinds["full" if keep == S else "partial"] += 1
            LP.stock_commit(A, sa, keep, S, counters=ca)
            LP.lazy_commit(B, sb, keep, S, counters=cb)
        if not same_y:
            raise AssertionError("seed %d: check outputs differ" % seed)
        same("check")
    return ca, cb, kinds


add = lambda x, y: y if x is None else {k: x[k] + y[k] for k in y}
for p_body in (0.9, 1.0):
    tot = [None, None, None]
    for seed in range(N_ST):
        ca, cb, kinds = stream(seed + int(p_body * 1000) * 10 ** 6, p_body)
        tot = [add(tot[0], ca), add(tot[1], cb), add(tot[2], kinds)]
    ca, cb, kinds = tot
    print("body w.p. %.1f: %d streams x %d events, every output, conv state and true recurrent state equal bit for bit" % (
        p_body, N_ST, N_EV))
    print("   events: %s" % kinds)
    print("   today: state reads %d, writes %d, replay kernels %d, recurrence steps %d" % (ca["reads"], ca["writes"], ca["replays"], ca["steps"]))
    print("   lazy : state reads %d, writes %d, flush replays %d, recurrence steps %d" % (cb["reads"], cb["writes"], cb["flushes"], cb["steps"]))

# the clean case: every forward takes the body (the 45 GB check loop), code-like accepts: 24 % full, else uniform keep
rng = np.random.default_rng(7)
A = LP.Entry(rng.standard_normal((3, LP.C)).astype(F), rng.standard_normal((LP.H, LP.DV, LP.DK)).astype(F))
B = LP.Entry(A[0].copy(), A[1].copy())
ca = dict(reads=0, writes=0, steps=0, replays=0, flushes=0); cb = dict(ca)
n = 5000
for _ in range(n):
    S = 16
    x, a, b = (rng.standard_normal((S, LP.C)).astype(F), rng.standard_normal((S, LP.H)).astype(F), rng.standard_normal((S, LP.H)).astype(F))
    sa, sb = LP.snapshot(A), LP.snapshot(B)
    ya, yb = LP.stock_forward(A, x, a, b, ca), LP.lazy_forward(B, x, a, b, cb)
    keep = S if rng.random() < 0.24 else int(rng.integers(1, S))
    LP.stock_commit(A, sa, keep, S, counters=ca); LP.lazy_commit(B, sb, keep, S, counters=cb)
    assert eq(ya, yb) and eq(A[0], B[0]) and eq(A[1], LP.peek_state(B))
print("check loop only (%d 16-row checks, 24 %% full accepts): per check, today %.2f state reads + %.2f writes + %.2f replay "
      "kernels, %.1f recurrence steps; lazy %.2f reads + %.2f writes + %.2f replay kernels, %.1f steps" % (
          n, ca["reads"] / n, ca["writes"] / n, ca["replays"] / n, ca["steps"] / n,
          cb["reads"] / n, cb["writes"] / n, cb["flushes"] / n, cb["steps"] / n))
print("PASS")
