"""MR_OBS (run.py tfs): per-call record of the gate traffic and the MoE block's input / output, to find the first call that
differs between two runs and whether its inputs were identical there: posted ids, post_ids' dep row ([fast-path ok,
table slot per id]), the released slots (spin), x, y. Record only after prefill (set .on).
compare: obs.py A.pkl B.pkl"""
import pickle, sys
import numpy as np


class Obs:
    def __init__(self, gate, Q, mx):
        self.mx, self.on, self.cur, self.rows = mx, False, [], []
        post0, spin0, blk0 = gate.post, gate.spin, Q.SparseMoeBlock.__call__

        def post(L, ids, *a, **k):
            dep, n = post0(L, ids, *a, **k)
            if self.on:
                self.cur.append(("post", L, ids, dep))
            return dep, n

        def spin(dep, L, n, *a, **k):
            out = spin0(dep, L, n, *a, **k)
            if self.on:
                self.cur.append(("spin" + str(k.get("phase", a[1] if len(a) > 1 else 0)), L, out))
            return out

        def block(m, x):
            y = blk0(m, x)
            if self.on:
                bm = getattr(m.switch_mlp, "_bm", None)
                self.cur.append(("moe", bm["L"] if bm else -1, x, y))
            return y
        gate.post, gate.spin, Q.SparseMoeBlock.__call__ = post, spin, block

    def flush(self, chunk):
        mx = self.mx
        conv = [[a.astype(mx.float32) if a.dtype in (mx.bfloat16, mx.float16) else a for a in r[2:]] for r in self.cur]
        mx.eval([a for c in conv for a in c])
        for r, c in zip(self.cur, conv):
            self.rows.append((chunk, r[0], r[1], [np.array(a) for a in c]))
        self.cur = []

    def save(self, path):
        pickle.dump(self.rows, open(path, "wb"))


def compare(pa, pb):
    A, B = pickle.load(open(pa, "rb")), pickle.load(open(pb, "rb"))
    print(f"{len(A)} vs {len(B)} records")
    first = None
    for i, (a, b) in enumerate(zip(A, B)):
        same_meta = a[:3] == b[:3]
        same = same_meta and len(a[3]) == len(b[3]) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a[3], b[3]))
        if not same:
            first = i
            break
    if first is None:
        print("all records identical"); return
    ch, kind, L = A[first][:3]
    print(f"first difference: record {first}: chunk {ch}, {kind}, layer {L}; B has {B[first][:3]}")
    for j in range(max(0, first - 6), min(len(A), first + 4)):      # the neighbourhood: same call's post / spin / moe
        a, b = A[j], B[j]
        eq = [bool(x.shape == y.shape and np.array_equal(x, y)) for x, y in zip(a[3], b[3])] if a[:3] == b[:3] else "meta"
        extra = ""
        if a[1] == "post" and a[:3] == b[:3]:
            extra = f" fast-path ok A {int(a[3][1].reshape(-1)[0])} B {int(b[3][1].reshape(-1)[0])}"
        if a[1] == "moe" and a[:3] == b[:3] and not eq[1]:
            extra = f" max|dy| {np.abs(a[3][1] - b[3][1]).max():.4g}, rows differing {int((np.abs(a[3][1] - b[3][1]).reshape(-1, a[3][1].shape[-1]).max(-1) > 0).sum())}"
        print(f"  rec {j}: chunk {a[0]} {a[1]} L{a[2]} equal {eq}{extra}")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
