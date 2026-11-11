"""MR_CSUM (run.py gen): per-op checksums to find the first computation that differs between two runs (B12LL under GPU load).
Order-independent integer sums on the GPU: sum(bits as uint32) mod 2^32 of each recorded tensor, so the checksum itself
is deterministic. Recorded per call: decoder layer input / output, attention output (GDN or full), MoE input / output,
posted ids, released slots. Read once per pass (pass hook); nothing recorded inside commit_verified_window (its PLE
replay's MoE output is never evaluated by the engine, and must not be forced here).
compare: csum.py A.json B.json"""
import json, sys


class CSum:
    def __init__(self, mx, Q, gate, nl=48, ext=None):
        self.mx, self.cur, self.passes, self.skip, self.L, self.fwd = mx, [], [], False, -1, False
        self.ext, self.hp, self.hs, self.bad = ext, [], [], []        # ext: loader handoff check (debug ring vs posted ids,
        if ext is not None:                                          # published ring vs released slots), per trunk call
            import numpy as np
            self.np = np
            self.dbg = mx.zeros((4 << 20,), mx.int32); mx.eval(self.dbg); ext.set_debug(self.dbg)
            self.dbgv = np.frombuffer(memoryview(self.dbg), np.int32)
            self.ringv = np.frombuffer(memoryview(gate.ring), np.int32)
        u = lambda a: a.view(mx.uint16).astype(mx.uint32) if a.dtype in (mx.bfloat16, mx.float16) else a.view(mx.uint32)
        self.cs = lambda a: mx.sum(u(a.reshape(-1)))

        def rec(kind, a):                                            # trunk forwards only (layer 0 .. nl-1 in order):
            if self.fwd and not self.skip and self.L >= 0:              # no MTP layer, no replay, no lazy appends
                self.cur.append((kind, self.L, self.cs(a)))
        lc0, moe0, gdn0, att0 = Q.DecoderLayer.__call__, Q.SparseMoeBlock.__call__, Q.GatedDeltaNet.__call__, Q.Attention.__call__
        post0, spin0 = gate.post, gate.spin

        def lc(m, hidden, **kw):
            Lp, self.L = self.L, getattr(m, "_b12_L", None)
            self.L = -1 if self.L is None else self.L
            if self.L == 0 and not self.skip:
                self.fwd = True
            rec("layer_in", hidden)
            out = lc0(m, hidden, **kw)
            rec("layer_out", out)
            if self.L == nl - 1:
                self.fwd = False
            self.L = Lp
            return out

        def moe(m, x):
            rec("moe_x", x)
            y = moe0(m, x)
            rec("moe_y", y)
            return y

        def gdn(m, *a, **k):
            y = gdn0(m, *a, **k); rec("gdn", y); return y

        def att(m, *a, **k):
            y = att0(m, *a, **k); rec("attn", y); return y

        def post(L, ids, *a, **k):
            dep, n = post0(L, ids, *a, **k); rec("post", ids)
            if self.fwd and not self.skip and self.L >= 0 and self.ext is not None:
                self.hp.append((L, gate.seq, ids.reshape(-1)))
            return dep, n

        def spin(dep, L, n, *a, **k):
            out = spin0(dep, L, n, *a, **k); rec("spin", out)
            ph = k.get("phase", a[1] if len(a) > 1 else 0)
            if self.fwd and not self.skip and self.L >= 0 and self.ext is not None and not ph:
                self.hs.append((L, gate.seq, out.reshape(-1)))
            return out
        Q.DecoderLayer.__call__, Q.SparseMoeBlock.__call__ = lc, moe
        Q.GatedDeltaNet.__call__, Q.Attention.__call__ = gdn, att
        gate.post, gate.spin = post, spin
        for cls in (Q.Qwen4ExpTextModel, Q.Model):
            cv0 = cls.commit_verified_window

            def cv(m, *a, _cv0=cv0, **k):
                self.skip = True
                try:
                    return _cv0(m, *a, **k)
                finally:
                    self.skip = False
            cls.commit_verified_window = cv

    def flush(self):
        vals = [c for _, _, c in self.cur]
        self.mx.eval(vals + [a for *_, a in self.hp] + [a for *_, a in self.hs])
        self.passes.append([[k, L, int(v.item())] for (k, L, _), v in zip(self.cur, vals)])
        np, p = getattr(self, "np", None), len(self.passes) - 1
        for L, seq, a in self.hp:                                    # what the loader read vs what the GPU posted
            g = np.array(a) & 0xFFFFF; o = self.ext.ring_off(L, seq); d = self.dbgv[o:o + g.size]
            if not np.array_equal(g, d):
                self.bad.append(["ids", p, L, seq, np.nonzero(g != d)[0][:8].tolist(), g[g != d][:4].tolist(), d[g != d][:4].tolist()])
        for L, seq, a in self.hs:                                    # what the loader published vs what the spin released
            g = np.array(a); o = self.ext.ring_off(L, seq); r = self.ringv[o:o + g.size]
            if not np.array_equal(g, r):
                self.bad.append(["slots", p, L, seq, np.nonzero(g != r)[0][:8].tolist(), g[g != r][:4].tolist(), r[g != r][:4].tolist()])
        self.cur, self.hp, self.hs = [], [], []

    def reset(self):
        self.cur, self.passes, self.hp, self.hs, self.bad = [], [], [], [], []


def compare(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    for p, (ra, rb) in enumerate(zip(A, B)):
        if ra == rb:
            continue
        i = next((j for j, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
        print(f"first difference: pass {p}, record {i} of {len(ra)} / {len(rb)}: A {ra[i] if i < len(ra) else '-'} "
              f"B {rb[i] if i < len(rb) else '-'}")
        for j in range(max(0, i - 8), min(len(ra), len(rb), i + 4)):
            print(f"  {j}: A {ra[j]}  B {rb[j]}{'' if ra[j] == rb[j] else '   <-- differs'}")
        return
    print(f"all {min(len(A), len(B))} common passes identical (passes {len(A)} vs {len(B)})")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
