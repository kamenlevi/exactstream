"""CPU logic test of the one-build pflm.py (Leap Thu 12 Nov): the real _split_layers with stand-in layers (tiny tensors on the
CPU; no model, no GPU). Five chunks of 3, 3, 3, 3 and 1 tokens with R57_PFLM_TOK=7 make the groups [0, 1] and [2, 3, 4], so the
short final chunk sits inside a group. Checked, for R57_PFLM_QD in (0, 1, 2, 3) x R57_PFLM_SEGALL in (0, 1):
  1. the outputs (hs) are bitwise the same in every arm with the same SEGALL setting (QD changes no math);
  2. each layer reads each of its E experts exactly once, all before the layer's MoE call; with QD, in slices of
     ceil(E / chunks) between the chunks, otherwise in one read;
  3. with QD = n, at most n chunks are in flight after each wait, and n + 1 at most at any moment;
  4. SEGALL: the router and the shared expert see per-chunk slices (3, 3 | 3, 3, 1) instead of whole groups (6 | 7).
usage: <venv python> test_onebuild.py   (prints PASS / FAIL lines; exit 1 on a failure)"""
import importlib.util, os, sys, types
import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(os.path.abspath(__file__))
E, NL, SIZES = 16, 2, [3, 3, 3, 3, 1]
LOG = []                                                   # ("A"|"E"|"R"|"M", layer, n)

q = types.ModuleType("mtplx.models.qwen4_exp")             # the two qwen4_exp helpers _split_layers calls
q.create_ssm_mask = lambda h, c: None
q._hyper_residual_write = lambda hyper, y, inject: hyper + y + inject
pk, pm = types.ModuleType("mtplx"), types.ModuleType("mtplx.models")
pk.models, pm.qwen4_exp = pm, q
sys.modules.update({"mtplx": pk, "mtplx.models": pm, "mtplx.models.qwen4_exp": q})


def load():
    spec = importlib.util.spec_from_file_location("pflm_onebuild", os.path.join(HERE, "pflm.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class MX:                                                  # pflm's mx, logging async_eval / eval
    def __init__(self, cur):
        self.cur = cur

    def __getattr__(self, k):
        return getattr(mx, k)

    def async_eval(self, *a):
        LOG.append(("A", self.cur[0], 0))
        return mx.async_eval(*a)

    def eval(self, *a):
        LOG.append(("E", self.cur[0], 0))
        return mx.eval(*a)


class Rec:                                                 # a per-token function that records the token counts it sees
    def __init__(self, name, w):
        self.name, self.w, self.seen = name, w, []

    def __call__(self, x):
        self.seen.append(x.shape[1])
        return x * self.w


class MLP:
    def __init__(self, L, pf, cur):
        self.L, self.pf, self.cur = L, pf, cur
        self.gate, self.shared_expert, self.shared_expert_gate = Rec("gate", 0.5), Rec("se", 0.25), Rec("seg", 2.0)
        self.switch_mlp = types.SimpleNamespace(_bm={"tier": {"E": E, "shapes4": [((2,), mx.float32)], "L": L}, "full": False})

    def __call__(self, M):
        LOG.append(("M", self.L, M.shape[1]))
        assert self.pf.LMC["have"] == set(range(E)), "MoE call before every expert was read"
        return M * 3.0 + self.gate(M) + self.shared_expert(M) * self.shared_expert_gate(M)


class Layer(dict):                                         # "ple" in layer -> False
    def __init__(self, L, pf, cur):
        super().__init__()
        self.L, self.cur, self.is_linear, self.mlp = L, cur, True, MLP(L, pf, cur)

    def attn_hyper_connection(self, h):
        self.cur[0] = self.L
        return h * 1.0, h * 0.5, h * 0.25

    def linear_attn(self, mixed, mask, c):
        return mixed + 1.0

    def mlp_hyper_connection(self, hyper, pending):
        bo, inject = pending
        return bo, hyper, inject


def arm(qd, segall):
    os.environ.update(R57_PFLM_TOK="7", R57_PFLM_QD=str(qd), R57_PFLM_SEGALL=str(segall), R57_PFLM_AHEAD="1")
    pf = load()
    cur = [0]
    pf.mx = MX(cur)
    reads = []
    pf.LMC["tier"] = types.SimpleNamespace(_read_rebuilt=lambda t, a, b, sc: (reads.append((t["L"], list(a))),
                                                                                 LOG.append(("R", t["L"], len(a)))))
    pf.LMC.update(have=set(), sc=None, t=None, reads=0)
    tm = types.SimpleNamespace(layers=[Layer(L, pf, cur) for L in range(NL)], ssm_idx=0, _ple_stage_idx=-1)
    xs = [mx.full((1, n, 4), float(k + 1)) for k, n in enumerate(SIZES)]
    hs = [x * 1.0 for x in xs]
    pf._scratch(tm.layers[0].mlp.switch_mlp._bm["tier"])   # the scratch's one-time allocation (its own mx.eval) before the log
    LOG.clear()
    pf._split_layers(tm, xs, hs, [None] * NL, lambda cs: [], False, None)
    mx.eval(*hs)
    return hs, reads, list(LOG), tm


ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + msg)


out = {}
for segall in (0, 1):
    for qd in (0, 1, 2, 3):
        hs, reads, log, tm = arm(qd, segall)
        out[(qd, segall)] = hs
        for L in range(NL):
            ids = [e for l_, a in reads if l_ == L for e in a]
            sl = [len(a) for l_, a in reads if l_ == L]
            pre = [ev for ev in log if ev[1] == L]
            pre = pre[:next(i for i, ev in enumerate(pre) if ev[0] == "M")]   # the layer's events before its first MoE call
            fl, mx_fl, after_wait = 0, 0, []
            for ev in pre:
                if ev[0] == "A":
                    fl += 1
                    mx_fl = max(mx_fl, fl)
                elif ev[0] == "E":
                    fl -= 1
                    after_wait.append(fl)
            want = [4, 4, 4, 4] if qd else [16]           # ceil(16 / 5) = 4 per slice
            check(sorted(ids) == list(range(E)) and sl == want,
                  f"QD={qd} SEGALL={segall} layer {L}: experts read once each, slices {sl} (want {want})")
            if qd:
                check(mx_fl <= qd + 1 and all(f <= qd for f in after_wait),
                      f"QD={qd} SEGALL={segall} layer {L}: in flight max {mx_fl}, after waits {after_wait}")
            else:
                check(not any(ev[0] == "E" for ev in pre), f"QD=0 SEGALL={segall} layer {L}: no waits (as before)")
        g = tm.layers[0].mlp.gate.seen
        want_g = [3, 3, 3, 3, 1] if segall else [6, 7]
        check(g == want_g, f"QD={qd} SEGALL={segall}: router token counts {g} (want {want_g})")
for segall in (0, 1):
    ref = out[(0, segall)]
    for qd in (1, 2, 3):
        same = all(mx.array_equal(a, b).item() for a, b in zip(ref, out[(qd, segall)]))
        check(same, f"SEGALL={segall}: outputs with QD={qd} bitwise == QD=0")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
