# CPU logic test of r57_screen's fixed pre-route capture (Wed 4 Nov 05:5x): exec _pr_top / _layer_pr / pr_eval from the file with
# mocks. usage: .venv/bin/python -I scripts/test_pr_capture.py scripts/r57_screen.py
import ast, sys, types
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
src = open(sys.argv[1]).read()
fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
       and n.name in ("_pr_top", "_layer_pr", "pr_eval")}
assert len(fns) == 3, fns.keys()
W, E, ROWS, NL_ = 8, 16, 3, 4
mx.random.seed(0)
GATES = [mx.random.normal((W, E)) for _ in range(NL_)]
class Read:                                      # GatedResidual stand-in: side-specific mix, pending written first
    def __init__(s, off): s.off = off; s.calls = []
    def __call__(s, h, pending=None):
        s.calls.append(pending is None)
        if pending is not None: h = h + pending[0]
        return h * s.off + 0.1 * s.off, h, mx.zeros(h.shape[:-1] + (4,))
class MLP:
    def __init__(s, g): s.gate = lambda z: z @ g
    def __call__(s, z): return z * 0.5
class Layer(dict):                               # "ple" in self works on a dict subclass
    def __init__(s, i):
        super().__init__(); s.i = i; s.is_linear = True
        s.attn_hyper_connection = Read(1.0 + i); s.mlp_hyper_connection = Read(-2.0 - i)
        s.mlp = MLP(GATES[i]); s.linear_attn = lambda m, a, c: m * 0.25
        if i == 1: s["ple"] = 1; s.ple = lambda h, ids, c: h * 0 + 7.0
lay = [Layer(i) for i in range(NL_)]
ns = dict(mx=mx, np=np, NL=NL_, lay=lay, PR_IDX={id(l): i for i, l in enumerate(lay)},
          SW_=[types.SimpleNamespace(_bm={"full": False, "L": 100 + i}) for i in range(NL_)],
          Q=types.SimpleNamespace(_hyper_residual_write=lambda h, bo, inj: h + bo),
          PR=dict(on=True, cur=None, y={}, blk={}, cap={}, dump=[]), PRS={}, PRD=dict(n=0, L=[], chk=[], row=[], X=[], Y=[]))
for f in fns.values(): exec(f, ns)
x = mx.random.normal((1, ROWS, W))
ns["PR"]["blk"] = {100 + i: [x * 0.01 * (i + 1), x * 0.02 * (i + 1)] for i in range(NL_)}   # L's partial outputs
# reference forward, by hand: the stream each layer starts from, its MLP-side read and its true MoE input
h, starts, mlp_in, outs, hyp = x, [], [], [], []
for i, l in enumerate(lay):
    if i == 1: h = h + 7.0
    starts.append(h)
    m, hy, inj = h * (1.0 + i) + 0.1 * (1.0 + i), h, None
    bo = m * 0.25
    hw = hy + bo
    mi = hw * (-2.0 - i) + 0.1 * (-2.0 - i)
    mlp_in.append(mi); hyp.append(hw)
    h = hw + mi * 0.5; outs.append(h)
top = lambda i, z: np.sort(np.array(mx.argpartition(-(z @ GATES[i]).astype(mx.float32), kth=9, axis=-1)[..., :10]).reshape(-1, 10), axis=1)
h2 = x
for l in lay:
    h2 = ns["_layer_pr"](l, h2, input_ids=None, ssm_mask=None, cache=None)
assert np.allclose(np.array(h2), np.array(outs[-1])), "forward changed"
cap = ns["PR"]["cap"]
assert sorted(cap) == [0, 1, 2] and all(sorted(cap[i]) == ["full", "fullA", "p3", "p6", "r57"] for i in cap), {i: sorted(c) for i, c in cap.items()}
for i in range(3):
    n = i + 1
    exp = dict(r57=top(n, mlp_in[i]),
               full=top(n, starts[n] * (-2.0 - n) + 0.1 * (-2.0 - n)),          # L+1's MLP-side read of its pre-attention stream
               fullA=top(n, outs[i] * (1.0 + n) + 0.1 * (1.0 + n)),             # L+1's attention-side read of L's output (no PLE)
               p3=top(n, (hyp[i] + x * 0.01 * n) * (-2.0 - n) + 0.1 * (-2.0 - n)),
               p6=top(n, (hyp[i] + x * 0.02 * n) * (-2.0 - n) + 0.1 * (-2.0 - n)))
    for k, v in exp.items():
        got = np.sort(np.array(cap[i][k][0]).reshape(-1, 10), axis=1)
        assert (got == v).all(), (i, k)
assert not np.allclose(np.array(starts[1]), np.array(outs[0])), "PLE must separate full from L0's output at pair 0 -> 1"
# truth = L+1's top-10 on its true MoE input -> "full" scored, dump holds (L+1, X = full's input, Y = the true MoE input)
capd = {100 + n: [mx.argpartition(-(mlp_in[n] @ GATES[n]), kth=9, axis=-1)[..., :10]] for n in range(1, NL_)}
dump = list(ns["PR"]["dump"])
assert [d[0] for d in dump] == [1, 2, 3], [d[0] for d in dump]
ns["pr_eval"](capd)
P = ns["PRS"]; D = ns["PRD"]
assert sorted(P) == ["full", "fullA", "p3", "p6", "r57"] and all(len(v) == 3 for v in P.values())
assert all(s[1] == ROWS for v in P.values() for s in v)
print("recall per variant:", {k: round(float(np.mean([s[2] for s in v])), 3) for k, v in P.items()})
X = np.concatenate(D["X"]); Y = np.concatenate(D["Y"]); L_ = np.concatenate(D["L"])
assert X.shape == (3 * ROWS, W) and Y.shape == X.shape and X.dtype == np.float32 and list(L_) == [1] * 3 + [2] * 3 + [3] * 3
for n in (1, 2, 3):
    assert np.allclose(X[L_ == n], np.array(starts[n] * (-2.0 - n) + 0.1 * (-2.0 - n)).reshape(-1, W))
    assert np.allclose(Y[L_ == n], np.array(mlp_in[n]).reshape(-1, W))
assert D["n"] == 1 and ns["PR"]["dump"] == [] and ns["PR"]["cap"] == {}
print("PASS: full = L+1's MLP-side pre-attention read (PLE in), fullA = the old attention-side read, p3/p6 MLP side, dump X/Y right")
