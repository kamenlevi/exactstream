# mock of e15_engine's time mode: a fake model whose row i logits depend only on (token i, position), so A8 == A16[:8] by design;
# checks the arm plumbing (8 rows for A8 / A8b), the bits pairs, the ratios of means and the verdict line. No GPU model.
import os, sys, types, contextlib
os.environ["E14_ARMS"] = "A16,A8,A16b,A8b,C32one"; os.environ["R57_G32TIME"] = "4"; os.environ["R57_G32TIME_WARM"] = "1"
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts/week98/e15"))
import mlx.core as mx
import e15_engine as E
V = 64
class Model:
    def __call__(self, x, cache=None):
        n = x.shape[1]; base = cache["pos"]
        t = x[0].astype(mx.float32)[:, None]; p = (mx.arange(n) + base).astype(mx.float32)[:, None]
        cache["calls"].append(n)
        return (mx.sin(t * 0.37 + p * 0.11 + mx.arange(V)[None, :] * 0.05))[None]
    def commit_verified_window(self, P, st, keep_tokens, verified_tokens):
        P["pos"] += keep_tokens; return True
class CS:
    @staticmethod
    def snapshot_untrimmable_cache_lazy(P): return types.SimpleNamespace(states=P["pos"])
    @staticmethod
    def rollback_after_verify(P, S0, n): P["pos"] -= n
ref2 = types.SimpleNamespace(CFG={"seam_hi": 16, "hc_b": False}, verify_scope=lambda m: contextlib.nullcontext())
fake = lambda: types.SimpleNamespace(ON=[False], CNT={})
def setup(G):
    E.S.update(ok=True, bofi=fake(), refc=fake(), v32one=fake(), seam0=16, hcb0=False)
E.setup = setup
G = dict(model=Model(), ref2=ref2, CS=CS, states=lambda P: [], say=print, OUT=__import__("tempfile").gettempdir())
for key in ("w1", "w2"):
    P = {"pos": 100, "calls": []}
    E.window(G, key, P, None, list(range(200, 300)), 10, None, key == "w1")
    print(key, "rows per call:", sorted(set(P["calls"])), "pos back to", P["pos"])
E.finish(G)
print("A8 records:", sum(1 for r in E.REC if r["arm"] == "A8"), "n:", {r["n"] for r in E.REC if r["arm"] in ("A8", "A8b")})
