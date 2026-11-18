# mock of e15_engine's sw mode: fake ref2 (v3-like CFG / apply / _bof_on / ar_step), fake bofi, fake model whose logits depend
# only on (token, position), so on == off and SD == AR hold; BUG=1 makes a 17-32-row prefill chunk under bofi.ON fire BOF2 and
# change its bits (the v1 bug) -> the verdict must be FAIL.
import os, sys, types, contextlib
os.environ["E15_MODE"] = "sw"; os.environ["E14_ARMS"] = "A16"
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98q/Leap_scripts/week98/e15"))
import mlx.core as mx
BUG = os.environ.get("BUG") == "1"
bofi = types.ModuleType("bofi"); bofi.ON = [False]; bofi.CNT = dict(bof=0, bof2=0, drift=0); bofi.TABLE32 = {}; bofi.__file__ = "fake"
bofi.self_test = lambda: None
sys.modules["bofi"] = bofi
PH = ["prefill"]
class Ref2:
    CFG = {"bof": True, "hc_b": True, "seam_hi": 16}; _ON = [True]; COUNTS = {"hc_b": 0}
    def _bof_on(self): pass
    def apply(self):
        bofi.ON[0] = bool(self.CFG["bof"]); self._ON[0] = True
    @contextlib.contextmanager
    def verify_scope(self, m):
        PH[0] = "verify"; yield; PH[0] = "prefill"
    def ar_step(self, model, cache, token):
        with self.verify_scope(model):
            lg = model(mx.array([[token]]), cache=cache)
        model.commit_verified_window(cache, None, keep_tokens=1, verified_tokens=1)
        return lg[0, -1]
ref2 = Ref2(); ref2.apply()
V = 48
class Model:
    def __call__(self, x, cache=None):
        n = x.shape[1]; base = cache["pos"]
        if bofi.ON[0]:
            if n <= 16: bofi.CNT["bof"] += 1
            elif n <= 32 and (PH[0] == "verify" or BUG): bofi.CNT["bof2"] += 1
        if ref2.CFG["hc_b"] and PH[0] == "verify": ref2.COUNTS["hc_b"] += 1
        t = x[0].astype(mx.float32)[:, None]; p = (mx.arange(n) + base).astype(mx.float32)[:, None]
        lg = mx.sin(t * 0.37 + p * 0.11 + mx.arange(V)[None, :] * 0.05)
        if BUG and bofi.ON[0] and PH[0] == "prefill" and 16 < n <= 32: lg = lg + 1e-3
        if PH[0] == "prefill": cache["h"] = cache.get("h", 0) + float(lg.sum().item())
        return lg[None]
    def commit_verified_window(self, P, st, keep_tokens, verified_tokens):
        P["pos"] += keep_tokens; return True
class CS:
    @staticmethod
    def snapshot_untrimmable_cache_lazy(P): return types.SimpleNamespace(states=P["pos"])
    @staticmethod
    def rollback_after_verify(P, S0, n): P["pos"] -= n
model = Model()
def prefill(ids, want_w=False, chunk=1024):
    P = {"pos": 0}; lg = None
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=P); P["pos"] += len(ids[a:a + chunk])
    return P, lg[0, -1], None, []
import e15_engine as E
G = dict(model=model, ref2=ref2, CS=CS, states=lambda P: [mx.array(P["pos"]), mx.array(P.get("h", 0.0))], say=print,
         prefill=prefill, set_arm=lambda a: None, BS=types.SimpleNamespace(GATE_N=[0]), BG=types.SimpleNamespace(MAXK=320),
         PFCHUNK=1024, OUT=__import__("tempfile").gettempdir())
E.cold(G, "w1", None, None, None, None)
for key, gen in (("w1", 378), ("w2", 255)):
    P = {"pos": gen}
    E.window(G, key, P, None, list(range(1000, 1500)), gen, None, key == "w1")
    assert P["pos"] == gen, P
E.finish(G)
