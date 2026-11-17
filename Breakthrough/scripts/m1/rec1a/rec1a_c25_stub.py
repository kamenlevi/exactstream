"""rec1a_c25_stub.py (Breakthrough, Tue 17 Nov 2026 07:2x; MODULE 1a milestone 2): runs rec1a_c25.run_test() on a toy
engine with c25's interfaces, so the engine-run wrapper is tested end to end without the 125B (< 1 GB, seconds).
The toy check, per pass (each piece mirrors a seq-dependent target of MODULE1A_ENGINE_MAP.md):
  s = the gate's pass counter (+1 per forward);
  (1) a seq word in setBytes (an int32 arange starting at s);
  (2) a table block read through a seq-dependent slice of BG.tables: a buffer offset ((s nl + 2) mod ntab) blocks;
  (3) a slot-ring read through an arange start ((s nl + 3) mod ring_rows) * maxk (setBytes);
  (4) offs = int32 [((s nl + 4) mod nreg) * 64, 7, s], built on the CPU per call (buffer words);
  (5) a system-scope post + a coherent spin against mtlrec's toy loader (one served row per pass; the ext's 'gated');
  (6) a KV write at a fixed prefix (the rollback keeps the prefix); (7) a recurrent state read from the snapshot and
  committed; float32 logits.
Each block / region / ring row holds its own index and the check subtracts the index it expects from the seq it read,
so the logits are seq-invariant when every target is patched consistently (as in the engine: seq picks where the
slots are read, not what they are), and a wrong rule shows.
STUB_MODE: ok (expect PASS); unexplained (an s*s word: expect line 1 FAIL on every arm); stale (the wrapper's
new_pass() returns the last seq: expect line 3 FAIL with line 2 still bitwise, Leap's point 4); badrule (the table
rule reads row L + 1: expect line 2 FAIL with lines 1, 3, 4 passing).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/m1/rec1a/rec1a_c25_stub.py   (STUB_MODE, STUB_OUT)"""
import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtlrec as R  # noqa: E402
R.load()
import rec1a_c25 as T  # noqa: E402
mx = T.bind_mx()
import numpy as np  # noqa: E402

MODE = os.environ.get("STUB_MODE", "ok")
OUT = os.path.expanduser(os.environ.get("STUB_OUT", os.path.join(HERE, "..", "..", "..", "data", "m1", "rec1a_engine")))
NL, TLN, NTAB, NREG, REG_INTS, MAXK, RROWS = 6, 3, 64, 16, 64, 32, 40
D, V, MAXT, OFF, NTOK = 64, 512, 256, 100, 1000
rng = np.random.default_rng(20261117)

HDR = """#pragma METAL internals : enable
#include <metal_stdlib>
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
namespace metal { constexpr constant thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }
#endif
"""
POST = """
  uint t = thread_position_in_grid.x;
  if (t == 0) {
    volatile coherent(system) device int* r = (volatile coherent(system) device int*)(row);
    int s = seqa[0];
    int v = int(x[0] * 1000.0f) & 0xFFFF;
    r[1] = v;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    r[0] = s;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    done[0] = v;
  }
"""
SPIN = f"""
  uint lane = thread_index_in_simdgroup;
  volatile coherent(system) device int* r = (volatile coherent(system) device int*)(rel);
  int s = seqa[0];
  int C = cap[0];
  int it = 0;
  bool got = false;
  for (; it < C; it++) {{
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    if (simd_all(r[0] == s)) {{ got = true; break; }}
  }}
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
  int slot = got ? r[1] : -1;
  for (uint j = lane; j < {D}; j += 32) y[j] = x[j] * float(slot) * 1e-4f + float(dep[0] & 1);
  if (lane == 0 && !got) atomic_fetch_add_explicit((device atomic_int*)const_cast<device int*>(status), 1, memory_order_relaxed);
"""
post_k = mx.fast.metal_kernel(name="bt_stub_post", input_names=["x", "row", "seqa"], output_names=["done"], header=HDR,
                              source=POST)
spin_k = mx.fast.metal_kernel(name="bt_stub_spin", input_names=["x", "rel", "seqa", "cap", "dep", "status"],
                              output_names=["y"], header=HDR, source=SPIN)


class Ext:
    @staticmethod
    def stats():
        return {"gated": R.toyloader_count()}


class Gate:
    def __init__(self):
        self.nl, self.ntab, self.nreg, self.maxk = NL, NTAB, NREG, MAXK
        # each block / region / ring row holds its own index, and the check subtracts the index it expects from the
        # seq it read (offs[2]): seq-invariant logits when every target is patched consistently, as in the engine
        # (seq picks WHERE the slots are read, not what they are); a wrong rule reads another index and shows
        self.tables = mx.array(np.repeat(np.arange(NTAB), TLN * 32).astype(np.int32))
        self.region = mx.array(np.repeat(np.arange(NREG), REG_INTS).astype(np.int32))
        self.ring = mx.array(np.repeat(np.arange(RROWS), MAXK).astype(np.int32))
        self.status = mx.zeros((4096,), mx.int32)
        self.row = mx.zeros((8,), mx.int32)
        self.rel = mx.zeros((8,), mx.int32)
        self.cap = mx.array([2_000_000], dtype=mx.int32)
        mx.eval(self.tables, self.region, self.ring, self.status, self.row, self.rel, self.cap)
        self.ext = Ext()
        self.seq = 1000                                   # the rounds cross 1024: seq and seq & 0x3FF must stay apart
        self.stale = False

    def pass_(self):                                      # the forward's own pass
        self.seq += 1
        return self.seq

    def new_pass(self):                                   # the wrapper's call; STUB_MODE=stale: the last seq again
        return self.seq if self.stale else self.pass_()


BG = Gate()
EMB = mx.array(rng.standard_normal((NTOK, D)).astype(np.float32) * 0.1)
WO = mx.array(rng.standard_normal((D, V)).astype(np.float32) * 0.1)
WQ = mx.array(rng.standard_normal((D, D)).astype(np.float32) * 0.1)
mx.eval(EMB, WO, WQ)
FORM = ["off"]


class Cache:
    def __init__(self):
        self.K = mx.array(rng.standard_normal((MAXT, D)).astype(np.float32) * 0.1)
        self.S = mx.array(rng.standard_normal((D,)).astype(np.float32))
        self.newS = None
        mx.eval(self.K, self.S)


class Model:
    def __call__(self, x_ids, cache):
        P = cache
        s = BG.pass_()
        n = x_ids.shape[1]
        e = EMB[x_ids[0]]                                                    # token ids: a per-call input
        offs = mx.array([((s * NL + 4) % NREG) * REG_INTS, 7, s], dtype=mx.int32)           # (4)
        sg = offs[2]                                                         # the seq as the GPU reads it
        a = (mx.arange(s, s + n, dtype=mx.int32) - sg).astype(mx.float32)[:, None] * 1e-2   # (1) 0..n-1 if consistent
        o = ((s * NL + 2) % NTAB) * TLN * 32
        tb = BG.tables[o:o + TLN * 32]                                                       # (2) block index
        tb = (tb[:D] - (sg * NL + 2) % NTAB).astype(mx.float32)[None, :] * 1e-1
        r0 = ((s * NL + 3) % RROWS) * MAXK
        rg = BG.ring[mx.arange(r0, r0 + n, dtype=mx.int32)] - (sg * NL + 3) % RROWS          # (3) 0 if consistent
        reg = mx.take(BG.region, offs[0] + mx.arange(n, dtype=mx.int32)) - (sg * NL + 4) % NREG
        sq = (rg + reg).astype(mx.float32)[:, None] * 1e-1
        seqa = mx.array([s], dtype=mx.int32)                                                 # (5)
        xin = mx.sigmoid(e[0]) * 0.8 + 0.1
        done = post_k(inputs=[xin, BG.row, seqa], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1,)],
                      output_dtypes=[mx.int32])[0]
        y = spin_k(inputs=[xin, BG.rel, seqa, BG.cap, done, BG.status], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(D,)], output_dtypes=[mx.float32])[0]
        bad = 0.0
        if MODE == "unexplained":
            q = (s * s) % 977
            bad = mx.arange(q, q + n, dtype=mx.int32).astype(mx.float32)[:, None] * 1e-5
        kv = e @ WQ                                                                          # (6)
        P.K[OFF:OFF + n] = kv
        Kp = P.K[:OFF + n]
        att = mx.softmax((e @ WQ) @ Kp.T / 8.0, axis=-1) @ Kp
        P.newS = P.S * 0.5 + e.mean(axis=0)                                                 # (7)
        h = att + P.newS + a + tb + sq + y[None, :] + bad
        if FORM[0] == "F32c6i":
            h = h + mx.tanh(h) * 0.01
        return (h @ WO)[None]

    def commit_verified_window(self, P, states, keep_tokens, verified_tokens):
        P.S = P.newS
        return True


class Snap:
    def __init__(self, states):
        self.states = states


class CS:
    @staticmethod
    def snapshot_untrimmable_cache_lazy(P):
        return Snap([P.S])


class Ref2:
    @staticmethod
    def verify_scope(model):
        return contextlib.nullcontext()


def states(P):
    return [P.K, P.S]


class CE:                                                 # c25_engine's interface: knobs / unknob / back / check
    @staticmethod
    def knobs(G, arm, n):
        FORM[0] = arm

    @staticmethod
    def unknob(G):
        FORM[0] = "off"

    @staticmethod
    def back(G, P, S0, n):
        P.S = S0.states[0]
        mx.eval(G["states"](P))

    @staticmethod
    def check(G, P, toks, n, arm, key, phase, rnd=-1):
        model, ref2, cs, st = G["model"], G["ref2"], G["CS"], G["states"]
        CE.knobs(G, arm, n)
        try:
            sn = cs.snapshot_untrimmable_cache_lazy(P)
            mx.synchronize()
            with ref2.verify_scope(model):
                lg = model(mx.array([toks]), cache=P)
            lg = lg[0].astype(mx.float32)
            mx.eval(lg)
            assert model.commit_verified_window(P, sn.states, keep_tokens=n, verified_tokens=n)
            mx.eval(st(P))
            mx.synchronize()
        finally:
            CE.unknob(G)
        return lg


def addr(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint8).ctypes.data


if MODE == "badrule":                                     # line 2's negative control: the table rule reads row L + 1
    _fit = T.fit

    def _bad_fit(*a, **k):
        out = _fit(*a, **k)
        return [(nm + " (+1: badrule)", dict(kw, b=kw["b"] + 1)) if nm.startswith("tab") else (nm, kw) for nm, kw in out]
    T.fit = _bad_fit
rc = R.toyloader_start(addr(BG.row), addr(BG.rel), 3, 1)
assert rc == 0, rc
G = dict(model=Model(), ref2=Ref2, CS=CS, states=states, BG=BG)
P = Cache()
S0 = CS.snapshot_untrimmable_cache_lazy(P)
gids = rng.integers(0, NTOK, 64).tolist()
BG.stale = MODE == "stale"
C = dict(arms=["F32o", "F32c6i"], rounds=16, retain_gb=1.0, out=OUT, tag=f"stub_{MODE}")
verdict = T.run_test(CE, G, "stubwin", P, S0, gids, 0, C)
served = R.toyloader_stop()
lines = {a: [bool(T.RESULT["arms"].get(a, {}).get(f"line{i}")) for i in (1, 2, 3, 4)] for a in C["arms"]}
want = {"ok": ([True] * 4, "PASS"), "unexplained": ([False, False, False, False], "FAIL"),
        "stale": ([True, True, False, True], "FAIL"), "badrule": ([True, False, True, True], "FAIL")}[MODE]
as_expected = verdict == want[1] and all(v == want[0] for v in lines.values())
T.say(f"STUB {MODE}: verdict {verdict}, lines {lines}; expected {want[1]} with lines {want[0]} on every arm -> "
      f"{'AS EXPECTED' if as_expected else 'NOT AS EXPECTED'}; toy loader served {served} rows")
T.write_out(C)
