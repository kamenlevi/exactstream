"""R-50c (Kamen via the orchestrator, Thu 2026-10-29 22:1x: profile the ~200 ms check end to end before BigMoE builds a
sync-free 16-row gate; NewIdeas GO 22:08 for the split; Leap, pre-registered Thu 22:20 EET in ~/Leap/TABLE_350.md
'R-50c'): where do the ~200 ms of a warm exact 16-row check at the 45 GB configuration go?
Harness: EvalSuite's REF harness, C 192, PF1 for prefills only, REF2 + SB on with ref2's defaults (R-50's slot call, the
commit in the verify phase), no drafter. Warm 16-row checks sliding by one token along HumanEval/68's REF2 row,
commit keep 1 (R-50b's windows), 2 passes of 14.
Per MoE layer-call (ref2's slot path, instrumented copy):
  sync    np.array(indices): the CPU waits for the GPU to finish everything queued up to this layer's router
          (the previous layer's tail + this layer's attention / GDN / HC / router) plus the sync latency
  ensure  ext.ensure of the layer's distinct experts (CPU; SSD reads for misses, counted)
  build   the slot map + the MoE graph build (CPU)
  moe     mx.eval(y): the MoE kernels on the GPU + a second sync
  between from the previous layer's mx.eval(y) to this layer's call (CPU: Python builds this layer's dense graph)
Per check: the whole check (wall), head = model() return -> logits evaluated, commit (+ its eval), and a bare sync's
latency (mx.eval of a 1-element op) measured on its own.
GPU busy ~ sum(sync) + sum(moe) + head - syncs x latency; CPU-side bubble (GPU idle) ~ sum(ensure + build + between).
Lines (NewIdeas 22:08): PASS = bubble >= 80 ms per check -> BigMoE builds R-55 GATE16 (sync-free 16-row gate, REF2's
slot kernel unchanged); KILL = bubble < 40 ms -> compute is the lever (REF2 kernels); BETWEEN otherwise.
Usage: [BigMoE venv] r50c_screen.py [C=192] [N=14] -> ~/Leap/data/r50c_screen.txt"""
import json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 192
N = int(sys.argv[2]) if len(sys.argv) > 2 else 14
R = 16
KEY = "humaneval/HumanEval/68"
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")
pc = time.perf_counter

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
ref2.apply()
Q_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: Q_mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
say(f"R-50c: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; REF2 + SB, slot call {ref2.CFG['slot_call']}, commit phase {ref2.CFG['commit_phase']}; {KEY}")

T = {"on": False, "calls": [], "last": None}
def misses():
    try:
        return int(st.stats()["misses"])
    except Exception:
        return 0

def _prof_slots(cls, self, x, indices, bm, *a, **kw):  # = ref2._sync_slots with timers (bit-identical work)
    lead, rows = ref2._rows(x)
    if (not T["on"] or not ref2._ON[0] or not ref2.CFG["slot_call"] or rows > ref2.R or "tier" in bm or bm.get("mtp")):
        return ref2._sync_slots(cls, self, x, indices, bm, *a, **kw)
    t_in = pc()
    between = (t_in - T["last"]) if T["last"] is not None else 0.0
    ids = np.array(indices)
    t1 = pc()
    uniq = np.unique(ids)
    if len(uniq) > int(bm["C"] * 0.9):
        T["last"] = None
        return ref2._ORIG["sync"](cls, self, x, indices, bm, *a, **kw)
    if BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS.SYNC["calls"] += 1; BS.SYNC["rows"] += ids.size // ids.shape[-1]
    m0 = misses()
    slots = np.array(BS.ext.ensure(bm["L"], mx.array(uniq.astype(np.int32))))
    t2 = pc()
    sl = mx.array(slots[np.searchsorted(uniq, ids)].astype(np.int32)).astype(indices.dtype)
    y = cls.__call__(self, x, sl, *a, **kw)
    t3 = pc()
    mx.eval(y)
    t4 = pc()
    ref2.COUNTS["slot_calls"] += 1
    T["calls"].append(dict(L=bm["L"], between=between, sync=t1 - t_in, ensure=t2 - t1, build=t3 - t2, moe=t4 - t3,
                           miss=misses() - m0, uniq=len(uniq)))
    T["last"] = t4
    return y
BS._sync_call = _prof_slots

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

items = {it["key"]: it for it in ED.load("dec_humaneval") if it["key"] == KEY}
ids = list(items[KEY]["prompt_ids"])
seq = list({d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}[KEY]["completion_ids"])

lat = []
for _ in range(50):                                      # a bare sync's latency
    z = mx.array([1.0]) + 1.0
    s0 = pc(); mx.eval(z); lat.append(pc() - s0)
LAT = med(lat)

CH = []
for p in range(2):
    pf1(True)
    cache = model.make_cache()
    lg = model(mx.array([ids]), cache=cache)
    mx.eval(lg, states(cache))
    pf1(False)
    for i in range(N):
        T["calls"] = []; T["last"] = None; T["on"] = True
        mx.synchronize()
        c0 = pc()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        with ref2.verify_scope(model):
            lg = model(mx.array([seq[i:i + R]]), cache=cache)
        c1 = pc()
        mx.eval(lg)
        c2 = pc()
        T["on"] = False
        model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=R)
        mx.async_eval(states(cache))
        mx.eval(states(cache))
        c3 = pc()
        calls = T["calls"]
        tail = (c1 - T["last"]) if T["last"] is not None else 0.0   # after the last MoE call: final norm / head graph build
        if i >= 2:
            CH.append(dict(wall=c3 - c0, check=c2 - c0, head=c2 - c1, tail=tail, commit=c3 - c2, n=len(calls),
                           **{k: sum(c[k] for c in calls) for k in ("between", "sync", "ensure", "build", "moe", "miss")}))
    del cache
    mx.clear_cache()

ms = lambda k: 1e3 * med([c[k] for c in CH])
say(f"  bare sync latency {1e3 * LAT:.3f} ms; MoE layer-calls per check {med([c['n'] for c in CH]):.0f}; "
    f"misses per check {med([c['miss'] for c in CH]):.0f}; warm checks {len(CH)} (windows 2..{N - 1}, 2 passes)")
say(f"  per check (medians, ms): wall {ms('wall'):.1f} = check {ms('check'):.1f} + commit {ms('commit'):.1f}")
say(f"    inside the check: id syncs (GPU up to each router + latency) {ms('sync'):.1f}; MoE kernels + sync "
    f"{ms('moe'):.1f}; ensure (CPU + SSD) {ms('ensure'):.1f}; slot map + MoE graph build {ms('build'):.1f}; between "
    f"layers (Python builds the dense graph) {ms('between'):.1f}; after the last MoE layer to the return {1e3 * med([c['tail'] for c in CH]):.1f}; "
    f"head eval {ms('head'):.1f}")
n_sync = 2 * med([c["n"] for c in CH]) + 1
gpu = ms("sync") + ms("moe") + ms("head") - 1e3 * LAT * n_sync
bubble = ms("ensure") + ms("build") + ms("between") + 1e3 * med([c["tail"] for c in CH]) + 1e3 * LAT * n_sync
unexpl = ms("check") - (ms("sync") + ms("moe") + ms("ensure") + ms("build") + ms("between") + ms("head")
                        + 1e3 * med([c["tail"] for c in CH]))
say(f"  GPU busy ~{gpu:.1f} ms; CPU-side bubble (GPU idle: ensure + build + between + tail + {n_sync:.0f} sync latencies) "
    f"~{bubble:.1f} ms; commit {ms('commit'):.1f} ms; unexplained in the check {unexpl:.1f} ms")
v = "PASS (build GATE16)" if bubble >= 80 else "KILL (compute is the lever)" if bubble < 40 else "BETWEEN"
say(f"R-50c verdict (PASS: bubble >= 80 ms per check; KILL: < 40 ms): {v}; {time.time() - t0:.0f} s")
open(os.path.join(D, "r50c_screen.txt"), "w").write("\n".join(lines) + "\n")
