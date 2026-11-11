"""R-55 GATE16 (NewIdeas GO Thu 2026-10-29 22:50, after 00:00; Leap's R-50c split): the sync-free 16-row gate on the 45 GB
exact path. A 16-row check carries 16 x top-10 = 160 expert ids per MoE layer, over the gate's 92-id routing row, so it
took BigMoE's plain _sync_call (= Leap's REF2 slot call, R-50): per layer np.array(indices) waits for the GPU, ext.ensure
loads the misses, Python maps ids -> slots, then REF2's experts kernel runs on the pool slots. R-55 widens the gate
(ext/bigmoe_ext.cpp + bigmoe/gated.py: BIGMOE_MAXK up to 185 = 4 more id lines per routing row and 6-line release
entries; MAXK <= 92 keeps its layout) so the 16-row check takes the sync-free gate: the GPU posts the ids, the native
loader makes them resident and writes the slots, the GPU spins on them, and REF2's experts kernel runs on the same pool
slots (R-50: each (row, expert) pair on its own, so bit-equal by construction).
Harness = Leap's R-50c (~/Leap/scripts/r50c_screen.py): EvalSuite's REF harness, C 192, PF1 for prefills only, REF2 + SB
with ref2's defaults (slot call, commit phase), no drafter; warm 16-row checks sliding by one token along each prompt's
REF2 row (EvalSuite results/ref2), commit keep 1.
Arms in one process (BIGMOE_MAXK=160): slot = calls over 40 ids take _sync_call (stream.GATE_N = 40: the shipped 45 GB
path); gate = calls up to 160 ids take the gate. HumanEval/68 slot gate gate slot, prealgebra/1640 gate slot slot gate;
each pass a fresh prefill and N windows; logits compared bitwise per window across all four passes.
Lines (NewIdeas 22:50): PASS = every logits hash equal AND warm check + commit (windows >= 2) <= 135 ms in the gate arm
(median over the prompts); KILL = > 165 ms; else BETWEEN. A spin-gate timeout or a PAUSE inside a pass invalidates it.
Usage: [BigMoE venv] BIGMOE_MAXK=160 BIGMOE_EXT_BUILD=build_next r55_screen.py [C=192] [N=14] -> data/r55/r55_screen.txt"""
import hashlib, json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in R-50c
os.environ.setdefault("HF_HUB_OFFLINE", "1")
assert os.environ.get("BIGMOE_MAXK") == "160", "run with BIGMOE_MAXK=160 (the widened gate rows)"
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/BigMoE/data/r55"); os.makedirs(OUT, exist_ok=True)
C = int(sys.argv[1]) if len(sys.argv) > 1 else 192
N = int(sys.argv[2]) if len(sys.argv) > 2 else 14
R = 16
PICK = (("dec_humaneval", "humaneval/HumanEval/68"), ("dec_math", "math500/test/prealgebra/1640.json"))
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
import ref2
import bigmoe.stream as BS
import bigmoe.gated as BG
from mtplx import cache_state as CS
Q = ref2.Q
ref2.apply()
Q_mid = Q._prefill_midloop_eval_layers
Q._prefill_midloop_eval_layers = lambda rows: Q_mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
assert BG.MAXK == 160 and BG.XLINES == 4, (BG.MAXK, BG.XLINES)
say(f"R-55: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; REF2 + SB, slot call {ref2.CFG['slot_call']}, commit phase {ref2.CFG['commit_phase']}; gate rows MAXK "
    f"{BG.MAXK} (XLINES {BG.XLINES}, ext {BG._BUILD}); {N} sliding windows per pass")
REF2ROWS = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def stat():
    s = st.stats()
    return int(s["misses"]), float(s["io_s"]), int(s["gated"])

def one_pass(ids, seq, arm):
    BS.GATE_N[0] = 40
    pf1(True)
    cache = model.make_cache()
    lg = model(mx.array([ids]), cache=cache)
    mx.eval(lg, states(cache))
    pf1(False)
    BS.GATE_N[0] = 160 if arm == "gate" else 40
    p0, g0 = BG.PAUSED["pause_s"], BG.PAUSED["n_gap"]
    ts, hs, ms, io, ng = [], [], [], [], []
    bad = ""
    for i in range(N):
        m0, i0, n0 = stat()
        mx.synchronize()
        t = time.perf_counter()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        with ref2.verify_scope(model):
            lg = model(mx.array([seq[i:i + R]]), cache=cache)
        mx.eval(lg)
        model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=R)
        mx.async_eval(states(cache))
        mx.eval(states(cache))
        ts.append(1e3 * (time.perf_counter() - t))
        hs.append(hashlib.sha1(np.array(mx.view(lg[0].reshape(-1), mx.uint8)).tobytes()).hexdigest()[:16])
        m1, i1, n1 = stat()
        ms.append(m1 - m0); io.append(1e3 * (i1 - i0)); ng.append(n1 - n0)
    try:
        st.gate.check()
    except RuntimeError as e:
        bad = f"INVALID ({e})"
    if BG.PAUSED["pause_s"] - p0 > 0.5 or BG.PAUSED["n_gap"] > g0:
        bad += f" PAUSED ({BG.PAUSED['pause_s'] - p0:.1f} s waited, {BG.PAUSED['n_gap'] - g0} gaps)"
    BS.GATE_N[0] = 40
    del cache
    mx.clear_cache()
    return ts, hs, ms, io, ng, bad

RES, equal, valid = [], True, True
for n_p, (ds, key) in enumerate(PICK):
    it = next(x for x in ED.load(ds) if x["key"] == key)
    ids, seq = list(it["prompt_ids"]), list(REF2ROWS[key]["completion_ids"])
    order = ("slot", "gate", "gate", "slot") if n_p % 2 == 0 else ("gate", "slot", "slot", "gate")
    P = {}
    for n, arm in enumerate(order):
        while os.path.exists(os.path.expanduser("~/.local/share/quiet/PAUSE")):
            time.sleep(5)
        ts, hs, ms, io, ng, bad = one_pass(ids, seq, arm)
        valid &= not bad
        P.setdefault(arm, []).append((ts, hs, ms, io))
        say(f"  {key} pass {n + 1} [{arm}]: check + commit ms per window {[round(x) for x in ts]}; misses {ms}; SSD ms "
            f"{[round(x) for x in io]}; loader-served calls per check {ng}; footprint now {footprint_now_gb()} {bad}")
    hashes = [h for runs in P.values() for (_, h, _, _) in runs]
    eq = all(h == hashes[0] for h in hashes)
    equal &= eq
    warm = {k: [t for (ts, _, _, _) in runs for t in ts[2:]] for k, runs in P.items()}
    wio = {k: [x for (_, _, _, io) in runs for x in io[2:]] for k, runs in P.items()}
    wm = {k: [x for (_, _, ms, _) in runs for x in ms[2:]] for k, runs in P.items()}
    RES.append((med(warm["gate"]), med(warm["slot"])))
    say(f"  {key}: warm checks (windows >= 2, 2 passes each): slot median {med(warm['slot']):.1f} ms, gate median "
        f"{med(warm['gate']):.1f} ms ({med(warm['slot']) - med(warm['gate']):+.1f} saved); misses per warm check slot "
        f"{np.mean(wm['slot']):.1f}, gate {np.mean(wm['gate']):.1f}; SSD ms per warm check slot {med(wio['slot']):.1f}, "
        f"gate {med(wio['gate']):.1f}; logits equal across all 4 passes for every window: {eq}")
g = med([r[0] for r in RES]); s = med([r[1] for r in RES])
v = ("INVALID (a pass timed out or paused)" if not valid else "FAIL (arms not bit-equal)" if not equal
     else "PASS" if g <= 135 else "KILL" if g > 165 else "BETWEEN")
say(f"R-55 verdict (PASS: bit-equal AND gate warm check + commit <= 135 ms at C {C}, median over the prompts; KILL: > 165 "
    f"ms): {v} (gate {g:.1f} ms vs slot {s:.1f} ms; {time.time() - t0:.0f} s)")
open(os.path.join(OUT, "r55_screen.txt"), "w").write("\n".join(lines) + "\n")
