"""R-52b SCRATCH (NewIdeas Thu 2026-10-29 21:3x, Kamen 21:1x: 45 GB first): a reused scratch for the STOCK 16-row check
path at C 192. BigMoE's plain _sync_call gathers each layer's needed experts into a fresh scratch pool (9 mx.take, then
clear_cache, ~330 MB per layer call); Leap's R-50 measured that as ~8 ms per call, ~384 ms per check, mostly fresh
allocation / page faults. Stock MLX's sorted gather needs the expert ORDER, so slot ids are not an option off REF2.
BIGMOE_SCRATCH_KEEP (bigmoe/stream.py, switched here per pass): the single-group gather runs at a FIXED size (0.9 C
experts; extra rows repeat a slot and are never indexed) and that call's clear_cache is skipped, so MLX's buffer cache
hands the same 9 buffers back every call; the call sees [:n] views with the shapes it saw before.
Harness as Leap's R-50b (scripts/r50b_screen.py): EvalSuite's REF harness (load_streamed + its install), C = 192, PF1 for
prefills only, stock MLX (no REF2, no SB), no drafter; 16-row checks sliding by one token along each prompt's REF2 row
(EvalSuite results/ref2), commit keep 1, so each check's experts are almost all resident. Per prompt four passes over the
same N windows, each from a fresh prefill: HumanEval/68 base keep keep base, prealgebra/1640 keep base base keep.
Per window the check's logits are compared across all passes (bitwise); time = median over warm checks (window >= 2).
Lines (NewIdeas): PASS = every window's logits equal AND >= 40 ms saved per warm check (median base - median keep, per
prompt, median over the prompts); KILL = under 15 ms; else BETWEEN.
Usage: [BigMoE venv] r52b_screen.py [C=192] [N=14] -> ~/BigMoE/data/r52b/r52b_screen.txt"""
import hashlib, json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.pop("BIGMOE_SCRATCH_KEEP", None)            # switched per pass below
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb
import numpy as np
import mlx.core as mx

OUT = os.path.expanduser("~/BigMoE/data/r52b"); os.makedirs(OUT, exist_ok=True)
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
import bigmoe.stream as BS
from mtplx import cache_state as CS
say(f"R-52b: full 125B via load_streamed (C = {C}), loaded in {time.time() - t0:.0f} s, "
    f"{mx.get_active_memory() / 1e9:.1f} GB active; stock MLX (no REF2, no SB), PF1 {BS._PF_PREFILL} for prefills only; "
    f"MAXK {BS.MAXK}; {N} sliding windows per pass")
REF2ROWS = {d["key"]: d for d in map(json.loads, open(os.path.expanduser("~/EvalSuite/results/ref2/decode.jsonl")))}

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def misses():
    try:
        return int(st.stats()["misses"])
    except Exception:
        return 0

def one_pass(ids, seq, keep):
    BS._SCRATCH_KEEP = False
    pf1(True)
    cache = model.make_cache()
    lg = model(mx.array([ids]), cache=cache)
    mx.eval(lg, states(cache))
    pf1(False)
    BS._SCRATCH_KEEP = keep
    ts, hs, ms = [], [], []
    for i in range(N):
        m0 = misses()
        mx.synchronize()
        t = time.perf_counter()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        lg = model(mx.array([seq[i:i + R]]), cache=cache)
        mx.eval(lg)
        model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=R)
        mx.eval(states(cache))
        ts.append(1e3 * (time.perf_counter() - t))
        hs.append(hashlib.sha1(np.array(mx.view(lg[0].reshape(-1), mx.uint8)).tobytes()).hexdigest()[:16])
        ms.append(misses() - m0)
    cache_gb = mx.get_cache_memory() / 1e9
    BS._SCRATCH_KEEP = False
    del cache
    mx.clear_cache()
    return ts, hs, ms, cache_gb

RES, equal = [], True
for n_p, (ds, key) in enumerate(PICK):
    it = next(x for x in ED.load(ds) if x["key"] == key)
    ids, seq = list(it["prompt_ids"]), list(REF2ROWS[key]["completion_ids"])
    order = (False, True, True, False) if n_p % 2 == 0 else (True, False, False, True)   # keep arm = True
    P = {}
    for n, keep in enumerate(order):
        while os.path.exists(os.path.expanduser("~/.local/share/quiet/PAUSE")):
            time.sleep(5)
        ts, hs, ms, cg = one_pass(ids, seq, keep)
        P.setdefault(keep, []).append((ts, hs, ms))
        fp = footprint_now_gb()
        say(f"  {key} pass {n + 1} [{'keep' if keep else 'base'}]: check + commit ms per window {[round(x) for x in ts]}; "
            f"misses per check {ms}; MLX cache at the pass end {cg:.2f} GB; footprint now {fp}")
    hashes = [h for runs in P.values() for (_, h, _) in runs]
    eq = all(h == hashes[0] for h in hashes)
    equal &= eq
    warm = {k: [t for (ts, _, _) in runs for t in ts[2:]] for k, runs in P.items()}
    save = med(warm[False]) - med(warm[True])
    RES.append(save)
    say(f"  {key}: warm checks (windows >= 2, 2 passes each): base median {med(warm[False]):.1f} ms, keep median "
        f"{med(warm[True]):.1f} ms; saved {save:.1f} ms per 48-layer check; logits equal across all 4 passes for every "
        f"window: {eq}; misses per warm check base {np.mean([m for (_, _, ms) in P[False] for m in ms[2:]]):.1f}, keep "
        f"{np.mean([m for (_, _, ms) in P[True] for m in ms[2:]]):.1f}")
save = med(RES)
v = "FAIL (arms not bit-equal)" if not equal else "PASS" if save >= 40 else "KILL" if save < 15 else "BETWEEN"
say(f"R-52b verdict (PASS: bit-equal AND >= 40 ms saved per warm check at C {C}, median over the prompts; KILL: < 15 ms): "
    f"{v} (savings {[round(s, 1) for s in RES]} ms; {time.time() - t0:.0f} s)")
open(os.path.join(OUT, "r52b_screen.txt"), "w").write("\n".join(lines) + "\n")
