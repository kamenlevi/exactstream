"""Fit12 R-59 DENSE16 (Fri 30 Oct 2026; NewIdeas QUEUE R-59, method OK 16:2x; PLAN.md 'R-59 DENSE16', written before any
run): where does the non-expert GPU work of a warm exact 16-row check go at the 45 GB config (C 224)?
Setup = Leap's R-50c (scripts/r50c_screen.py, imported tooling read-only): EvalSuite's REF harness (EM.load('ref', C)),
ref2.apply(), SB, PF1 prefill of HumanEval/68, warm 16-row checks sliding by one token along its REF2 row, commit keep 1.
(1) In place: DecoderLayer.__call__ is patched (16 rows run eagerly; mx.compile serves <= 4 rows only) to eval after each
    stage of every trunk layer: PLE, HC read (attention side), GDN or QSA attention (+ its cache writes), HC read (MLP side,
    with the pending write), MoE block (router + shared expert + routed experts through ref2's slot path), HC write; the
    head (norm + LM head) is the model call's return -> logits evaluated. Stage ms per check = its intervals' sum minus
    n_syncs x a bare sync's latency; medians over the profiled checks.
(2) Op groups (Leap's gdn16_profile method): the stateless stages from inputs captured in one check (HC read x2, HC write,
    router, shared expert), each run for all 48 layers in one eval under ref2.verify_scope; median of 15 minus one latency.
Floors: the parameter bytes each stage reads at 280 GB/s; GFLOP = 2 x linear MACs x 16 rows (attention / recurrence math
not counted). Usage: [BigMoE venv] fit_r59.py [C=224] [N=10] -> ~/Fit12/data/r59/r59_c<C>.txt"""
import json, os, sys, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # as R-50c: the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("~/Leap/third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

OUT = os.path.expanduser("~/Fit12/data/r59")
os.makedirs(OUT, exist_ok=True)
C = int(sys.argv[1]) if len(sys.argv) > 1 else 224
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
R, KEY, BW, REPS = 16, "humaneval/HumanEval/68", 280e9, 15
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
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
Q._prefill_midloop_eval_layers = lambda rows: Q_mid(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB, as R-50c


def find_layers(m):
    for path in ("language_model.model.layers", "model.layers", "layers"):
        o = m
        try:
            for a in path.split("."):
                o = getattr(o, a)
        except AttributeError:
            continue
        if isinstance(o, list) and o and isinstance(o[0], Q.DecoderLayer):
            return o, path
    raise SystemExit("no trunk DecoderLayer list found")


LAYERS, LPATH = find_layers(model)
IDX = {id(l): i for i, l in enumerate(LAYERS)}
say(f"R-59: full 125B via EM.load('ref', C = {C}), loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB "
    f"active; REF2 + SB, slot call {ref2.CFG['slot_call']}, commit phase {ref2.CFG['commit_phase']}, key buckets "
    f"{ref2.CFG.get('key_buckets')}; {KEY}; {len(LAYERS)} trunk layers at model.{LPATH} "
    f"({sum(l.is_linear for l in LAYERS)} GDN, {sum(not l.is_linear for l in LAYERS)} QSA, {sum('ple' in l for l in LAYERS)} PLE)")


# ---------------------------------------------------------------- floors
def pbytes(mods):
    return sum(v.nbytes for m in mods for _, v in tree_flatten(m.parameters()))


def macs(mods):                                          # linear layers' multiply-adds per row
    n = 0
    for m in mods:
        for _, sub in m.named_modules():
            if isinstance(sub, nn.QuantizedLinear):
                n += sub.weight.shape[0] * sub.weight.shape[1] * 32 // sub.bits
            elif isinstance(sub, nn.Linear):
                n += sub.weight.shape[0] * sub.weight.shape[1]
    return n


GMODS = {"hc_read_attn": [l.attn_hyper_connection for l in LAYERS], "hc_read_mlp": [l.mlp_hyper_connection for l in LAYERS],
         "gdn": [l.linear_attn for l in LAYERS if l.is_linear], "attn": [l.self_attn for l in LAYERS if not l.is_linear],
         "router": [l.mlp.gate for l in LAYERS], "shared": [m for l in LAYERS for m in (l.mlp.shared_expert, l.mlp.shared_expert_gate)],
         "ple": [l.ple for l in LAYERS if "ple" in l]}
TM = model.language_model if hasattr(model, "language_model") else model
TT = getattr(TM, "model", None)                          # the head: the final HC read (hyper_connection_mixer) + the LM head
HEAD = [m for m in (getattr(TM, "lm_head", None), getattr(TT, "hyper_connection_mixer", None), getattr(TT, "norm", None))
        if m is not None]
GMODS["head"] = HEAD
FLOOR = {k: (pbytes(v), macs(v)) for k, v in GMODS.items()}
say("  floors (params read once per check; GFLOP at 16 rows, linear layers only): " + "; ".join(
    f"{k} {b / 1e9:.3f} GB = {1e3 * b / BW:.2f} ms, {2 * mx_ * R / 1e9:.1f} GFLOP" for k, (b, mx_) in FLOOR.items()))


# ---------------------------------------------------------------- (1) in place
P = {"on": False, "rec": {}, "n": {}, "t": 0.0, "cap": None, "other": 0}
_orig = Q.DecoderLayer.__call__


def _arrs(c):
    out = []
    s = getattr(c, "state", None)
    for a in (s if isinstance(s, (list, tuple)) else [s]):
        if isinstance(a, mx.array):
            out.append(a)
    rows = getattr(c, "_mtplx_verify_rows", None)
    if isinstance(rows, (list, tuple)):
        out.extend(a for a in rows if isinstance(a, mx.array))
    return out


def _tick(name, *arrs):
    mx.eval(*arrs)
    t = pc()
    P["rec"][name] = P["rec"].get(name, 0.0) + (t - P["t"])
    P["n"][name] = P["n"].get(name, 0) + 1
    P["t"] = t


def _prof_call(self, hidden, *, input_ids, ssm_mask, cache):
    if not P["on"] or id(self) not in IDX:
        if P["on"]:
            P["other"] += 1
        return _orig(self, hidden, input_ids=input_ids, ssm_mask=ssm_mask, cache=cache)
    L, cap = IDX[id(self)], P["cap"]
    mx.eval(hidden)
    P["t"] = pc()
    if "ple" in self:
        hidden = hidden + self.ple(hidden, input_ids, cache)
        _tick("ple", hidden)
    if cap is not None:
        cap["hc_read_attn"][L] = hidden
    mixed, hyper, inject = self.attn_hyper_connection(hidden)
    _tick("hc_read_attn", mixed, hyper, inject)
    if self.is_linear:
        bo = self.linear_attn(mixed, ssm_mask, cache)
        _tick("gdn", bo, *_arrs(cache))
    else:
        bo = self.self_attn(mixed, cache)
        _tick("attn", bo, *_arrs(cache))
    if cap is not None:
        cap["hc_read_mlp"][L] = (hyper, bo, inject)
    mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(bo, inject))
    _tick("hc_read_mlp", mixed, hyper, inject)
    if cap is not None:
        cap["moe_in"][L] = mixed
    bo = self.mlp(mixed)
    _tick("moe_block", bo)
    if cap is not None:
        cap["hc_write"][L] = (hyper, bo, inject)
    hidden = Q._hyper_residual_write(hyper, bo, inject)
    _tick("hc_write", hidden)
    return hidden


Q.DecoderLayer.__call__ = _prof_call


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
for _ in range(200):                                     # a bare sync's latency
    z = mx.array([1.0]) + 1.0
    s0 = pc(); mx.eval(z); lat.append(pc() - s0)
LAT = med(lat)

paused = os.path.exists(PAUSE)
pf1(True)
cache = model.make_cache()
lg = model(mx.array([ids]), cache=cache)
mx.eval(lg, states(cache))
pf1(False)
CH, CAP = [], None
for i in range(N + 3):                                   # checks 0-1 warm up; the last one also captures the op-group inputs
    prof = i >= 2
    P.update(on=prof, rec={}, n={}, other=0)
    if i == N + 2:
        CAP = {"hc_read_attn": {}, "hc_read_mlp": {}, "moe_in": {}, "hc_write": {}}
        P["cap"] = CAP
    mx.synchronize()
    c0 = pc()
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with ref2.verify_scope(model):
        lg = model(mx.array([seq[i:i + R]]), cache=cache)
    c1 = pc()
    mx.eval(lg)
    c2 = pc()
    P["on"], P["cap"] = False, None
    model.commit_verified_window(cache, snap.states, keep_tokens=1, verified_tokens=R)
    mx.eval(states(cache))
    c3 = pc()
    paused = paused or os.path.exists(PAUSE)
    if prof and i < N + 2:
        row = {k: 1e3 * (v - P["n"][k] * LAT) for k, v in P["rec"].items()}
        row.update(head=1e3 * (c2 - c1 - LAT), check=1e3 * (c2 - c0), wall=1e3 * (c3 - c0), syncs=sum(P["n"].values()),
                   other=P["other"])
        CH.append(row)

say(f"  bare sync latency {1e3 * LAT:.3f} ms (median of 200); {len(CH)} profiled warm checks (C {C}), PAUSE seen {int(paused)}")
keys = ["ple", "hc_read_attn", "gdn", "attn", "hc_read_mlp", "moe_block", "hc_write", "head"]
say("  (1) in place, ms per check (median over checks; each stage's syncs x latency subtracted):")
for k in keys:
    v = [c.get(k, 0.0) for c in CH]
    fb = FLOOR.get(k, (0, 0))
    fl = 1e3 * fb[0] / BW
    say(f"    {k:13s} {med(v):7.2f} (min {min(v):6.2f} max {max(v):6.2f})" + (f" | floor {fl:5.2f} ms -> {med(v) / fl:4.1f}x" if fl > 0 else ""))
non = [sum(c.get(k, 0.0) for k in keys if k != "moe_block") for c in CH]
say(f"    non-expert sum (all but moe_block) {med(non):.2f} ms; check {med([c['check'] for c in CH]):.1f} ms with "
    f"{med([c['syncs'] for c in CH]):.0f} forced syncs (R-50c's unforced warm check: 181.5 ms at C 192); wall "
    f"{med([c['wall'] for c in CH]):.1f}; other DecoderLayer calls {max(c['other'] for c in CH)}")

# ---------------------------------------------------------------- (2) op groups
GRP = {
    "hc_read_attn": lambda: [LAYERS[L].attn_hyper_connection(h) for L, h in CAP["hc_read_attn"].items()],
    "hc_read_mlp": lambda: [LAYERS[L].mlp_hyper_connection(hy, pending=(bo, inj)) for L, (hy, bo, inj) in CAP["hc_read_mlp"].items()],
    "hc_write": lambda: [Q._hyper_residual_write(hy, bo, inj) for L, (hy, bo, inj) in CAP["hc_write"].items()],
    "router": lambda: [_route(LAYERS[L].mlp, x) for L, x in CAP["moe_in"].items()],
    "shared": lambda: [Q.attention_gate(LAYERS[L].mlp.shared_expert(x), LAYERS[L].mlp.shared_expert_gate(x)) for L, x in CAP["moe_in"].items()],
}


def _route(mlp, x):                                      # = SparseMoeBlock._verify_call's routing ops
    g = mx.softmax(mlp.gate(x), axis=-1, precise=True)
    inds = mx.argpartition(g, kth=-mlp.top_k, axis=-1)[..., -mlp.top_k:]
    sc = mx.take_along_axis(g, inds, axis=-1)
    if mlp.norm_topk_prob:
        sc = sc / sc.sum(axis=-1, keepdims=True)
    return inds, sc


say(f"  (2) op groups, all {len(LAYERS)} layers in one eval under ref2.verify_scope, median of {REPS} minus one latency:")
with ref2.verify_scope(model):
    for k, fn in GRP.items():
        mx.eval(fn())                                    # warm-up
        ts = []
        for _ in range(REPS):
            mx.synchronize()
            s0 = pc(); mx.eval(fn()); ts.append(pc() - s0 - LAT)
        fb = FLOOR.get(k, (0, 0))
        fl = 1e3 * fb[0] / BW
        say(f"    {k:13s} {1e3 * med(ts):7.2f} (min {1e3 * min(ts):6.2f})" + (f" | floor {fl:5.2f} ms -> {1e3 * med(ts) / fl:4.1f}x" if fl > 0 else ""))
paused = paused or os.path.exists(PAUSE)
say(f"  footprint now {footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB, MLX peak {mx.get_peak_memory() / 1e9:.1f} GB; "
    f"PAUSE seen {int(paused)}")
json.dump({"C": C, "lat_ms": 1e3 * LAT, "checks": CH, "floors": {k: list(v) for k, v in FLOOR.items()}},
          open(os.path.join(OUT, f"r59_c{C}.json"), "w"))
open(os.path.join(OUT, f"r59_c{C}.txt"), "w").write("\n".join(lines) + "\n")
