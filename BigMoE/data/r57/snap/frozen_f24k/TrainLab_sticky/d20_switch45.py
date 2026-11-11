"""D-20 SWITCH45 (MTP Head, Fri 10-30; NewIdeas' QUEUE row D-20; harness = a copy of Leap's live loop
/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts/r40s3k_screen.py at faff8e5, speed mode; Leap reviews it). The text-type switch in the 45 GB loop: DFlash2
L1-20 on code / math turns, the native MTP chain (K drafts, K + 1-row checks) on chat / instruction / agent turns, the
type from NewIdeas' prompt rule (scripts/type_rule.py, copied below). Every check is REF2b's exact path (/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts/
ref2.py, imported read-only; Leap: exact at any row count up to 16, rows padded to 16 inside each call).
Per prompt, interleaved arms on the same prompt: 'dfl' (block 16, the loop today), 'mtp7' / 'mtp15' (the MTP chain,
stock head; D20_HEAD=<run> installs a D-22 head from ~/TrainLab/ckpt instead). The MTP head's history: the prompt's
pairs (trunk widened stream at u, token u+1) from each prefill chunk, then after each check the kept rows' pairs (spec125
._MTPHead semantics; the chain's own cache entries are trimmed after drafting). Drafting time includes that history
append. Exactness: every arm's tokens equal (and EvalSuite's REF2 row where one exists).
PASS (NewIdeas): chat / agent mean tok/s >= +10 % with drafting cost, code / math unchanged (they stay on DFlash2).
KILL: < +4 %. Usage: [BigMoE venv] d20_switch45.py [C=224] [CAP=256] [ARMS=dfl,mtp7,mtp15]
Resumable: one line per (prompt, arm) in ~/TrainLab/data/d20_runs.jsonl; done pairs are skipped."""
import json, os, re, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_TIER", None)
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["TL_INFER"] = "1"                            # mtp125 (only with D20_HEAD): keep MTPLX's fused HC read
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF
from tokenizers import Tokenizer

D = os.path.expanduser("~/Leap/data")
TD = os.path.expanduser("~/TrainLab/data")
OUT = os.path.join(TD, "d20_runs.jsonl")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 224
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "dfl,mtp7,mtp15").split(",")
HEAD = os.environ.get("D20_HEAD", "")
R, K16 = 16, 16
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PICK = os.environ.get("D20_KEYS", "humaneval/HumanEval/68,humaneval/HumanEval/91,gsm8k/708,math500/test/algebra/1282.json,"
                      "ifeval/1220,ifeval/3280,ifeval/1561,multi/bg0,multi/de3,agent/bash17,agent/edit00").split(",")
BUDGET, HARD = 700, 740                                   # s of running time: no new run / no new check after these
ACT = [0.0]
def _tick():                                             # running time (a quiet window's SIGSTOP freezes it too)
    last = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        ACT[0] += min(now - last, 3.0)
        last = now
threading.Thread(target=_tick, daemon=True).start()
el = lambda: ACT[0]
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v)) if len(v) else float("nan")

# ---------------------------------------------------------------- NewIdeas' type rule (scripts/type_rule.py, copied)
TOK = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
CODE = re.compile(r"(^|\n)\s*(def |class |import |from \w+ import|#include|fn |func |public |const |let |var )|```|\breturn\b|\w+\(.*\)\s*:|>>>")
MATH = re.compile(r"\\boxed|\\frac|\\sqrt|\$[^$]+\$|\b(solve|compute|calculate|how many|how much|what is the (sum|product|value|area|probability))\b"
                  r"|\d+\s*[-+*/^=]\s*\d+", re.I)
TOOLS = re.compile(r"<tools>|\"type\":\s*\"function\"|tool_call|<function|\bfunctions?\b.*\bparameters\b", re.I | re.S)
def rule(item):
    full = item.get("prompt") or TOK.decode(item["prompt_ids"], skip_special_tokens=False)
    user = item.get("user") or full
    if TOOLS.search(full):
        return "agent"
    if CODE.search(user) or CODE.search(full[-4000:]):
        return "code"
    if MATH.search(user):
        return "math"
    return "chat"

# ---------------------------------------------------------------- the model, with its MTP head (the REF arm loads mtp=False)
import bigmoe.loader as BL
_ls = BL.load_streamed
BL.load_streamed = lambda md, C_, mtp=False: _ls(md, C_, mtp=True)
t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)                # EvalSuite's REF arm: load_streamed + its install + configure
BL.load_streamed = _ls
model = rt.model
assert getattr(model.language_model, "mtp", None) is not None, "no MTP head loaded"
import ref2
import bigmoe.stream as BS
from mtplx import cache_state as CS
Q = ref2.Q
TXT = model.language_model.model
if HEAD:
    sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/TrainLab_root"))
    import mtp125 as M
    T_ = M.Trainable(model.language_model.mtp)
    T_.load_weights(os.path.expanduser(f"~/TrainLab/ckpt/d22_{HEAD}.safetensors"))
    M.install(model.language_model.mtp, T_, dtype=mx.bfloat16)
say(f"D-20 SWITCH45: full 125B via load_streamed (C = {C}) with the MTP head ({HEAD or 'stock'}), loaded in "
    f"{time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB active; arms {ARMS}; {CAP} tokens")
ref2.apply()
st_ok = ref2.self_test()
ref2.remove(); ref2.reset_counts()
say(f"  ref2 self-test: {st_ok}")

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

def stats():
    try:
        s = st.stats()
        return np.array([s["misses"], s["io_s"]], dtype=np.float64)
    except Exception:
        return np.zeros(2)

# ---------------------------------------------------------------- taps for DFlash2 (Leap's hook)
TAP = {"on": False, "x": [[] for _ in TAPS]}
lay = TXT.layers
TAP_OF = {id(lay[L]): i for i, L in enumerate(TAPS)}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    out = _layer_call(self, *a, **kw)
    i = TAP_OF.get(id(self))
    if TAP["on"] and i is not None:
        o = out if isinstance(out, mx.array) else out[0]
        B, L, W = o.shape
        TAP["x"][i].append(o.reshape(B, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
    return out
Q.DecoderLayer.__call__ = _layer
def tap_reset():
    TAP["x"] = [[] for _ in TAPS]
def taps():
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)

# ---------------------------------------------------------------- DFlash2 L1-20 (Leap's eager drafter)
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
draft.bind(model)
mx.eval(draft.parameters())
MASK, SW = int(draft.config.mask_token_id), int(draft.config.sliding_window)

def head2s(hid):
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K16, axis=-1)[..., -K16:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K16, B)), axis=-2).reshape(*s[:-1], K16 * B)
    j = mx.argpartition(sel, -K16, axis=-1)[..., -K16:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)

def chain(blk, unary, cand, hp):
    sel = draft.candidate_selector
    pred, path = blk[:, 0], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)

def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))

# ---------------------------------------------------------------- the native MTP head (spec125._MTPHead's semantics)
class MTPHead:
    def __init__(self):
        self.cache = model.make_mtp_cache()
        self.logits = self.h = None

    def append(self, hidden, next_ids):
        """commit pairs (hidden row i, next_ids[i]); the last one runs through mtp_forward for the next draft"""
        if hidden.shape[1] > 1:
            model.mtp_update_cache(hidden[:, :-1], mx.array([next_ids[:-1]]), mtp_cache=self.cache)
        self.logits, self.h = model.mtp_forward(hidden[:, -1:], mx.array([next_ids[-1:]]), mtp_cache=self.cache,
                                                return_hidden=True)

    def draft(self, k):
        toks, h = [mx.argmax(self.logits[0, -1].astype(mx.float32))], self.h
        for _ in range(k - 1):
            lgx, h = model.mtp_forward(h, toks[-1].reshape(1, 1), mtp_cache=self.cache, return_hidden=True)
            toks.append(mx.argmax(lgx[0, -1].astype(mx.float32)))
        mx.eval(toks)
        if k > 1:
            self.cache[0].trim(k - 1)                    # the chain's own entries are not committed history
        return [int(t.item()) for t in toks]

_midloop = Q._prefill_midloop_eval_layers
def arm_on(ref):
    if ref:
        ref2.apply()
        Q._prefill_midloop_eval_layers = lambda rows: _midloop(rows) or (4 if 2 <= int(rows) <= 16 else 0)
    else:
        ref2.remove()
        Q._prefill_midloop_eval_layers = _midloop

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

def prefill(ids, want_w, chunk=1024):
    """Leap's prefill (chunks of 1024 on EvalSuite's PF1 path, taps), plus each chunk's widened stream for the MTP head"""
    pf1(True)
    cache, lg, ws = model.make_cache(), None, []
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        mx.eval(lg, states(cache), [x[-1] for x in TAP["x"]])
        if want_w:
            w = TXT._last_widened
            assert w.shape[1] == min(chunk, len(ids) - a), w.shape
            mx.eval(w)
            ws.append((a, w))
    TAP["on"] = False
    pf1(False)
    return cache, lg[0, -1], taps(), ws

def run(ids, arm):
    while os.path.exists(PAUSE):
        time.sleep(5)
    mtp = arm.startswith("mtp")
    KM = int(arm[3:]) if mtp else 0
    paused, cut = False, False
    arm_on(True)
    ref2.reset_counts()
    mx.synchronize()
    t1 = time.perf_counter()
    cache, last, T, ws = prefill(ids, mtp)
    pf = time.perf_counter() - t1
    P = len(ids)
    tok = int(mx.argmax(last).item())
    out, cyc, dr_t, ck_t, kept, rows_l = [tok], [], [], [], [], []
    hist_t = 0.0
    if mtp:                                              # the prompt's pairs (W[u], token u+1); the last pair uses tok
        ref2.remove()
        t2 = time.perf_counter()
        head, full = MTPHead(), list(ids) + [tok]
        for a, w in ws:
            head.append(w, full[a + 1:a + 1 + w.shape[1]])
            mx.eval(head.logits)
        hist_t = time.perf_counter() - t2
        ref2.apply()
        pend = None
    else:
        dcache = draft.make_cache()
        start = max(0, P - (SW - 1))
        for c in dcache:
            c.offset = start
        ctx = T[start:P]
    a = P
    s0 = stats()
    td = c_prev = time.perf_counter()
    while len(out) < CAP and tok not in EOS:
        if el() > HARD:
            cut = True
            break
        c0 = time.perf_counter()
        if out[1:]:
            cyc.append(c0 - c_prev)
        c_prev = c0
        ref2.remove()
        if mtp:
            if pend is not None:                         # the last check's kept rows -> the head's history
                head.append(*pend)
            dr = head.draft(KM)                          # always K (Leap: checks of <= 4 rows may take a compiled path)
        else:
            dr = np.array(propose(mx.array([[tok] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)[0]).tolist()
            if (trim := dcache[0].offset - a) > 0:
                DF._trim_recent_cache(dcache, trim)
        ref2.apply()
        c1 = time.perf_counter()
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        if not mtp:
            tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([[tok] + dr]), cache=cache)
        wv = TXT._last_widened if mtp else None
        TAP["on"] = False
        am = mx.argmax(lg[0], axis=-1)
        tp = None if mtp else taps()
        mx.eval(am, *( [wv[:, :len(dr) + 1]] if mtp else [tp]))
        am = am.tolist()
        k = 0
        while k < len(dr) and dr[k] == am[k]:
            k += 1
        model.commit_verified_window(cache, snap.states, keep_tokens=k + 1, verified_tokens=len(dr) + 1)
        mx.async_eval(states(cache))                     # SB: the commit replay submitted alone
        c2 = time.perf_counter()
        dr_t.append(c1 - c0); ck_t.append(c2 - c1); kept.append(k + 1); rows_l.append(len(dr) + 1)
        new = dr[:k] + [am[k]]
        for j, x in enumerate(new):
            if x in EOS:
                new = new[:j + 1]
                break
        if mtp:                                          # rows 0..len(new)-1 hold the pairs (W[a + i], new[i])
            pend = (wv[:, :len(new)], new)
        out += new
        tok, a = out[-1], a + k + 1
        if not mtp:
            ctx = tp[:k + 1]
        paused = paused or os.path.exists(PAUSE)
    mx.eval(states(cache))
    dec = time.perf_counter() - td
    ms, io = (stats() - s0) / max(len(kept), 1)
    arm_on(False)
    del cache
    mx.clear_cache()
    out = out[:CAP]
    return dict(out=out, pf=pf, hist=hist_t, dec=dec, n=len(out), checks=len(kept), kept=float(np.mean(kept)) if kept else 0.0,
                rows=float(np.mean(rows_l)) if rows_l else 0.0, cyc=1e3 * med(cyc), dr=1e3 * med(dr_t),
                ck=1e3 * med(ck_t), paused=paused, cut=cut, ms=float(ms), io=1e3 * float(io))

# ---------------------------------------------------------------- driver: interleaved arms per prompt, resumable
RES = os.path.expanduser("~/EvalSuite/results")
REFROWS = {d["key"]: d["completion_ids"] for d in map(json.loads, open(f"{RES}/ref2/decode.jsonl"))}
items = {}
SETOF = {"humaneval": "humaneval", "math500": "math", "gsm8k": "math", "ifeval": "ifeval", "multi": "multi", "agent": "agent"}
for key in PICK:
    s = SETOF[key.split("/")[0]]
    items.update({it["key"]: it for it in ED.load("dec_" + s) if it["key"] == key})
assert all(k in items for k in PICK), [k for k in PICK if k not in items]
done = {}
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l)
        done[(d["key"], d["arm"], d.get("head", "stock"), d.get("C"))] = d
same = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

say("  warm-up (discarded): " + PICK[0])
run(list(items[PICK[0]]["prompt_ids"])[:256], "dfl")
for key in PICK:
    it = items[key]
    typ = rule(it)
    for arm in ARMS:
        if (key, arm, HEAD or "stock", C) in done:
            continue
        if el() > BUDGET:
            say(f"  time budget: stopped before {key} {arm} (resumable)")
            break
        r = run(list(it["prompt_ids"]), arm)
        ref_ok = None
        if key in REFROWS:
            m = min(len(REFROWS[key]), r["n"])
            ref_ok = same(r["out"], REFROWS[key]) >= m
        rec = dict(key=key, arm=arm, type=typ, head=HEAD or "stock", C=C, cap=CAP, tok_s=r["n"] / r["dec"],
                   **{k_: r[k_] for k_ in ("n", "checks", "kept", "rows", "cyc", "dr", "ck", "pf", "hist", "ms", "io", "paused", "cut")},
                   ref2_row=ref_ok, out=r["out"])
        done[(key, arm, HEAD or "stock", C)] = rec
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        say(f"  {key} [{typ}] {arm}: {r['n']} tokens, {rec['tok_s']:.1f} tok/s, kept {r['kept']:.2f} of {r['rows']:.1f} rows, "
            f"median cycle {r['cyc']:.1f} ms (draft {r['dr']:.1f} + check {r['ck']:.1f}); prefill {r['pf']:.1f} s"
            + (f" + MTP history {r['hist']:.2f} s" if arm.startswith("mtp") else "")
            + f"; == REF2 row: {ref_ok}" + ("; PAUSED" if r["paused"] else "") + ("; CUT" if r["cut"] else ""))
    else:
        continue
    break

# ---------------------------------------------------------------- summary over everything saved so far
recs = [d for d in done.values() if d["head"] == (HEAD or "stock") and d["C"] == C and d["cap"] == CAP]
keys = sorted({d["key"] for d in recs})
say(f"D-20 summary (C {C}, {CAP} tokens, head {HEAD or 'stock'}; wall-clock tok/s per prompt, unpaused, cut runs dropped):")
for k_ in keys:
    rs = {d["arm"]: d for d in recs if d["key"] == k_}
    base = rs.get("dfl")
    eq = all(base is not None and d["out"] == base["out"] for d in rs.values())
    say(f"  {k_} [{next(iter(rs.values()))['type']}]: " + ", ".join(f"{a} {d['tok_s']:.1f}" + ("p" if d["paused"] else "")
                                                               for a, d in sorted(rs.items())) + f"; tokens equal across arms {eq}")
for arm in ARMS:
    if arm == "dfl":
        continue
    pairs = [(d, next((e for e in recs if e["key"] == d["key"] and e["arm"] == "dfl"), None)) for d in recs if d["arm"] == arm]
    pairs = [(d, b) for d, b in pairs if b is not None and not d["paused"] and not b["paused"] and not d["cut"] and not b["cut"]
             and d["type"] in ("chat", "agent")]
    if pairs:
        g = np.mean([d["tok_s"] for d, _ in pairs]) / np.mean([b["tok_s"] for _, b in pairs]) - 1
        eqs = all(d["out"] == b["out"] for d, b in pairs)
        say(f"  {arm} vs dfl on chat / agent ({len(pairs)} prompts): mean tok/s {np.mean([d['tok_s'] for d, _ in pairs]):.2f} vs "
            f"{np.mean([b['tok_s'] for _, b in pairs]):.2f} = {100 * g:+.1f} %; tokens equal {eqs}; "
            f"{'PASS' if eqs and g >= 0.10 else 'KILL' if g < 0.04 else 'BETWEEN' if eqs else 'FAIL (tokens)'} "
            f"(PASS >= +10 %, KILL < +4 %)")
say(f"  footprint now {footprint_now_gb():.1f} GB, lifetime peak {footprint_peak_gb():.1f} GB, MLX peak {mx.get_peak_memory() / 1e9:.1f} GB; "
    f"{el():.0f} s running")
open(os.path.join(TD, "d20_switch45.txt"), "a").write("\n".join(lines) + "\n")
