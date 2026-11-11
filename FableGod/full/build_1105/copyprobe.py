"""copyprobe.py (Fable FULL, LOSSY350 T-A1; ~/FableGod/full/build_1105/INSTALL.md): rolled-back timing passes over copied
rows inside BigMoE's r57 screen. It measures what a WIDE STRICT COPY PASS costs: W rows whose tokens are the prompt's own
continuation (the signature + docstring a HumanEval answer re-emits), no drafter call.

How it runs: apply_screen.py adds two things to a COPY of scripts/r57_screen.py: `CPROBE = <this module>; CPROBE.install(
globals())` before set_arm, and `CPROBE.maybe(globals(), locals())` in run()'s loop right before the real check. With
R57_COPYPROBE unset the copy is the screen. The probe never commits: every pass is rolled back (the warm-split's pattern,
r57_screen.py's `if wsp:` block), so the run's tokens are the arm's own. Its tok/s is NOT comparable (the probe warms the
pools); the screen's clocks are shifted by the probe's time.

A probe point = a check whose last K (8) committed tokens occur in the prompt with >= 31 prompt tokens after them before
the prompt's closing code fence. There the module times, each twice in a row (pass 1 as the pools are, pass 2 warm):
  g4 / g8 / g16   the sync-free gate path as the arm runs it (<= 16 rows): [tok + copied tokens], every copied row a sure
                  row at cf 1.0 (QOM's rule; strict with R57_QOM_SPLIT), the last row exact, as the engine's own check.
                  Their warm ms against rows give the fast path's cost per strict copied row and its intercept.
  x32 / x64       a wide EXACT pass: rows > 16 leave the gate for EvalSuite's block (the prefill path), as R57_ROWS=32 did.
  s32 / s64       a wide STRICT pass on that same path: ranks 0-1 + the block's members kept, the other ranks routed to the
                  row's rank-0 expert at weight 0 (no new read), the last row exact. This module's block wrapper does it.
  t32 / t64       EXPERIMENTAL (R57_COPYPROBE=...+t): top-2 only as a narrow [W, 2] call, which fits the sync-free gate
                  (W x 2 ids <= BIGMOE_MAXK) and the tier call. Never run at HEAD; it goes last and is dropped on an error.
A different pass goes first at each probe point (the cold one), in turn over g16 / s32 / s64. Probe points of a run do not
overlap (the next one starts past the rows the last one touched), so each first pass meets its rows as the run left them.

Env: R57_COPYPROBE=g+x+s (variants, '+'-joined; '1' = g+x+s), R57_COPYPROBE_MAX=3 (probe points per run),
R57_COPYPROBE_BUDGET=240 (s of probing per process; no new probe point after it: the screen itself stops at 600 s),
R57_COPYPROBE_OUT=<jsonl> (default data/r57/{TAG}_copyprobe.jsonl). Records: one JSON line per pass (see _pass).
CPU dry-run: dryrun.py beside this file (a fake engine; no mlx)."""
import json, os, sys, time

K = 8                                                    # the k-gram that triggers a probe point (MATCH's shortest)
GATE_ROWS = (4, 8, 16)
WIDTHS = (32, 64)
CFG = dict(variants=("g", "x", "s"), maxp=3, off=set(), turn=0, budget=240.0, spent=0.0)
CP = dict(mode=None, sure=None, calls=0)                 # the wide block wrapper's switch: None = the screen's own block
STATE = {}                                               # per run name: the prompt's k-grams, the fence, probe count
_OUT = [None]


def configure(env=os.environ):
    v = env.get("R57_COPYPROBE", "")
    CFG["variants"] = ("g", "x", "s") if v in ("", "1") else tuple(x for x in v.replace(",", "+").split("+") if x)
    assert set(CFG["variants"]) <= {"g", "x", "s", "t"}, CFG["variants"]
    CFG["maxp"] = int(env.get("R57_COPYPROBE_MAX", "3"))
    CFG["budget"] = float(env.get("R57_COPYPROBE_BUDGET", "240"))   # s of probing per process (the screen stops at 600 s)
    CFG["off"] = set(); CFG["turn"] = 0; CFG["spent"] = 0.0
    return CFG


def strict_mask(mx, inds, sure):
    """the split form's rule for a block: a sure row keeps ranks 0-1 and every rank whose expert is in the block's loads
    (ranks 0-1 of the sure rows, every rank of the others); an unsure row keeps all. inds [..., rows, K] (rank-sorted),
    sure [rows] bool -> keep [..., rows, K] bool. The same expression as r57_screen.py's _fast_call (U3's union)."""
    K_ = inds.shape[-1]
    ids2 = inds.reshape(-1, K_)
    top2 = mx.arange(K_)[None, :] < 2
    cand = (~sure)[:, None] | top2
    in_u = ((ids2[:, :, None, None] == ids2[None, None, :, :]) & cand[None, None, :, :]).any(axis=(2, 3))
    return (cand | in_u).reshape(inds.shape)


def install(G):
    """wrap the MoE block: calls with more than 16 rows while CP['mode'] is 's' or 't' take this module's block; every
    other call (CP['mode'] None, <= 16 rows, an unsorted or unstreamed block) goes to the screen's block unchanged."""
    configure()
    mx, np = G["mx"], G["np"]
    Q = G.get("_QX")
    if Q is None:
        from mtplx.models import qwen4_exp as Q
    EM = G["EM"]
    prev = Q.SparseMoeBlock.__call__

    def _cp_block(self, x):
        mode = CP["mode"]
        bm = getattr(self.switch_mlp, "_bm", None)
        if (mode is None or bm is None or x.ndim < 2 or int(np.prod(x.shape[:-1])) <= 16 or not EM.STATE.sorted
                or CP["sure"] is None or int(np.prod(x.shape[:-1])) != CP["sure"].shape[0]):
            return prev(self, x)
        CP["calls"] += 1
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)      # EvalSuite's block (evalsuite/model.py block), verbatim
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        order = mx.argsort(-scores, axis=-1)
        inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        if mode == "s":                                              # strict rows, route mode: a dropped rank reads the
            km = strict_mask(mx, inds, CP["sure"])                   # row's rank-0 expert (no new read) at weight 0
            scores = mx.where(km, scores, mx.zeros_like(scores))
            inds = mx.where(km, inds, inds[..., :1])
            y = self.switch_mlp(x, inds)
        else:                                                        # 't': ranks 0-1 only, a narrow call ([rows, 2] ids
            inds, scores = inds[..., :2], scores[..., :2]            # fit the sync-free gate); the tier call combines
            t, bt = bm.get("tier"), sys.modules.get("bigmoe.tier")
            fused = t is not None and bt is not None and bt._COMBINE and bt._GLU == "3c"
            if fused:
                t["scores"], t["shared"] = scores, shared
            try:
                y = self.switch_mlp(x, inds)
            finally:
                if fused:
                    t.pop("scores", None); t.pop("shared", None)
        if y.ndim > x.ndim:                                          # [..., K, D]: not combined in the switch
            y = (y * scores[..., None]).sum(axis=-2)
        return y + shared

    Q.SparseMoeBlock.__call__ = _cp_block
    CP["installed"] = True
    return _cp_block


def _emit(G, rec):
    if _OUT[0] is None:
        _OUT[0] = os.environ.get("R57_COPYPROBE_OUT") or os.path.join(G.get("OUT", "."), f"{G.get('TAG', 'r57')}_copyprobe.jsonl")
    with open(_OUT[0], "a") as f:
        f.write(json.dumps(rec) + "\n")


def new_state(ids, decode, im_end):
    """the prompt's K-grams (-> the position of their last token) and the end of its code block: the last token holding
    a ``` fence before the prompt's last <|im_end|> (the whole prompt if there is none)."""
    last = len(ids) - 1 - ids[::-1].index(im_end) if im_end in ids else len(ids)
    end = next((i for i in range(last - 1, -1, -1) if "```" in decode(ids[i])), last)
    grams = {}
    for i in range(K - 1, end):
        grams[tuple(ids[i - K + 1:i + 1])] = i               # a later occurrence wins (the code block is the prompt's end)
    return dict(P=len(ids), end=end, grams=grams, n=0, seen=0, dead=False, hi=0)


def find_copy(S, out):
    """the prompt position whose K-gram equals the last K committed tokens, or None"""
    if len(out) < K:
        return None
    j = S["grams"].get(tuple(out[-K:]))
    return j if j is not None and j + 1 < S["end"] else None


def plan(n_cont, variants, off, turn):
    """the passes of one probe point, in order: [(variant, rows)]; the pass that goes first rotates with `turn`."""
    widths = [w for w in WIDTHS if n_cont >= w - 1]
    safe = []
    if "g" in variants:
        safe += [("g", r) for r in GATE_ROWS]
    for v in ("x", "s"):
        if v in variants and v not in off:
            safe += [(v, w) for w in widths]
    first = [p for p in (("g", 16), ("s", 32), ("s", 64)) if p in safe]      # the three the verdict reads cold
    first = first if len(first) > 1 else [p for p in (("g", 16), ("x", 32), ("x", 64)) if p in safe]
    if first:
        f = first[turn % len(first)]
        safe = [f] + [p for p in safe if p != f]
    risky = [("t", w) for w in widths] if "t" in variants and "t" not in off else []
    return safe + risky


def _offsets(cache):
    return [getattr(c, "offset", None) if c is not None else None for c in cache]


def _pass(G, L, toks, rows, variant, nxt):
    """one rolled-back pass over toks[:rows]; nxt = the copied token after the last row (or None).
    Record: v, rows, ms (graph build + eval wait), build ms, miss, io ms (demand SSD busy), svc ms (gate service), uses (the
    ext's distinct experts computed, summed over layer calls), stl (stalled layer calls), rb ms (the rollback), acc (leading
    copied rows the pass confirms: its argmax == the copied token; the last row counts when nxt is known), agree (rows
    confirmed anywhere), n (rows with a copied token to confirm), strict (the FAST gate or this module's block was on),
    wc (wide block calls this module built during the pass: 0 on an s / t pass = it ran the screen's exact block), err."""
    mx, np, CS, ref2, FG = G["mx"], G["np"], G["CS"], G["ref2"], G["FG"]
    model, cache, stats, states = G["model"], L["cache"], G["stats"], G["states"]
    fz = L.get("fz")
    inp = mx.array([list(toks[:rows])])
    off0 = _offsets(cache); wc0 = CP["calls"]
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    s0 = stats(); t0 = time.perf_counter()
    err = None; am = None; built = False; strict = False
    tb = te = t0
    try:
        sure = np.array([True] * (rows - 1) + [False])
        if variant == "g" and fz is not None and fz.get("q1u"):
            qcf = np.ones(rows, np.float32) if G.get("QSPLIT") is not None else None
            FG.update(on=True, keep=None, sure=mx.array(sure), sure_np=sure, t3=bool(fz.get("t3")),
                      qom=bool(fz.get("qom")), qcf=mx.array(qcf) if qcf is not None else None)
            strict = True
        elif variant in ("s", "t"):
            CP["mode"], CP["sure"] = variant, mx.array(sure)
            strict = True
        try:
            with ref2.verify_scope(model):
                lg = model(inp, cache=cache)
        finally:
            FG.update(on=False, keep=None, sure=None, sure_np=None, t3=False, cold=False, qom=False, qcf=None)
            CP["mode"], CP["sure"] = None, None
        built = True
        tb = time.perf_counter()
        amx = mx.argmax(lg[0], axis=-1)
        mx.eval(amx)
        te = time.perf_counter()
        am = amx.tolist()
    except Exception as e:                                           # never cost the run: record, roll back, go on
        err = repr(e)[:300]
    d = stats() - s0
    try:
        if built:
            CS.rollback_after_verify(cache, snap, rows)
        else:                                                        # the graph build stopped part way: trim each entry by
            for c, o in zip(cache, off0):                            # what it grew, then restore the recurrent states
                if c is not None and o is not None and hasattr(c, "trim") and getattr(c, "offset", o) > o:
                    c.trim(c.offset - o)
            CS.restore_cache(cache, snap)
        mx.eval(states(cache))
    except Exception as e:
        err = (err + " | " if err else "") + "rollback: " + repr(e)[:200]
    tr = time.perf_counter()
    rec = dict(v=variant, rows=rows, ms=round(1e3 * (te - t0), 3), build=round(1e3 * (tb - t0), 3), miss=int(d[0]),
               io=round(1e3 * float(d[1]), 3), svc=round(1e3 * float(d[2]), 3), uses=int(d[4]) if len(d) > 4 else None,
               stl=int(d[5]) if len(d) > 5 else None, rb=round(1e3 * (tr - te), 3), strict=strict, err=err,
               wc=CP["calls"] - wc0)                           # wide block calls this module built (s / t: one per MoE layer)
    if am is not None:
        want = list(toks[1:rows]) + ([nxt] if nxt is not None else [])
        hit = [int(a_ == w_) for a_, w_ in zip(am, want)]
        rec.update(acc=next((i for i, h in enumerate(hit) if not h), len(hit)), agree=sum(hit), n=len(want))
    return rec


def _maybe(G, L):
    if L.get("mtp") or not CP.get("installed") or CFG["spent"] > CFG["budget"] or not L.get("name"):
        return                                                       # (an unnamed run is the screen's discarded warm-up)
    name = L["name"]
    ids, out = L["ids"], L["out"]
    S = STATE.get(name)
    if S is None or S["P"] != len(ids) or S["seen"] > len(out):     # a new run
        tkz = G["_TKZ"]
        S = STATE[name] = new_state(ids, lambda t: tkz.decode([int(t)]), tkz.token_to_id("<|im_end|>"))
    S["seen"] = len(out)
    if S["dead"] or S["n"] >= CFG["maxp"] or len(ids) + len(out) - 1 < S["hi"]:   # hi: past the rows the last point
        return                                                       # touched, so a first pass meets them as the run left them
    j = find_copy(S, out)
    if j is None:
        return
    cont = ids[j + 1:S["end"]]
    if len(cont) < min(WIDTHS) - 1:
        return
    order = plan(len(cont), CFG["variants"], CFG["off"], CFG["turn"])
    if not order:
        return
    CFG["turn"] += 1
    S["hi"] = len(ids) + len(out) - 1 + max(r for _, r in order)
    toks = [L["tok"]] + list(cont)
    head = dict(kind="point", run=name, p=S["n"], a=len(ids) + len(out) - 1, src=j, left=len(cont),
                order=[f"{v}{r}" for v, r in order], fast=G.get("FAST"), split=G.get("QSPLIT"), d17=bool(G.get("D17")))
    _emit(G, head)
    for i, (v, rows) in enumerate(order):
        if v in CFG["off"] or S["dead"]:
            continue
        for rep in (0, 1):
            rec = _pass(G, L, toks, rows, v, cont[rows - 1] if len(cont) >= rows else None)
            rec.update(kind="pass", run=name, p=S["n"], a=head["a"], first=bool(i == 0 and rep == 0), rep=rep)
            _emit(G, rec)
            if rec["err"]:
                if v in ("s", "t"):
                    CFG["off"].add(v)                                # this variant does not run in this engine: drop it
                if "rollback" in rec["err"] or v in ("g", "x"):
                    S["dead"] = True                                 # the cache may be off: no more probes in this run
                break
    S["n"] += 1


def maybe(G, L):
    """the screen's hook: probe if this check is a probe point; returns the seconds spent (the screen shifts its clocks)"""
    t0 = time.perf_counter()
    try:
        _maybe(G, L)
    except Exception as e:
        try:
            _emit(G, dict(kind="error", run=L.get("name"), err=repr(e)[:300]))
        except Exception:
            pass
    finally:
        CP["mode"], CP["sure"] = None, None
    dt = time.perf_counter() - t0
    CFG["spent"] += dt
    return dt
