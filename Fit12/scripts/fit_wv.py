"""Fit12 WV1 (2026-10-24): which op makes B12LL's verify width-variant (WT1: tokens leave U1 when widths vary 2-4)?
From one prefilled state (spec125.restore_state deep-copies it), the trunk runs spec125's verify forward on the same tokens at
widths 4, 3 and 2 (snapshot_untrimmable_cache_lazy + ensure_eager_window_capacity + verify_capture_scope, as generate does).
Class-level wrappers (active only during the probe; trunk layers, S > 1) record the input and output of every PLE, GDN,
attention, MoE block (B12's block) and decoder layer (B12's layer_call). The hyper-connections are compiled, so they are
judged from their neighbours: a module whose INPUT rows differ while the previous module's output rows matched points at the
HC between them. Rows 0..m-1 of width 4 are compared bitwise with width m. F12_WV=code,prose,... runs it instead of arms."""
import os

PROMPTS = [p for p in os.environ.get("F12_WV", "").split(",") if p]
ACTIVE = [False]
LOG = []
LID = {}


def apply(Q, model):
    if not PROMPTS:
        return False
    for i, l in enumerate(model.language_model.model.layers):
        LID[id(l)] = i
        for n in ("linear_attn", "self_attn", "ple", "mlp"):
            if n in l:
                LID[id(l[n])] = i
    for cls, name in ((Q.GatedDeltaNet, "gdn"), (Q.Attention, "attn"), (Q.PLELayer, "ple"), (Q.SparseMoeBlock, "moe"),
                      (Q.DecoderLayer, "layer")):
        orig = cls.__call__

        def call(self, *a, _orig=orig, _name=name, **kw):
            import mlx.core as mx
            out = _orig(self, *a, **kw)
            x = a[0]
            if ACTIVE[0] and id(self) in LID and x.ndim >= 3 and x.shape[1] > 1:
                mx.eval(x, out)
                LOG.append((_name, LID[id(self)], x, out))
            return out
        cls.__call__ = call
    return True


def probe(P, ids, model, state, B, spec125, reset_generation_state, mx):
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    out, _ = spec125.generate(model, ids, 4, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
    toks = [int(t) for t in out[:4]]
    rec = {}
    for S in (4, 3, 2):
        reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
        cache, _ = spec125.restore_state(model, state, with_head=True)
        spec125.snapshot_untrimmable_cache_lazy(cache)
        spec125.ensure_eager_window_capacity(cache, S)
        LOG.clear(); ACTIVE[0] = True
        with model.verify_capture_scope():
            lg, hid = spec125._split(model(mx.array([toks[:S]]), cache=cache, return_hidden=True))
        mx.eval(lg, hid)
        ACTIVE[0] = False
        rec[S] = (list(LOG), lg)
    for m in (3, 2):
        L4, lg4 = rec[4]
        Lm, lgm = rec[m]
        if len(L4) != len(Lm):
            print(f"[wv] {P} width 4 vs {m}: call sequences differ ({len(L4)} vs {len(Lm)})", flush=True)
            continue
        first, nbad = None, 0
        for (n4, l4, x4, o4), (nm, lm, xm, om) in zip(L4, Lm):
            xi = bool(mx.array_equal(x4[:, :m], xm[:, :m]).item())
            oi = bool(mx.array_equal(o4[:, :m], om[:, :m]).item())
            if not (xi and oi):
                nbad += 1
                if first is None:
                    d = mx.abs(o4[:, :m].astype(mx.float32) - om[:, :m].astype(mx.float32)).max().item()
                    first = f"{n4} L{l4}: input {'equal' if xi else 'DIFFERS'}, output {'equal' if oi else 'DIFFERS'} (max |d out| {d:.3g})"
        lq = bool(mx.array_equal(lg4[:, :m], lgm[:, :m]).item())
        a4 = mx.argmax(lg4[0, :m].astype(mx.float32), axis=-1).tolist(); am = mx.argmax(lgm[0, :m].astype(mx.float32), axis=-1).tolist()
        print(f"[wv] {P} width 4 vs {m} (rows 0-{m - 1}, tokens {toks[:m]}): {len(L4)} recorded calls, {nbad} differ; first: "
              f"{first or 'none'}; logits rows equal {lq}, argmax {a4} vs {am}", flush=True)


PROMPTS2 = [p for p in os.environ.get("F12_WV2", "").split(",") if p]


def _vpass(model, spec125, mx, cache, toks):
    """one verify pass of len(toks) rows, every row kept (as generate does on full acceptance); returns logits, commit ok"""
    S = len(toks)
    snap = spec125.snapshot_untrimmable_cache_lazy(cache)
    spec125.ensure_eager_window_capacity(cache, S)
    with model.verify_capture_scope():
        lg, hid = spec125._split(model(mx.array([toks]), cache=cache, return_hidden=True))
    mx.eval(lg, hid)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=S, verified_tokens=S)
    if not ok:
        spec125.rollback_after_verify(cache, snap, verified_tokens=S)
        lg, hid = spec125._split(model(mx.array([toks]), cache=cache, return_hidden=True))
        mx.eval(lg, hid)
    model.clear_verify_capture(cache)
    return lg, ok


def _arrays(c):
    st = getattr(c, "state", None)
    if st is None:
        return []
    if not isinstance(st, (list, tuple)):
        st = [st]
    return [a for a in st if a is not None and hasattr(a, "shape")]


def probe2(P, ids, model, state, B, spec125, reset_generation_state, mx):
    """WV2: commit the same 4 tokens as one 4-row pass or as two 2-row passes (all rows kept), then compare every cache
    array bitwise and the logits of an identical next 4-row pass."""
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    out, _ = spec125.generate(model, ids, 8, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
    t = [int(x) for x in out[:8]]
    res = {}
    for name, chunks in (("4", [t[0:4]]), ("2+2", [t[0:2], t[2:4]])):
        reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
        cache, _ = spec125.restore_state(model, state, with_head=True)
        oks = [_vpass(model, spec125, mx, cache, c)[1] for c in chunks]
        arrs = [_arrays(c) for c in cache]
        mx.eval([a for l in arrs for a in l])
        lg, ok2 = _vpass(model, spec125, mx, cache, t[4:8])
        res[name] = (arrs, lg, oks + [ok2])
    (a4, lg4, ok4), (a22, lg22, ok22) = res["4"], res["2+2"]
    bad, first = 0, None
    for L, (x, y) in enumerate(zip(a4, a22)):
        for j, (u, v) in enumerate(zip(x, y)):
            same = u.shape == v.shape and bool(mx.array_equal(u, v).item())
            if not same:
                bad += 1
                if first is None:
                    d = (mx.abs(u.astype(mx.float32) - v.astype(mx.float32)).max().item() if u.shape == v.shape else float("nan"))
                    first = f"cache[{L}] array {j} {tuple(u.shape)} {u.dtype}: max |d| {d:.3g}"
    lq = bool(mx.array_equal(lg4, lg22).item())
    am4 = mx.argmax(lg4[0].astype(mx.float32), axis=-1).tolist(); am22 = mx.argmax(lg22[0].astype(mx.float32), axis=-1).tolist()
    kinds = sorted({type(c).__name__ for c in [*[None]]} if False else {type(c).__name__ for c in spec125.restore_state(model, state, with_head=False)[0]})
    print(f"[wv2] {P} tokens {t}: commits ok 4 {ok4}, 2+2 {ok22}; cache arrays differing {bad} (cache kinds {kinds}); first: "
          f"{first or 'none'}; next 4-row pass logits equal {lq}, argmax {am4} vs {am22}", flush=True)


PROMPTS3 = [p for p in os.environ.get("F12_WV3", "").split(",") if p]


def _vpass_keep(model, spec125, mx, cache, toks, keep):
    """a verify pass of len(toks) rows keeping the first `keep` (generate's commit; on refusal, rollback + re-forward)"""
    S = len(toks)
    snap = spec125.snapshot_untrimmable_cache_lazy(cache)
    spec125.ensure_eager_window_capacity(cache, S)
    with model.verify_capture_scope():
        lg, hid = spec125._split(model(mx.array([toks]), cache=cache, return_hidden=True))
    mx.eval(lg, hid)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=keep, verified_tokens=S)
    if not ok:
        spec125.rollback_after_verify(cache, snap, verified_tokens=S)
        lg2, hid = spec125._split(model(mx.array([toks[:keep]]), cache=cache, return_hidden=True))
        mx.eval(hid)
    model.clear_verify_capture(cache)
    return ok


def probe3(P, ids, model, state, B, spec125, reset_generation_state, mx):
    """WV3: positions p0..p0+4 committed with a partial-accept replay (keep 1 or 2 of a 4-row pass) vs by full accepts only."""
    reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
    out, _ = spec125.generate(model, ids, 8, k_mtp=3, max_draft=3, lookup=0, prefill_chunk=512, prefilled=state, stats={})
    t = [int(x) for x in out[:8]]
    plans = {"ref (2 full + 3 full)": [(t[0:2], 2), (t[2:5], 3)],
             "keep1 (4 keep 1, then 4 full)": [(t[0:4], 1), (t[1:5], 4)],
             "keep2 (4 keep 2, then 3 full)": [(t[0:4], 2), (t[2:5], 3)]}
    res = {}
    for name, steps in plans.items():
        reset_generation_state(model); B.STASH.clear(); B.PENDH.clear()
        cache, _ = spec125.restore_state(model, state, with_head=True)
        oks = [_vpass_keep(model, spec125, mx, cache, s, k) for s, k in steps]
        arrs = [_arrays(c) for c in cache]
        mx.eval([a for l in arrs for a in l])
        res[name] = (arrs, oks)
    ref = res["ref (2 full + 3 full)"][0]
    for name in list(plans)[1:]:
        arrs, oks = res[name]
        bad, where = 0, []
        for L, (x, y) in enumerate(zip(ref, arrs)):
            for j, (u, v) in enumerate(zip(x, y)):
                if not (u.shape == v.shape and bool(mx.array_equal(u, v).item())):
                    bad += 1
                    if len(where) < 4:
                        d = mx.abs(u.astype(mx.float32) - v.astype(mx.float32)).max().item() if u.shape == v.shape else float("nan")
                        where.append(f"cache[{L}][{j}] {u.dtype} max|d| {d:.3g}")
        print(f"[wv3] {P} {name}: commits ok {oks}; cache arrays differing from ref: {bad}; first: {where or 'none'}", flush=True)


CAP = os.environ.get("F12_CAP", "")
CAP_LAYERS = tuple(int(x) for x in os.environ.get("F12_CAP_LAYERS", "2,12,24,36,46").split(","))
CAP_TEXTS = [t for t in os.environ.get("F12_CAP_TEXTS", "").split(",") if t]   # B10a: "A,B" -> F12_CAP/A, F12_CAP/B


def capture(Q, model, B, tok, mx):
    """Q2: record the MoE input x (rows) of CAP_LAYERS during one B12 prefill of ~4K tokens of local text (PLAN.md +
    fit_b12.py), saved as float32 [n, 2560] to F12_CAP/x_L<L>.npy (Fit12's own files; read-only otherwise)."""
    import numpy as np
    # the router's input IS the MoE input (bt_b12's decoupled prefill calls blk.gate(x) directly, not the block's __call__)
    gates = {i: l["mlp"]["gate"] for i, l in enumerate(model.language_model.model.layers) if i in CAP_LAYERS}
    lid = {id(g): i for i, g in gates.items()}
    got = {i: [] for i in CAP_LAYERS}
    for cls in {type(g) for g in gates.values()}:
        orig = cls.__call__

        def call(self, x, *a, _orig=orig, **kw):
            i = lid.get(id(self))
            if i is not None:
                r = x.reshape(-1, x.shape[-1])
                if r.shape[0] >= 64:
                    got[i].append(np.array(r.astype(mx.float32)))
            out = _orig(self, x, *a, **kw)
            if i is not None and CAP_TEXTS and r.shape[0] >= 64:      # B10a: each row's top-10 experts (router logits)
                lg = np.array(out.reshape(-1, out.shape[-1]).astype(mx.float32))
                gid[i].append(np.argpartition(-lg, 10, axis=-1)[:, :10].astype(np.int16))
            return out
        cls.__call__ = call
    gid = {i: [] for i in CAP_LAYERS}
    if not CAP_TEXTS:
        text = open(os.path.expanduser("~/Fit12/PLAN.md")).read()[:12000] + "\n\n" + open(os.path.expanduser("~/Fit12/scripts/fit_b12.py")).read()[:8000]
        runs = [(CAP, tok.encode(text))]
    else:
        runs = [(os.path.join(CAP, t), b10_text(t, tok)) for t in CAP_TEXTS]
    for d, ids in runs:
        for i in CAP_LAYERS:
            got[i].clear(); gid[i].clear()
        B.b12_prefill_state(model, ids, 512, 4096)
        os.makedirs(d, exist_ok=True)
        for i in CAP_LAYERS:
            X = np.concatenate(got[i]) if got[i] else np.zeros((0, 2560), np.float32)
            np.save(os.path.join(d, f"x_L{i}.npy"), X.astype(np.float16) if CAP_TEXTS and os.path.basename(d)[0] in "AR" else X)
            if gid[i]:
                np.save(os.path.join(d, f"ids_L{i}.npy"), np.concatenate(gid[i]))
            print(f"[cap] {d} layer {i}: {X.shape[0]} rows saved ({len(ids)} prompt tokens)", flush=True)


def b10_text(t, tok):
    """B10a capture texts (~4K tokens each). A0..A7 = FIT texts, none from the Eval suite: notes (MoonShots / NewIdeas /
    Breakthrough / Leap markdown), Swift from ~/OjoX, Python from ~/Leap + ~/MoonShots (all read-only). B = JUDGE text: the 4-bit reference model's own prompts + completions
    (~/EvalSuite/cache/ref/tf_gen.jsonl token ids, read-only), round-robin over its sets."""
    import glob, json
    if t[0] in "RU":                            # B10-r: fit_b10r_text.py's token ids (R0..R9 calibration, U fragility judge)
        return json.load(open(os.path.expanduser(f"~/Fit12/data/b10r/texts/{t}.json")))["ids"]
    if t.startswith("A"):                       # A0..A7: eight different ~4K-token documents (calibration, ~32K tokens)
        H = os.path.expanduser
        sw = sorted(glob.glob(H("~/OjoX/**/*.swift"), recursive=True), key=os.path.getsize)[::-1]
        py = sorted(glob.glob(H("~/Leap/scripts/*.py")) + glob.glob(H("~/MoonShots/scripts/*.py")), key=os.path.getsize)[::-1]
        src = {"A0": [H("~/MoonShots/FINDINGS.md")], "A1": [H("~/NewIdeas/IDEAS.md")], "A2": [H("~/Breakthrough/SCOREBOARD.md")],
               "A3": sw[0:2], "A4": sw[2:5], "A5": py[0:3], "A6": py[3:8], "A7": [H("~/Leap/HANDOVER.md"), H("~/MoonShots/IDEAS.md")]}[t]
        return tok.encode("\n\n".join(open(f, errors="replace").read()[:20000] for f in src))[:4096]
    rows = [json.loads(l) for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl"))]
    by = {}
    for r in rows:
        by.setdefault(r["set"], []).append(r["prompt_ids"] + r["completion_ids"])
    ids, k = [], 0
    while len(ids) < 4096 and any(by.values()):
        for st in sorted(by):
            if by[st]:
                ids += by[st].pop(0)
        k += 1
    return ids[:4096]
