"""DIAG95 (BigMoE, Wed 18 Nov 2026; Leap's ask via the orchestrator, which decides whether exact gets trees this week).

M4C (11:3x) left one key: mbpp/30's token 95, where the TREE path and the CHAIN path pick different tokens with Leap's
contract ON (REF2_REFC=1). This finds the FIRST layer / sublayer whose output differs bitwise between them.

Runs inside the r57 screen (R57_DIAG95=1): the screen hands over its globals after the model load, ref2, refc and xknob are
installed; tree16 is installed here. Context: mbpp/30's prompt + the contract chain arm's answer tokens 0-89 prefilled
(data/r57/bt98c/MC_C_tokens.json); the decision row is answer token 94 (it picks token 95).
- T1, the check itself: from the same prefill, tree16.tree_verify of a 16-node tree whose main chain is answer 90-94 + the
  chain's token 95 (10 random siblings off it) vs a chain forward of the same 6 tokens. Every main-chain row is compared per
  layer and sublayer (0 PLE, 1 attn HC read, 2 GDN or attention, 3 mlp HC read, 4 MoE, 6 layer out) and at the logits.
- T2, the commit: the tree route (prefill, snapshot, tree_verify, tree16_commit.tree_commit of nodes 0-4) vs the chain route
  (prefill, snapshot, chain verify, commit_verified_window keep 5), then the same 4-row probe (answer 95-98); the probe rows
  are compared the same way.
Writes data/r57/diag95/diag95.json and prints one verdict line per test."""
import contextlib
import json
import os
import sys
import time

import numpy as np


def run(G):
    mx, model, ref2 = G["mx"], G["model"], G["ref2"]
    t0 = time.time()

    def say(s):
        print(f"[diag95] {s}", flush=True)

    Q = sys.modules["mtplx.models.qwen4_exp"]
    T16L = G["T16L"]
    TV, TC, tree_meta = T16L.TV, T16L.TC, T16L.tree_meta
    from mtplx import cache_state as CS
    from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
    T16L.install(ref2)
    G["ref_on"](True)
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    say(f"contract: {G['refc'].report() if G.get('REFC') else 'OFF'}")

    @contextlib.contextmanager
    def verify_scope():                                  # ref2's own scope when it has one (tree16_loop.verify's contract)
        if hasattr(ref2, "verify_scope"):
            with ref2.verify_scope(model):
                yield
        else:
            with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
                    model.verify_capture_scope():
                yield

    def states(cache):
        return [c.state for c in cache if hasattr(c, "state")]

    def prefill(ids):
        cache = model.make_cache()
        lg = model(mx.array([ids]), cache=cache)
        mx.eval(lg, states(cache))
        return cache

    def fwd(cache, tokens):
        with verify_scope():
            lg, hw = model(mx.array([tokens]), cache=cache, return_hidden=True)
        lg = lg[0].astype(mx.float32)
        mx.eval(lg, hw)
        return lg, hw[0]

    IDX = {}
    for i, L in enumerate(inner.layers):                 # module instance -> (layer, sublayer)
        IDX[id(L)] = (i, "6_layer")
        IDX[id(L.attn_hyper_connection)] = (i, "1_attn_hc")
        IDX[id(L.mlp_hyper_connection)] = (i, "3_mlp_hc")
        IDX[id(L.mlp)] = (i, "4_moe")
        if getattr(L, "is_linear", False):
            IDX[id(L.linear_attn)] = (i, "2_gdn")
        else:
            IDX[id(L.self_attn)] = (i, "2_attn")
        if getattr(L, "ple", None) is not None:
            IDX[id(L.ple)] = (i, "0_ple")
    REC = {"on": False, "d": {}}

    def wrap(cls):                                       # outermost: after tree16's own patches
        f0 = cls.__call__

        def f(self, *a, **kw):
            out = f0(self, *a, **kw)
            if REC["on"] and id(self) in IDX:
                o = out[0] if isinstance(out, tuple) else out
                mx.eval(o)
                REC["d"].setdefault(IDX[id(self)], []).append(np.array(o[0].astype(mx.float32)))
            return out
        cls.__call__ = f
    for cls in (Q.DecoderLayer, Q.GatedResidual, Q.Attention, Q.GatedDeltaNet, Q.SparseMoeBlock, Q.PLELayer):
        wrap(cls)

    def record(fn):
        REC["on"], REC["d"] = True, {}
        try:
            out = fn()
        finally:
            REC["on"] = False
        return out, {k: v[0] for k, v in REC["d"].items()}

    def first_diff(ra, rows_a, rb, rows_b):
        """the first (layer, sublayer, row) in model order whose output differs bitwise; rows_a / rows_b: matching rows"""
        for key in sorted(set(ra) & set(rb)):
            xa, xb = ra[key][rows_a], rb[key][rows_b]
            if xa.shape != xb.shape or xa.tobytes() != xb.tobytes():
                bad = [j for j in range(len(rows_a)) if xa[j].tobytes() != xb[j].tobytes()]
                return dict(layer=key[0], sublayer=key[1][2:], rows=bad, max_abs=float(np.max(np.abs(xa - xb))),
                            compared=len(set(ra) & set(rb)))
        return None

    it = next(x for x in G["ED"].load("dec_mbpp") if x["key"] == "mbpp/30")
    P = [int(t) for t in it["prompt_ids"]]
    A = [int(t) for t in [x for arm in json.load(open(os.path.expanduser(
        "~/BigMoE/data/r57/bt98c/MC_C_tokens.json")))["mbpp/30"].values() for x in arm][0]]
    a = 90
    ids = P + A[:a]
    main = A[a:a + 5] + [A[95]]                          # root = answer 90 ... answer 94, then the chain's token 95
    rng = np.random.default_rng(95)
    par = [-1, 0, 1, 2, 3, 4]                            # main chain = nodes 0-5 (DFS preorder, children best-first)
    par += [4, 6, 3, 2, 1, 10, 0, 12, 13, 0]             # then the siblings in DFS order (backtracking up the chain)
    toks = list(main)
    for n in range(6, 16):                               # random sibling tokens, distinct from their siblings
        used = {toks[c] for c in range(n) if par[c] == par[n]}
        t = int(rng.integers(1000, 50000))
        while t in used:
            t = int(rng.integers(1000, 50000))
        toks.append(t)
    T = len(toks)
    meta = tree_meta(toks, par, len(ids), ids[-T16L.PREV:])
    R = dict(prompt=len(P), prefill=len(ids), main=main, decision_row=4, chain_token_95=A[95], tree_T=T)
    say(f"mbpp/30: prompt {len(P)}, prefill {len(ids)} tokens; main chain {main}; tree T {T}")
    # T1
    ct, cc = prefill(ids), prefill(ids)                  # prefills outside the recording (the hooks keep each first call)
    (lt, ht), rt = record(lambda: _tree(model, ct, toks, meta, verify_scope, TV, mx))
    (lc, hc), rc = record(lambda: fwd(cc, main))
    d1 = first_diff(rt, list(range(6)), rc, list(range(6)))
    lg_same = all(bool(mx.all(lt[n] == lc[n]).item()) for n in range(6))
    cand = {}
    for name, L_ in (("tree", lt), ("chain", lc)):
        row = L_[4]
        cand[name] = dict(argmax=int(mx.argmax(row).item()), l364=float(row[364].item()), l10598=float(row[10598].item()))
    R["t1"] = dict(first_diff=d1, logits_bitwise=lg_same, decision=cand)
    say(f"T1 (check): {'ALL SUBLAYERS BITWISE' if d1 is None else 'FIRST DIFF ' + json.dumps(d1)}; logits bitwise {lg_same}; "
        f"decision row: tree {cand['tree']}, chain {cand['chain']}")
    # T2: the main-chain commit and two SIDE-branch commits (tree16_commit's 'tree' kind), each + the same 4-row probe
    probe = A[95:99]
    R["t2"] = []
    for path in ([0, 1, 2, 3, 4], [0, 1, 10, 11], [0, 1, 2, 3, 8]):
        k = len(path)

        def tree_route():
            c1 = prefill(ids)
            s1 = CS.snapshot_untrimmable_cache_lazy(c1)
            with verify_scope():
                lt_, ht_ = TV.tree_verify(model, c1, toks, meta)
            mx.eval(lt_, ht_)
            kind = TC.tree_commit(model, c1, s1.states, path, T, len(ids), text=inner, ref2=ref2)
            mx.eval(states(c1))
            return c1, kind

        def chain_route():
            c2 = prefill(ids)
            s2 = CS.snapshot_untrimmable_cache_lazy(c2)
            fwd(c2, [toks[n] for n in path])
            assert model.commit_verified_window(c2, s2.states, keep_tokens=k, verified_tokens=k)
            mx.eval(states(c2))
            return c2
        c1, kind = tree_route()
        (lp1, hp1), rp1 = record(lambda: fwd(c1, probe))
        c2 = chain_route()
        (lp2, hp2), rp2 = record(lambda: fwd(c2, probe))
        d2 = first_diff(rp1, list(range(4)), rp2, list(range(4)))
        p_same = bool(mx.all(lp1 == lp2).item())
        R["t2"].append(dict(path=path, commit_kind=kind, first_diff=d2, probe_logits_bitwise=p_same))
        say(f"T2 path {path} (commit '{kind}') + probe: {'ALL SUBLAYERS BITWISE' if d2 is None else 'FIRST DIFF ' + json.dumps(d2)}; "
            f"probe logits bitwise {p_same}")
    d2 = next((x["first_diff"] for x in R["t2"] if x["first_diff"] is not None), None)
    R.update(wall_s=round(time.time() - t0, 1), peak_gb=round(mx.get_peak_memory() / 1e9, 2))
    out = os.path.expanduser("~/BigMoE/data/r57/diag95")
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(os.path.join(out, "diag95.json"), "w"), indent=1, default=str)
    say(f"DIAG95 done ({R['wall_s']:.0f} s, peak {R['peak_gb']:.1f} GB): T1 {'bitwise' if d1 is None else 'layer ' + str(d1['layer']) + ' ' + d1['sublayer']}; "
        f"T2 {'bitwise' if d2 is None else 'layer ' + str(d2['layer']) + ' ' + d2['sublayer']}")


def _tree(model, cache, toks, meta, verify_scope, TV, mx):
    with verify_scope():
        lt, ht = TV.tree_verify(model, cache, toks, meta)
    lt = lt.astype(mx.float32)
    mx.eval(lt, ht)
    return lt, ht
