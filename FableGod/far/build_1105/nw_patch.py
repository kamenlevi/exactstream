"""NEVER-WAIT harness patch for the decision map (Fable FAR + ZERO, NEVERWAIT_TEST.md 2.4-2.5; Thu 5 Nov 2026).
Anchored edits to dm_common.py and dm_ablations.py (each anchor must occur exactly once; nothing else is touched):
  dm_common.py
    1. the NEVER-WAIT block (state NW / NWHC / NWDL, _row_lm, _nw_late, _nw_stash, _nw_write, _nw_reset) and, in
       _pert_scores, the late flags read at the SAME row offset as the keep flags (kpos is advanced once, by _row_rm);
       a call with late terms returns the kept-and-not-late scores and leaves the kept-and-late scores in NW["sl"];
    2. _combine_p (>= 32-row chunks) and the narrow path of the MoE wrapper stash delta_L = the late ranks' weighted
       sum of this call's rows (the model's own combine, shared * 0; (y * s_late).sum on the narrow path);
    3. the hc wrapper keeps layer L's MLP-side inject gates, the read picked by identity (mlp_hyper_connection);
    4. a post-wrap on DecoderLayer.__call__ writes delta_{M - d} on the stream layer M returns, through layer
       (M - d)'s own gates (Q._hyper_residual_write); at the last layer the stash must be empty (assert);
    5. discover_layers / clear / run_window / measure: the two id maps, the resets, the mask's `late` array, a
       "never-wait" note per window and res[window]["nw"]; a late-term count that differs from the mask's is a WARNING.
  dm_ablations.py
    nw_mask(path, d, zero) and the arms (snapshot folder <DM>/out/far/, written by install_nw.sh).
An arm without PERT["nw"] takes the unpatched code path (CUR["lm"] is None): its ops are the ones the map ran before.
Usage: python -I nw_patch.py <dir with dm_common.py and dm_ablations.py> <out dir>   (CPU; writes only <out dir>)
Exit 0 = written (or already patched: copied as is), 1 = an anchor did not match (nothing written)."""
import os, sys
src, dst = sys.argv[1], sys.argv[2]
SNAP = os.environ.get("NW_SNAP", "/Users/kamen/FableGod/decision_map/out/far")   # where install_nw.sh snapshots the masks

def rep(s, old, new, where):
    if s.count(old) != 1: raise SystemExit(f"nw_patch: anchor '{where}' occurs {s.count(old)} times (want 1); nothing written")
    return s.replace(old, new)

c = open(os.path.join(src, "dm_common.py")).read(); a = open(os.path.join(src, "dm_ablations.py")).read()
done_c, done_a = "NEVER-WAIT" in c, "def nw_mask(" in a

if not done_c:
    # 5a. discover_layers: the layer and its MLP-side read, by identity
    c = rep(c, '''    LAYERS.clear(); missing = []
''', '''    LAYERS.clear(); NWDL.clear(); NWHC.clear(); missing = []
''', "discover_layers clear")
    c = rep(c, '''        if isinstance(getattr(layer, "mlp", None), Q.SparseMoeBlock): LAYERS[id(layer.mlp)] = (i, "moe")
        else: missing.append((i, "mlp"))
''', '''        if isinstance(getattr(layer, "mlp", None), Q.SparseMoeBlock): LAYERS[id(layer.mlp)] = (i, "moe")
        else: missing.append((i, "mlp"))
        NWDL[id(layer)] = i; NWHC[id(getattr(layer, "mlp_hyper_connection", None))] = i   # never-wait: the layer; its MLP-side read
''', "discover_layers maps")

    # 1. the block and _pert_scores
    c = rep(c, '''def _pert_scores(scores):
    k = PERT.get("topk")
    if CUR.get("rm") is not None:                         # Q1-UNION: keep the ranks the window's mask keeps (mass)
        kp = _row_rm(scores.shape[-2])
        if kp.all(): return scores
''', '''# ---- NEVER-WAIT (Fable FAR + ZERO, ~/FableGod/far/NEVERWAIT_TEST.md 2.4-2.5; Thu 2026-11-05) ----------------------
# An rmask arm with PERT["nw"] = d (dm_ablations.nw_mask): CUR["lm"] is the window's `late` flags [rows, 48, 10], rank-
# sorted like CUR["rm"]. A kept-and-late rank's term is NOT in layer L's block output; the late ranks' weighted sum
# delta_L (layer L's experts on layer L's input, the model's own scores, no renormalisation) is stashed and written d
# layers late: right after layer (L + d)'s own MoE write, on the stream that layer returns, through layer L's OWN
# MLP-side inject gates (_hyper_residual_write is linear, so the term lands in the streams as it would have, late).
# d = 0 writes it straight after layer L's own write (the self-check: the twin up to the order of two adds).
# One MoE call per layer per forward chunk, so the stash holds exactly the current chunk's rows; the mask resolves the
# tail (no late flag at layers >= 48 - d), so nothing is pending when the last layer returns (asserted).
NW = {"sl": None, "stash": {}, "inj": {}, "writes": 0, "terms": 0}
NWHC = {}     # id(layers[i].mlp_hyper_connection) -> i : the MLP-side read, by identity (both reads are (i, "hc"))
NWDL = {}     # id(layers[i]) -> i
def _nw_reset():
    NW.update(sl=None, stash={}, inj={}, writes=0, terms=0)

def _row_lm(n, li, p0):
    """This call's n rows' late flags [n, 10] at layer li, read at the row offset p0 the keep flags were read at
    (_row_rm has advanced CUR['kpos'] once; it is never called twice). Rows past the mask: not late."""
    lt = np.zeros((n, 10), bool); lm = CUR["lm"]
    m = max(0, min(n, lm.shape[0] - p0)); lt[:m] = lm[p0:p0 + m, li]
    return lt

def _nw_late(path, rm):
    """The window's late flags for a never-wait arm: the mask's `late` beside its `keep`. PERT['nw_zero'] forces them
    off (self-check i: the never-wait path with no late flag must reproduce the twin bit for bit)."""
    d = int(PERT["nw"])
    assert rm is not None and 0 <= d <= 2, "never-wait needs an rmask arm and a lateness of 0, 1 or 2 layers"
    assert not any(PERT.get(k_) for k_ in ("t3", "gate", "reuse", "q", "topk")), "never-wait is not wired with t3 / gate / reuse / q / topk"
    if PERT.get("nw_zero"): return np.zeros(rm.shape, bool)
    lm = np.load(path)["late"].astype(bool)
    assert lm.shape == rm.shape and not (lm & ~rm).any(), f"{path}: late must have keep's shape and lie within keep"
    assert d == 0 or not lm[:, lm.shape[1] - d:].any(), f"{path}: the mask must resolve the tail (no late flag in the last {d} layers)"
    return lm

def _nw_stash(li, delta):
    NW["stash"].setdefault(li, []).append(delta)          # this call's rows, in order

def _nw_write(layer, hidden):
    """Post-wrap of DecoderLayer.__call__: layer M has returned its written stream; write delta_{M - d} on it through
    layer (M - d)'s own inject gates. At the last layer nothing may be left pending."""
    M = NWDL.get(id(layer))
    if M is None: return hidden
    st = NW["stash"]; L = M - int(PERT["nw"])
    if L in st:
        parts = st.pop(L); D = parts[0].shape[-1]; lead = tuple(hidden.shape[:-1]); inj = NW["inj"].get(L)
        dl = parts[0].reshape(-1, D) if len(parts) == 1 else mx.concatenate([p.reshape(-1, D) for p in parts], axis=0)
        assert inj is not None and tuple(inj.shape[:-1]) == lead, f"never-wait: layer {L}'s inject gates are missing or misshapen"
        assert dl.shape[0] == int(np.prod(lead)), f"never-wait: layer {L}'s late rows {dl.shape[0]} != the chunk's rows {lead}"
        hidden = Q._hyper_residual_write(hidden, dl.reshape(*lead, D).astype(hidden.dtype), inj)
        NW["writes"] += 1
    if M == len(NWDL) - 1:                                # the end of this forward chunk
        assert not st, f"never-wait: late terms of layers {sorted(st)} were never written (the mask's tail rule)"
        NW["inj"].clear()
    return hidden

def _pert_scores(scores):
    k = PERT.get("topk")
    NW["sl"] = None                                       # never-wait: this call's late scores, set below if it has late terms
    if CUR.get("rm") is not None:                         # Q1-UNION: keep the ranks the window's mask keeps (mass)
        p0 = CUR["kpos"].get(CUR["L"], 0)                 # never-wait: the row offset _row_rm reads at (it advances kpos ONCE)
        kp = _row_rm(scores.shape[-2])
        lt = _row_lm(scores.shape[-2], CUR["L"], p0) & kp if CUR.get("lm") is not None else None
        if lt is not None and lt.any():                   # split the kept ranks: now (in place) and late (stashed by the caller)
            order = mx.argsort(-scores.astype(mx.float32), axis=-1); rank = mx.argsort(order, axis=-1)
            g_ = lambda m_: mx.take_along_axis(mx.array(m_).reshape(rank.shape), rank, axis=-1).astype(scores.dtype)
            NW["sl"] = scores * g_(lt); NW["terms"] += int(lt.sum())
            return scores * g_(kp & ~lt)
        if kp.all(): return scores
''', "_pert_scores")

    # 2. the stash, both chunk widths
    c = rep(c, '''    li = CUR["L"]
    scores = _pert_scores(scores)
    sh = shared * 0 if PERT.get("shared_zero") else shared
''', '''    li = CUR["L"]
    scores = _pert_scores(scores)
    if NW["sl"] is not None: _nw_stash(li, _combine(y_sorted, inv_order, NW["sl"], shared * 0))   # never-wait: delta_L [rows, D]
    sh = shared * 0 if PERT.get("shared_zero") else shared
''', "_combine_p")
    c = rep(c, '''        y = self.switch_mlp(x, inds)
        y = (y * _pert_scores(scores)[..., None]).sum(axis=-2)
''', '''        y = self.switch_mlp(x, inds)
        sc_ = _pert_scores(scores)
        if NW["sl"] is not None: _nw_stash(li, (y * NW["sl"][..., None]).sum(axis=-2))   # never-wait: delta_L
        y = (y * sc_[..., None]).sum(axis=-2)
''', "narrow path")

    # 3. the inject gates of the MLP-side read
    c = rep(c, '''            y = orig(self, *a, **kw)
            post = ACTIVE.get(kind)
''', '''            y = orig(self, *a, **kw)
            if kind == "hc" and CUR.get("lm") is not None and id(self) in NWHC and isinstance(y, tuple) and len(y) == 3:
                NW["inj"][NWHC[id(self)]] = y[2]          # never-wait: layer li's MLP-side inject gates (the read, by identity)
            post = ACTIVE.get(kind)
''', "hc wrapper")

    # 4. the post-wrap on DecoderLayer.__call__
    c = rep(c, '''    wrap(Q.GatedDeltaNet, "gdn"); wrap(Q.Attention, "att"); wrap(Q.GatedResidual, "hc")
    if "moe" in _ORIG: return
''', '''    wrap(Q.GatedDeltaNet, "gdn"); wrap(Q.Attention, "att"); wrap(Q.GatedResidual, "hc")
    if "dl" not in _ORIG:                                 # never-wait: the late write, after the layer's own MoE write
        orig_dl = Q.DecoderLayer.__call__; _ORIG["dl"] = orig_dl
        def dl_call(self, hidden, **kw):
            out = orig_dl(self, hidden, **kw)
            return _nw_write(self, out) if CUR.get("lm") is not None else out
        Q.DecoderLayer.__call__ = dl_call
    if "moe" in _ORIG: return
''', "DecoderLayer wrap")

    # 5b. resets
    c = rep(c, '''    PREV.clear(); CUR["pos"] = {}; CUR["kpos"] = {}; CAPR["L"] = {}; CAPR["X"] = {}; CAPR["W"] = {}   # fresh per window
''', '''    PREV.clear(); CUR["pos"] = {}; CUR["kpos"] = {}; CAPR["L"] = {}; CAPR["X"] = {}; CAPR["W"] = {}   # fresh per window
    _nw_reset()
''', "run_window reset")
    c = rep(c, '''    ACTIVE.clear(); PREV.clear(); PERT.clear(); CUR["k"] = None; CUR["rm"] = None; CUR["skip"] = None; CAPR["on"] = False; CAPR["x"] = False; CAPR["s"] = False
''', '''    ACTIVE.clear(); PREV.clear(); PERT.clear(); CUR["k"] = None; CUR["rm"] = None; CUR["skip"] = None; CAPR["on"] = False; CAPR["x"] = False; CAPR["s"] = False
    CUR["lm"] = None; _nw_reset()
''', "clear")

    # 5c. measure: the late flags, the note, the count check
    c = rep(c, '''        CUR["rm"] = np.load(os.path.join(OUT, PERT["rmask"], nm))["keep"] if PERT.get("rmask") else None
''', '''        CUR["rm"] = np.load(os.path.join(OUT, PERT["rmask"], nm))["keep"] if PERT.get("rmask") else None
        CUR["lm"] = _nw_late(os.path.join(OUT, PERT["rmask"], nm), CUR["rm"]) if PERT.get("nw") is not None else None   # never-wait
''', "measure late")
    c = rep(c, '''            else:
                rk = CUR["rm"][a:len(ids) - 1]; extra += f"; kept experts per row-layer {rk.sum() / max(rk.shape[0] * 48, 1):.2f}"
''', '''            else:
                rk = CUR["rm"][a:len(ids) - 1]; extra += f"; kept experts per row-layer {rk.sum() / max(rk.shape[0] * 48, 1):.2f}"
            if CUR.get("lm") is not None:                   # never-wait: what moved; every late flag must have been used once
                lk = CUR["lm"][a:len(ids) - 1]; nl_ = int(CUR["lm"].sum())
                res[nm]["nw"] = dict(d=int(PERT["nw"]), late=int(lk.sum()), rows=int(lk.shape[0]), writes=NW["writes"], terms=NW["terms"])
                extra += f"; never-wait d {PERT['nw']}: late terms per answer row {lk.sum() / max(lk.shape[0], 1):.2f}, {NW['writes']} late writes"
                if NW["terms"] != nl_ or NW["stash"]:
                    extra += f"; WARNING never-wait used {NW['terms']} late terms, the mask holds {nl_} (rows misaligned)"
                    res[nm]["warn"] = extra
''', "measure note")

if not done_a:
    a += '''def nw_mask(path, d, zero=False):
    """NEVER-WAIT (Fable FAR + ZERO, ~/FableGod/far/NEVERWAIT_TEST.md; Thu 2026-11-05): q1_union(path) whose masks also
    hold `late` [rows, 48, 10] (rank-sorted like keep; ~/FableGod/far/build_1105/nw_masks.py). A kept-and-late rank's term
    of layer L is written d layers late through layer L's own inject gates (dm_common._nw_write); the 4-bit stock path,
    the model's own weights, no _t3. zero = the late flags forced off (self-check i: must equal the twin bit for bit)."""
    def setup():
        PERT.update(rmask=path, renorm=False, nw=d)
        if zero: PERT.update(nw_zero=True)
    return setup
_NWD = "%s/"   # snapshots of ~/FableGod/far/build_1105/masks/ (MD5SUMS beside them), installed by install_nw.sh
ABLATIONS["qom_0_live_ball_c208"] = q1_union(_NWD + "qom_0_live_ball_c208")   # tier 1: the twin, QOM cf 0 + DRAFT17 on 208 uniform slots
ABLATIONS["qom_0_live_ball_c208_nw1"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw1", 1)   # tier 1, decisive: every demand miss one layer late
ABLATIONS["qom_0_live_ball_c208_nw1r1"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw1r1", 1)   # tier 1, fallback: rank-0 experts wait
ABLATIONS["qom_0_live_ball_nw1"] = nw_mask(_NWD + "qom_0_live_ball_nw1", 1)   # tier 1: nw1 at the published pool (twin qom_0_live_ball)
ABLATIONS["qom_0_split0.99_live_ball_nw1"] = nw_mask(_NWD + "qom_0_split0.99_live_ball_nw1", 1)   # tier 1: the stack (twin qom_0_split0.99_live_ball)
ABLATIONS["qom_0_live_ball_c266"] = q1_union(_NWD + "qom_0_live_ball_c266")   # tier 1b: the twin on 266 uniform slots (the engine's by-day base-pool exposure, ZERO 5d)
ABLATIONS["qom_0_live_ball_c266_nw1"] = nw_mask(_NWD + "qom_0_live_ball_c266_nw1", 1)   # tier 1b: nw1 at 266 slots
ABLATIONS["qom_0_live_ball_c208_nwz"] = nw_mask(_NWD + "qom_0_live_ball_c208", 1, zero=True)   # self-check i: the twin's masks through the never-wait path
ABLATIONS["qom_0_live_ball_c208_nw0"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw0", 0)   # self-check ii: lateness 0
ABLATIONS["qom_0_live_ball_c208_nw2"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw2", 2)   # tier 2: two layers late
ABLATIONS["qom_0_live_ball_c208_nw1s"] = nw_mask(_NWD + "qom_0_live_ball_c208_nw1s", 1)   # tier 2: late only on rows whose draft cf >= 0.9
ABLATIONS["qom_0_split0.99_live_ball_c208"] = q1_union(_NWD + "qom_0_split0.99_live_ball_c208")   # tier 2: the stack's twin at 208 slots
ABLATIONS["qom_0_split0.99_live_ball_c208_nw1"] = nw_mask(_NWD + "qom_0_split0.99_live_ball_c208_nw1", 1)   # tier 2: the stack at 208 slots
ABLATIONS["qom_0_split0.9_live_ball_nw1"] = nw_mask(_NWD + "qom_0_split0.9_live_ball_nw1", 1)   # tier 2: split 0.9 + nw1 (twin qom_0_split0.9_live_ball)
''' % SNAP

os.makedirs(dst, exist_ok=True)
open(os.path.join(dst, "dm_common.py"), "w").write(c); open(os.path.join(dst, "dm_ablations.py"), "w").write(a)
print(f"nw_patch: dm_common.py {'already patched' if done_c else 'patched'}, dm_ablations.py {'already patched' if done_a else 'patched'} -> {dst}")
