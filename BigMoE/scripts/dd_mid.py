#!/usr/bin/env python3
"""BREAK98 MID CAPTURE (Tue 17 Nov 2026; the orchestrator 07:4x, NewIdeas 07:41 / 07:44): the 125B's PRE-MIXER widened
stream (the 10,240-wide hyper-connection stream h, fp16) after L decoder layers, L in MID_LAYERS (default 2,4,8,16,24),
at every answer row (gen-1 .. N-2) of the bank's heplus24 + mbppbank texts, on the bank's own runtime path. The model,
preset, prefill and verify loop are dd_bank.py's, copied verbatim (its lines 16-97 and 116-175); only the taps and the
driver are new.
- Layer L = the output of TXT.layers[L-1]: the stream the final mixer would read if the trunk stopped after L layers.
  L = 48 would be the bank's wd (TXT._last_widened, which the final hyper-connection mixer reads).
- Self-checks per file, against the bank file (~/datasets/deepdraft/bank/<key>.npz): (1) this run's final widened
  stream == the stored wd BITWISE on every row; (2) the mean of the 4 HC streams after layers[15] == the stored raw
  tap 15 bitwise (a mid-layer check of the same pass); (3) the top-32 ids == the stored tk on every answer row; (4) no
  inf / nan after the fp16 cast (max |x| per layer is logged).
- Output per tag: <MID_OUT>/mid_<tag>.npz (default data/deepdraft/ceiling): X [N, 10240] fp16 (rows ordered file,
  layer, row), fi / row / layer [N] int32 (fi indexes `files` = tree_t30b_<tag>_n127k8.npz's file order), files,
  layers; + mid_<tag>.json (per-file checks, times, the footprint). The npz is stored (uncompressed) and streamed file
  by file; X is its only 2-D 10,240-wide member (ed_stage2.py picks the state array by that shape).
usage: BIGMOE_EXT_BUILD=build_qom [BigMoE venv] dd_mid.py C [TAGS]   (C 100 and build_qom as the bank's own launch: the
  records3 store is native+p11+pad, which needs an ext with set_direct_split; the copied default build_next lacks it;
  TAGS default heplus24,mbppbank)
  env: MID_LAYERS; MID_FILES=k (head test: the first k files of each tag); MID_OUT (folder); MID_SFX (name suffix);
  AWAYMIN (> 0: stop at a file boundary when away_secs < AWAYMIN); MID_STOP (a file: stop at the next file boundary).
  A stop drops the unfinished tag's file (finished tags stay; a re-run skips them). rc 0 = written + all checks PASS,
  3 = written but a check FAILED, 4 = stopped.
"""
import json, os, random, re, shutil, sys, threading, time
os.environ.pop("MTPLX_FUSED_GATE_UP", None)            # the streamed engine has no fused gate | up
os.environ.pop("BIGMOE_VERIFY_MIDLOOP", None)          # SB comes from the lambda below, as in S45
os.environ.pop("BIGMOE_MTP_STREAM", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["BIGMOE_TIER"] = "4433333333"; os.environ["BIGMOE_TIER_GLU"] = "3c"   # k69_live R57_PRESET=45 (REF2-T)
os.environ.setdefault("BIGMOE_TIER_CL", "112")
os.environ.pop("BIGMOE_TIER_COMBINE", None); os.environ.pop("BIGMOE_OOO", None)  # D-23's preset (D23_NEWDEF=0)
os.environ.setdefault("BIGMOE_MAXK", str(10 * int(os.environ.get("DLOG_ROWS", "16"))))   # E2: 10 x rows (160 as shipped)
os.environ.setdefault("BIGMOE_EXT_BUILD", "build_next")
assert os.environ.get("BIGMOE_MAXK") == str(10 * int(os.environ.get("DLOG_ROWS", "16")))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(1, os.path.expanduser("~/Leap/third_party"))
import evalsuite.env                                     # EvalSuite's forced / refused env (read-only use), before mlx
from evalsuite import model as EM, data as ED
from evalsuite.decode import footprint_now_gb, footprint_peak_gb
import numpy as np
import mlx.core as mx
import dflash_model_mlx as DF

H = os.path.expanduser
PAUSE = H("~/.local/share/quiet/PAUSE")
C = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TRUNC, R, K, TOPK = 0.3, 16, 16, 32
ROWS = int(os.environ.get("DLOG_ROWS", "16"))           # E2: rows per check (16 = as shipped; fewer at small C, exact)
EOS = {248044, 248046}
TAPS = (5, 15, 25, 35, 45)
PRESET = "45 (REF2-T 4433333333/3c, C_l 112, MAXK 160, build_next, L1-20 + cut 0.3; D-23's)"

ACT = [0.0]                                              # running time without SIGSTOP freezes (Leap's ticker)
def _tick():
    last = time.time()
    while True:
        time.sleep(1)
        now = time.time()
        ACT[0] += min(now - last, 3.0)
        last = now
threading.Thread(target=_tick, daemon=True).start()
el = lambda: ACT[0]
def say(s):
    print(f"[dlog {time.strftime('%H:%M:%S')}] {s}", flush=True)

# ---------------------------------------------------------------- the default's model (D-23's preset; no MTP head needed)
t0 = time.time()
rt, st = EM.load(EM.variant("ref"), C=C)
model = rt.model
try:                                                     # E2 small-C mode: rows x top-10 must fit every layer's base pool
    _cm = min(sw._bm["C"] for sw in st._tier_sws)
    if ROWS * 10 > _cm:
        say(f"E2: smallest base pool C_m {_cm} < {ROWS} rows x 10: rows -> {_cm // 10}")
        ROWS = _cm // 10
    say(f"E2: C {C}, base pool C_m >= {_cm}, rows per check {ROWS}, BIGMOE_MAXK {os.environ.get('BIGMOE_MAXK')}")
except Exception as e:
    say(f"E2: C_m check skipped ({e!r})")
import ref2
import bigmoe.stream as BS
import bigmoe.gated as BG
import bigmoe.tier as _BT
from mtplx import cache_state as CS
Q = ref2.Q
assert BG.MAXK == int(os.environ["BIGMOE_MAXK"]) and BG.XLINES == (0 if BG.MAXK <= 61 else min(4, (BG.MAXK - 61 + 30) // 31)), \
    (BG.MAXK, BG.XLINES)                                # 160 / 4 as shipped; E2 small-C mode: 10 x rows
assert not _BT._COMBINE and not _BT._OOO, "D-23's pre-OOO preset expected"
TXT = model.language_model.model
lay = TXT.layers
SW_ = [l.mlp.switch_mlp for l in lay]
TOKZ = rt.tokenizer
say(f"loaded in {time.time() - t0:.0f} s, {mx.get_active_memory() / 1e9:.1f} GB active; C {C}; preset {PRESET}")

def set_arm(arm):                                        # k69_live's: the prefill as shipped ('slot'), the loop on the gate
    BS.GATE_N[0] = min(40, BG.MAXK) if arm == "slot" else BG.MAXK   # 40 / 160 as shipped (E2 small-C: <= the pool)
    BS._PF_RANKED = False
    for sw in SW_:
        sw._bm.pop("next", None)

def pf1(on):
    if not on and BS._STAGE["bufs"] is not None:
        BS._stage_drop()
    BS._PF_PREFILL = on

# ---------------------------------------------------------------- MID taps (new): the widened stream after L layers + a check tap
LAYERS = tuple(int(x) for x in os.environ.get("MID_LAYERS", "2,4,8,16,24").split(","))
assert len(set(LAYERS)) == len(LAYERS) and all(1 <= L < len(lay) for L in LAYERS), LAYERS
RAWCHK = TAPS.index(15)                                  # the bank's raw tap 15: the mean of the 4 HC streams after layers[15]
ROLE = {}
for L in LAYERS:
    ROLE.setdefault(id(lay[L - 1]), []).append(L)
ROLE.setdefault(id(lay[15]), []).append("R")
TAP = {"on": False, "X": {}, "R": []}
_layer_call = Q.DecoderLayer.__call__
def _layer(self, *a, **kw):
    out = _layer_call(self, *a, **kw)
    roles = ROLE.get(id(self))
    if TAP["on"] and roles:
        o = out if isinstance(out, mx.array) else out[0]
        B, L_, W = o.shape
        for r in roles:
            if r == "R":                                 # exactly dd_bank.py's tap recipe
                TAP["R"].append(o.reshape(B, L_, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16))
            else:
                TAP["X"][r].append(o[0].astype(mx.float16))
    return out
Q.DecoderLayer.__call__ = _layer
def tap_reset():
    TAP["X"] = {L: [] for L in LAYERS}; TAP["R"] = []
tap_reset()
def tap_last():
    return [TAP["X"][L][-1] for L in LAYERS] + [TAP["R"][-1]]
def tap_get(calls):
    assert all(len(TAP["X"][L]) == calls for L in LAYERS) and len(TAP["R"]) == calls, \
        ({L: len(TAP["X"][L]) for L in LAYERS}, len(TAP["R"]), calls)   # every tap fired once per forward
    cat = lambda x: mx.concatenate(x) if len(x) > 1 else x[0]
    return {L: cat(TAP["X"][L]) for L in LAYERS}, cat(TAP["R"])


draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.load_weights(H("~/Leap/data/l120_ft.safetensors"), strict=False)
draft.bind(model)
mx.eval(draft.parameters())
MASK, SWIN = int(draft.config.mask_token_id), int(draft.config.sliding_window)

def head2s(hid):
    lg = draft.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)

def chain(blk, unary, cand, hp):
    sel = draft.candidate_selector
    pred, path, conf = blk[:, 0], [], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        s_ = mx.argmax(sc, axis=-1)
        pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
        path.append(pred)
        conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
    return mx.stack(path, axis=1), mx.stack(conf, axis=1)

def propose(blk, ctx, dcache):
    hid = draft.hidden_states(blk, ctx, dcache, 1)
    cand, unary = head2s(hid)
    return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))

_midloop = Q._prefill_midloop_eval_layers
def ref_on(on):
    if on:
        ref2.apply()
        Q._prefill_midloop_eval_layers = lambda rows: _midloop(rows) or (4 if 2 <= int(rows) <= 16 else 0)   # SB
    else:
        ref2.remove()
        Q._prefill_midloop_eval_layers = _midloop

def states(cache):
    return [c.state for c in cache if c is not None and hasattr(c, "state")]

# ---------------------------------------------------------------- the logger: reads arrays the loop already evaluated
def top32(lg):
    """Leap's dfl2c_extract recipe: the 125B's top-32 ids (best first) and log-probs (fp16) per row"""
    lg = lg.astype(mx.float32)
    top = mx.argpartition(-lg, TOPK - 1, axis=-1)[:, :TOPK]
    v = mx.take_along_axis(lg, top, axis=-1)
    o = mx.argsort(-v, axis=-1)
    top, v = mx.take_along_axis(top, o, axis=-1), mx.take_along_axis(v, o, axis=-1)
    return top.astype(mx.int32), (v - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float16)

def hbits(tp):
    """the drafter's own context input for these tap rows, hidden_norm(fc(taps)) in bf16 (stored as uint16 bits in save)"""
    return draft.hidden_norm(draft.fc(tp.astype(mx.bfloat16)))

def prefill(ids, chunk=1024):
    """dd_bank.py's prefill (chunks of 1024 on the PF1 path) + each chunk's widened stream, with the MID taps"""
    pf1(True)
    cache, lg, ws, calls = model.make_cache(), None, [], 0
    tap_reset(); TAP["on"] = True
    for a in range(0, len(ids), chunk):
        lg = model(mx.array([ids[a:a + chunk]]), cache=cache)
        w = TXT._last_widened
        calls += 1
        mx.eval(lg, states(cache), tap_last(), w)
        assert w.shape[1] == min(chunk, len(ids) - a), w.shape
        ws.append(w[0].astype(mx.float16))
    TAP["on"] = False
    pf1(False)
    X, R_ = tap_get(calls)
    return cache, lg[0, -1], X, R_, mx.concatenate(ws) if len(ws) > 1 else ws[0]



# ---------------------------------------------------------------- driver (new): the bank's runtime path per file, per tag
import subprocess, zipfile
CE = H("~/BigMoE/data/deepdraft/ceiling"); BANK = H("~/datasets/deepdraft/bank")
OUT = H(os.environ.get("MID_OUT", CE)); os.makedirs(OUT, exist_ok=True)
SFX = os.environ.get("MID_SFX", ""); assert re.fullmatch(r"[a-z0-9_]*", SFX), SFX
TAGS = (sys.argv[2] if len(sys.argv) > 2 else "heplus24,mbppbank").split(",")
NF = int(os.environ.get("MID_FILES", "0"))
AWAYMIN = int(os.environ.get("AWAYMIN", "0")); STOPF = os.environ.get("MID_STOP", "")
RC = [0]


def away_secs():
    try:
        return int(subprocess.run(["bash", H("~/.local/share/quiet/away_secs.sh")], capture_output=True, text=True,
                                  timeout=20).stdout.strip())
    except Exception:
        return -1                                        # no reading = at the Mac (as g1_follow.sh)


def stop_why():
    if STOPF and os.path.exists(STOPF):
        return f"stop file {STOPF}"
    if AWAYMIN > 0:
        a = away_secs()
        if a < AWAYMIN:
> (a quote from the project owner, kept private)
    return None


def run_file(f):
    """one bank file on dd_bank.py's runtime path: its answer-row states per layer + the checks against the bank"""
    z = np.load(os.path.join(BANK, f))
    ids = [int(t) for t in z["ids"]]
    gen = int(z["gen"]); N = len(ids); n = N - 1
    t1 = time.perf_counter()
    set_arm("slot"); ref_on(True)
    cache, last, X0, R0, W0 = prefill(ids[:gen])
    ref2.remove()
    tk0, _ = top32(last[None])
    Xs = {L: [X0[L][gen - 1:gen]] for L in LAYERS}      # the last prompt row = the first answer row (gen - 1)
    mx.eval([x[0] for x in Xs.values()])
    del X0
    Rs, Ws, tks = [R0], [W0], [tk0]
    ref2.apply(); set_arm("gate")
    a = gen
    while a < n:
        ch = ids[a:min(a + ROWS, n)]
        snap = CS.snapshot_untrimmable_cache_lazy(cache)
        tap_reset(); TAP["on"] = True
        with ref2.verify_scope(model):
            lg = model(mx.array([ch]), cache=cache)
        TAP["on"] = False
        X_, R_ = tap_get(1); wv = TXT._last_widened
        mx.eval(lg, list(X_.values()), R_, wv)
        model.commit_verified_window(cache, snap.states, keep_tokens=len(ch), verified_tokens=len(ch))
        mx.async_eval(states(cache))
        ref2.remove()
        tk_, _ = top32(lg[0, :len(ch)])
        ref2.apply()
        for L in LAYERS:
            Xs[L].append(X_[L][:len(ch)])
        Rs.append(R_[:len(ch)]); Ws.append(wv[0, :len(ch)].astype(mx.float16)); tks.append(tk_)
        a += len(ch)
    mx.eval(states(cache))
    ref2.remove()
    nr = n - gen + 1
    X = {L: np.concatenate([np.array(x) for x in Xs[L]]) for L in LAYERS}
    R = np.concatenate([np.array(x) for x in Rs])[:n]; W = np.concatenate([np.array(x) for x in Ws])[:n]
    tk = np.concatenate([np.array(x) for x in tks])[:nr]
    assert all(X[L].shape == (nr, 10240) and X[L].dtype == np.float16 for L in LAYERS), {L: X[L].shape for L in LAYERS}
    assert R.shape == (n, 2560) and W.shape == (n, 10240) and tk.shape == (nr, 32), (R.shape, W.shape, tk.shape)
    del cache
    mx.clear_cache()
    wd_b = np.asarray(z["wd"]); raw_b = np.asarray(z["raw"])[:, 2560 * RAWCHK:2560 * (RAWCHK + 1)]
    tk_b = np.asarray(z["tk"])[gen - 1:n]
    assert wd_b.shape == W.shape and raw_b.shape == R.shape and tk_b.shape == tk.shape, (wd_b.shape, raw_b.shape, tk_b.shape)
    ck = dict(key=f, N=N, gen=gen, rows=nr,
              wd_rows_differ=int((W.view(np.uint16) != wd_b.view(np.uint16)).any(axis=1).sum()),       # bitwise, every row
              raw15_rows_differ=int((R.view(np.uint16) != raw_b.view(np.uint16)).any(axis=1).sum()),
              tk_rows_differ=int((tk != tk_b).any(axis=1).sum()),
              nonfinite={L: int((~np.isfinite(X[L])).sum()) for L in LAYERS},
              absmax={L: float(np.abs(X[L].astype(np.float32)).max()) for L in LAYERS},
              s=round(time.perf_counter() - t1, 1))
    ck["ok"] = ck["wd_rows_differ"] == 0 and ck["raw15_rows_differ"] == 0 and ck["tk_rows_differ"] == 0 \
        and not any(ck["nonfinite"].values())
    return X, ck


def write_small(zf, name, v):
    with zf.open(name + ".npy", "w", force_zip64=True) as fo:
        np.lib.format.write_array(fo, np.asanyarray(v), allow_pickle=False)


say(f"mid capture: tags {TAGS}, layers {LAYERS}, C {C}, rows per check {ROWS}, files per tag {NF or 'all'} -> {OUT}"
    f" (suffix '{SFX}'); away gate {AWAYMIN} s; stop file {STOPF or '-'}")
for tag in TAGS:
    dst = os.path.join(OUT, f"mid_{tag}{SFX}.npz"); js = os.path.join(OUT, f"mid_{tag}{SFX}.json")
    if os.path.exists(dst):
        say(f"{tag}: {dst} exists, skipped")
        continue
    files = [str(x) for x in np.load(os.path.join(CE, f"tree_t30b_{tag}_n127k8.npz"), allow_pickle=True)["files"]]
    if NF > 0:
        files = files[:NF]
    nrs = []
    for f in files:
        b = np.load(os.path.join(BANK, f))
        nrs.append(len(b["ids"]) - int(b["gen"]))       # answer rows gen-1 .. N-2
    NT = sum(nrs) * len(LAYERS)
    say(f"{tag}: {len(files)} files, {sum(nrs)} answer rows x {len(LAYERS)} layers -> X [{NT}, 10240] fp16 "
        f"({NT * 10240 * 2 / 1e9:.2f} GB) -> {dst}")
    tmp = dst + ".tmp"
    zf = zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
    fx = zf.open("X.npy", "w", force_zip64=True)
    np.lib.format.write_array_header_1_0(fx, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                                             "fortran_order": False, "shape": (NT, 10240)})
    FI, ROW, LAY, CKS, nb, why, t0 = [], [], [], [], 0, None, time.perf_counter()
    for i, f in enumerate(files):
        why = stop_why()
        if why:
            break
        while os.path.exists(PAUSE):
            time.sleep(5)
        X, ck = run_file(f)
        gen, nr = ck["gen"], ck["rows"]
        assert nr == nrs[i], (f, nr, nrs[i])
        for L in LAYERS:
            fx.write(np.ascontiguousarray(X[L]).tobytes()); nb += X[L].nbytes
            FI.append(np.full(nr, i, np.int32)); ROW.append(np.arange(gen - 1, gen - 1 + nr, dtype=np.int32))
            LAY.append(np.full(nr, L, np.int32))
        del X
        CKS.append(ck)
        say(f"{tag} {i + 1}/{len(files)} {f}: {ck['N']} tokens (prompt {gen}), {nr} answer rows in {ck['s']} s; rows "
            f"differing from the bank: wd {ck['wd_rows_differ']}, raw15 {ck['raw15_rows_differ']}, tk {ck['tk_rows_differ']}; "
            f"nonfinite {sum(ck['nonfinite'].values())}, max |x| {max(ck['absmax'].values()):.0f}"
            + ("" if ck["ok"] else "  CHECK FAILED") + f"; footprint {footprint_now_gb():.1f} GB, peak {footprint_peak_gb():.1f} GB")
    if why:
        fx.close(); zf.close(); os.remove(tmp)
        say(f"STOPPED before {tag} file {len(CKS) + 1}/{len(files)}: {why}; the unfinished {tag} file was dropped")
        RC[0] = 4
        break
    fx.close()
    assert nb == NT * 10240 * 2, (nb, NT)
    fi_, row_, lay_ = np.concatenate(FI), np.concatenate(ROW), np.concatenate(LAY)
    assert len(fi_) == len(row_) == len(lay_) == NT, (len(fi_), NT)
    ok = all(c["ok"] for c in CKS)
    for k, v in (("fi", fi_), ("row", row_), ("layer", lay_), ("files", np.array(files)),
                 ("layers", np.array(LAYERS, np.int32)), ("checks_ok", np.array([c["ok"] for c in CKS]))):
        write_small(zf, k, v)
    zf.close()
    os.replace(tmp, dst)
    S = dict(tag=tag, files=len(files), answer_rows=int(sum(nrs)), layers=list(LAYERS), X_shape=[NT, 10240], dtype="float16",
             bytes=os.path.getsize(dst), seconds=round(time.perf_counter() - t0, 1), all_checks_pass=ok,
             wd_rows_differ=sum(c["wd_rows_differ"] for c in CKS), raw15_rows_differ=sum(c["raw15_rows_differ"] for c in CKS),
             tk_rows_differ=sum(c["tk_rows_differ"] for c in CKS),
             nonfinite=sum(sum(c["nonfinite"].values()) for c in CKS),
             absmax={str(L): max(c["absmax"][L] for c in CKS) for L in LAYERS},
             footprint_peak_gb=round(footprint_peak_gb(), 2), C=C, rows_per_check=ROWS, preset=PRESET,
             path=f"runtime: prompt prefill (1,024 grid) + {ROWS}-row verify calls, C {C} (dd_bank.py's)",
             layer_def="layer L = the output of TXT.layers[L-1] (the pre-mixer widened stream after L layers; 48 = wd)",
             per_file=CKS)
    json.dump(S, open(js, "w"), indent=1)
    say(f"{tag}: WROTE {dst} ({S['bytes'] / 1e9:.2f} GB, {S['seconds']:.0f} s); checks "
        f"{'PASS' if ok else 'FAILED'}: rows differing wd {S['wd_rows_differ']}, raw15 {S['raw15_rows_differ']}, "
        f"tk {S['tk_rows_differ']}; nonfinite {S['nonfinite']}; max |x| {S['absmax']}")
    if not ok:
        RC[0] = max(RC[0], 3)
say(f"mid capture finished, rc {RC[0]}; footprint peak {footprint_peak_gb():.1f} GB")
sys.exit(RC[0])
