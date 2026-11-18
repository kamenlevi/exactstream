"""r57_draft: the DRAFTER side of r57's tree loop (MoonShots, Wed 18 Nov 2026; BigMoE's interface 07:52, the orchestrator
08:1x). BigMoE freezes this module into its snapshot. Imports, read-only: ra_sf / sf_tree / s22_format (MoonShots),
tree16_draft / dd_ttt -> mtp125 (BigMoE).

  install(model, fmt, weights=None) -> provenance dict   once, after the 125B load (replaces t16b's mtp125 install)
  begin_window(model, prompt_ids)                          per prompt: the screen's session extras (no-op for 'stored')
  draft(model, head, tok, n=15, k=8, sync_free=True)       -> tree16_draft.draft's dict (tokens / parents int32 in DFS
                                                              preorder, children best-first; depth, T, path_p, p, pop,
                                                              main_chain_len) + 'ms' (wall) + 'bad' (SF frontier flag)
  check_install(model) -> {module: bool}                   the installed weights vs the format's reference

Formats:
  q8h32   MTP Head's t30b_q8h32.safetensors: P's 10 Linears + the 6 HC mixes 8-bit g64 (the artifact's packed tensors),
          se_gatew fp32 with its output in the stream's dtype (CastLinear), experts top-10, ScreenedHead FR-Spec 32K
          (base sorted by token id) + the session extras.
  c2      S22's latent fp32 checkpoint (weights=...), C2's cut via s22_format.FORMATS['c2'] (4-bit Linears / HC, 8-bit
          router, top-4, screen 8K), quantized at load as M-47 (16 / 16 bitwise vs M3's fake quant).
  stored  dd_t30b.safetensors in fp32 with mtp125.head_train's block numerics where the live incremental path allows:
          the fc step as head_train (fp32 sum, then the stream dtype), the MoE block = mtp125.moe_train(...) cast to
          the stream dtype, q / k / v / o fp32 weights with bf16 outputs (the MTP KV cache stores bf16 rows, so the
          attention cannot be head_train's all-fp32 one), the full lm_head, no screen.

head.logits stay FULL-VOCABULARY for every consumer (MTPHead.draft_cut included): the screened row is scattered to V
with -inf off the screen. draft() switches the head to screened rows for its own forwards, rebuilds the root's
screened row from head.logits (bitwise the same values), maps screened index -> token id, and adds the root token to
the session extras first (M3's screen = base + every committed token through the anchor; accepted drafts are already
in the screen, so the roots are the only committed tokens that can be new). If the root token is new, the root's row
is recomputed from head.h with the grown screen (what MTPHead.append would have produced).
The MTP cache is never written by the sync-free draft; the host-synced arm (Route A, tree16_draft.draft as is) trims
it back itself."""
import hashlib
import json
import os
import sys
import time

import numpy as np

H = os.path.expanduser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import ra_sf as RA  # noqa: E402
import tree16_draft as T16D  # noqa: E402

DD_T30B = H("~/datasets/deepdraft/ckpt/dd_t30b.safetensors")
STORED_ATTN = os.environ.get("R57D_STORED_ATTN", "fp32")      # 'stored': fp32 | bf16 q / k / v (+ the KV rows they write)
PK_ON = os.environ.get("R57_TREE_PK") == "1"                   # the persistent level (pk_level.PK), q8h32 + sync-free only;
                                                               # bitwise == draft_sf, falls back to it per tree
ST = {"fmt": None, "new_screen": None, "sh": None, "prov": None, "orig_hc": None, "P": None}


def _sha(path, n=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _lm(model):
    return getattr(model, "language_model", model)


class _Root:
    """an MTPHead view for one draft: the real cache, the root's row in the draft's index space, the root's hidden"""

    def __init__(self, cache, logits, h):
        self.cache, self.logits, self.h = cache, logits, h


class _CastOut(nn.Module):
    """fp32 weight, fp32 product, output in OUT_DTYPE (the stream's)"""

    def __init__(self, w, out_dtype):
        super().__init__()
        self.weight = w
        self.out_dtype = out_dtype

    def __call__(self, x):
        return (x.astype(mx.float32) @ self.weight.T).astype(self.out_dtype)


class _MoeTrain(nn.Module):
    """'stored': mtp125.moe_train(moe, P, x) (fp32 router softmax, argpartition top-k, the stored experts by sorted
    gather_qmm, the fp32 shared expert x sigmoid(fp32 gate)), cast to the stream dtype as head_train (mtp125 L172)"""

    def __init__(self, moe, P, M):                            # NativeMoE's attributes, for code that inspects layer.mlp
        super().__init__()
        self.gate, self.switch_mlp = moe.gate, moe.switch_mlp
        self.shared_expert, self.shared_expert_gate = moe.shared_expert, getattr(moe, "shared_expert_gate", None)
        self.top_k, self.norm_topk_prob = moe.top_k, getattr(moe, "norm_topk_prob", True)
        self.P = P
        object.__setattr__(self, "_M", M)

    def __call__(self, x):
        return self._M.moe_train(self, self.P, x).astype(x.dtype)


def _install_stored(lm, wpath):
    import dd_ttt as DD
    M = DD.M                                                  # dd_topk's mtp125 (DD.M), the same Trainable / moe_train
    head = lm.mtp
    L0 = head.layers[0]
    att, moe = L0.self_attn, L0.mlp
    P = M.Trainable(head)
    P.load_weights(wpath)
    mx.eval(P.parameters())
    sdt = head.pre_fc_norm_hidden.weight.dtype                # the stream dtype (bf16)

    def prep(widened, tok_emb):                               # mtp125.head_train L163-168
        B, S, W_ = widened.shape
        hn = head.pre_fc_norm_hidden(widened).reshape(B, S, head._hc, -1).astype(mx.float32)
        en = P.fc_embedding(head.pre_fc_norm_embedding(tok_emb).astype(mx.float32))
        return (P.fc_hidden(hn) + en[:, :, None, :]).reshape(B, S, W_).astype(widened.dtype)
    object.__setattr__(head, "_mtp_prepare_inputs_impl", prep)   # mtp_forward's route (fuse_and_run)
    object.__setattr__(head, "_prepare_inputs_eager", prep)      # mtp_update_cache's route (fuse_and_run_history calls
    head.fc_hidden, head.fc_embedding = P.fc_hidden, P.fc_embedding   # _prepare_inputs_eager with the head's own fc)
    if getattr(att, "qkv_fused", None) is not None:
        att.qkv_fused = None
    qkv_dt = mx.float32 if STORED_ATTN == "fp32" else sdt      # fp32: q / k / v and the attention in fp32 (dd_topk's
    att.q_proj, att.k_proj = _CastOut(P.q_proj.weight, qkv_dt), _CastOut(P.k_proj.weight, qkv_dt)   # qkv / ttt_attn);
    att.v_proj, att.o_proj = _CastOut(P.v_proj.weight, qkv_dt), _CastOut(P.o_proj.weight, sdt)      # o_proj -> stream
    L0.mlp = _MoeTrain(moe, P, M)
    mx.eval(head.parameters())
    ST["P"] = P
    return ["fc: head_train's fp32 sum -> stream dtype", "q / k / v / o: fp32 weights, outputs in the stream dtype",
            f"MoE block: mtp125.moe_train -> stream dtype (top-{moe.top_k} as loaded)", "draft head: the full lm_head"]


def install(model, fmt, weights=None):
    """once per process, after the 125B load"""
    assert ST["fmt"] is None, "install() runs once per process"
    lm = _lm(model)
    head = lm.mtp
    prov = {"fmt": fmt}
    if fmt in ("q8h32", "c2"):
        assert fmt == "q8h32" or weights, "c2 needs S22's latent checkpoint (weights=...)"
        L0 = head.layers[0]
        ST["orig_hc"] = {(hn, sub): getattr(g, sub).weight for hn, g in (
            ("attn_hc", L0.attn_hyper_connection), ("mlp_hc", L0.mlp_hyper_connection),
            ("mixer", head.hyper_connection_mixer)) for sub in ("input_mix_weight_down", "input_mix_weight_up")}
        RA.SCREENED["on"] = False                             # full-vocabulary rows for every consumer
        done, new_screen, screen = RA.install_format(lm, fmt, weights, screen_order="id")
        wpath = RA.Q8H32_FILE if fmt == "q8h32" else weights
        prov.update(weights=wpath, weights_sha256=_sha(wpath), screen_file=RA.RANK_F, screen_sha256=_sha(RA.RANK_F),
                    screen=int(screen), screen_order="id", extra_cap=4096, top_k=int(L0.mlp.top_k), done=done)
        ST["new_screen"] = new_screen
    elif fmt == "stored":
        wpath = weights or DD_T30B
        done = _install_stored(lm, wpath)
        prov.update(weights=wpath, weights_sha256=_sha(wpath), screen=None, top_k=int(head.layers[0].mlp.top_k),
                    done=done)
    else:
        raise ValueError(f"fmt {fmt!r}: q8h32 | c2 | stored")
    from mtplx.runtime_options import qwen4_opdiet_enabled as _od
    prov["env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("MTPLX_", "TL_", "R57_"))}
    prov["opdiet"] = {it: bool(_od(it)) for it in ("rope", "resid", "bank")}   # the static draft follows the same
    ST["fmt"], ST["prov"] = fmt, prov                                          # switches as the live path
    print(f"[r57_draft] install {json.dumps(prov)}", flush=True)
    return prov


def begin_window(model, prompt_ids):
    """per prompt: a fresh screen with the prompt's tokens as session extras (first-appearance order)"""
    if ST["fmt"] == "stored":
        return
    sh = ST["new_screen"]()
    sh.add(np.asarray(prompt_ids, np.int32))
    ST["sh"] = sh


def observe(model, ids):
    """optional, for harnesses that commit tokens without drafting from them: add ids to the session extras"""
    if ST["fmt"] != "stored" and len(ids):
        ST["sh"].add(np.asarray(ids, np.int32))


def _screened_row(full_row, sh):
    """the screened row (base, used extras, -inf for the unused extra slots) gathered from the full-V row"""
    ne = int(sh.n_extra)
    ids = mx.concatenate([sh.ids, sh.xids[:ne]]) if ne else sh.ids
    row = mx.take(full_row.astype(mx.float32), ids)
    pad = int(sh.cap) - ne
    return mx.concatenate([row, mx.full((pad,), -float("inf"), mx.float32)]) if pad else row


def draft(model, head, tok, n=15, k=8, sync_free=True):
    """tree16_draft.draft's contract on BigMoE's MTPHead (cache, logits [1, 1, V] full-vocabulary, h); tokens as
    vocabulary ids. sync_free=False: Route A (tree16_draft.draft as is) on the same rows, the host-synced reference."""
    t0 = time.perf_counter()
    lm = _lm(model)
    if ST["fmt"] == "stored":
        imap, root = None, head.logits[0, -1].astype(mx.float32)
    else:
        sh = ST["sh"]
        assert sh is not None, "begin_window() first"
        ne0 = int(sh.n_extra)
        sh.add(np.array([int(tok)], np.int32))
        imap = RA.idmap(sh)
        if int(sh.n_extra) != ne0:                            # a new extra: the root's row as append would give it now
            RA.SCREENED["on"] = True
            try:
                root = lm._mtp_draft_head_logits(lm.mtp.hyper_connection_mixer(head.h))[0, -1].astype(mx.float32)
            finally:
                RA.SCREENED["on"] = False
        else:
            root = _screened_row(head.logits[0, -1], sh)
    view = _Root(head.cache, root[None, None, :], head.h)
    L0 = int(head.cache[0].offset)
    assert not getattr(head.cache[0], "fixed_capacity", False), "a fixed-capacity MTP cache: RoPE from rope_offset"
    RA.SCREENED["on"] = imap is not None
    try:
        out = None
        if sync_free and PK_ON and ST["fmt"] == "q8h32":
            pk = ST.setdefault("pk", {}).get((n, k))
            if pk is None:
                import pk_level as PKL
                pk = ST["pk"][(n, k)] = PKL.PK(lm, n=n, k=k)
            out = pk.draft(view, int(tok), ST["sh"], imap)     # None: the 2-pass attention range or a barrier timeout
            if out is not None:
                ST["pk_trees"] = ST.get("pk_trees", 0) + 1
            else:
                ST["pk_fallback"] = ST.get("pk_fallback", 0) + 1
        if out is not None:
            pass
        elif sync_free:
            out = RA.draft_sf(lm, view, int(tok), n=n, k=k, imap=imap)
        else:
            imap_np = None if imap is None else np.array(imap)

            def fwd(h_, idx_):
                tk_ = int(idx_) if imap_np is None else int(imap_np[int(idx_)])
                return model.mtp_forward(h_, mx.array([[tk_]]), mtp_cache=head.cache, return_hidden=True)
            out = T16D.draft(model, view, int(tok), n=n, k=k, fwd=fwd)
            if imap_np is not None:
                out["tokens"] = np.concatenate([out["tokens"][:1], imap_np[out["tokens"][1:]]]).astype(np.int32)
            out["bad"] = 0
    finally:
        RA.SCREENED["on"] = False
    assert int(head.cache[0].offset) == L0, "the draft moved the MTP cache"
    out["ms"] = (time.perf_counter() - t0) * 1e3
    return out


def check_install(model):
    """the installed weights vs the format's reference, bitwise: q8h32 = the artifact's packed tensors (16 packed +
    se_gatew); c2 = M3's fake quant of the checkpoint / the shipped HC weights (M-47's check); stored = dd_t30b"""
    lm = _lm(model)
    head = lm.mtp
    fmt, out = ST["fmt"], {}

    def eq(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())
    if fmt == "stored":
        W = mx.load(ST["prov"]["weights"])
        P = ST["P"]
        for name in ("fc_hidden", "fc_embedding", "q_proj", "k_proj", "v_proj", "o_proj", "router", "se_gate", "se_up",
                     "se_down", "se_gatew"):
            out[f"P.{name}"] = eq(getattr(P, name).weight, W[f"{name}.weight"].astype(mx.float32))
        att = head.layers[0].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[f"live.{name}"] = eq(getattr(att, name).weight, getattr(P, name).weight)
        return out
    L0 = head.layers[0]
    mods = {name: RA._get(head, path) for name, path in RA.P_TO_LIVE.items()}
    hcs = {(hn, sub): getattr(RA._get(head, path), sub) for hn, path in RA.HC_TO_LIVE.items()
           for sub in ("input_mix_weight_down", "input_mix_weight_up")}
    if fmt == "q8h32":
        W = mx.load(RA.Q8H32_FILE)
        for name, m in mods.items():
            if name == "se_gatew":
                out[name] = eq(m.weight, W["P.se_gatew.weight"])
            else:
                out[name] = all(eq(getattr(m, a), W[f"P.{name}.{a}"]) for a in ("weight", "scales", "biases"))
        for (hn, sub), m in hcs.items():
            out[f"{hn}.{sub}"] = all(eq(getattr(m, a), W[f"head.{hn}.{sub}.{a}"]) for a in ("weight", "scales", "biases"))
    else:
        import s22_format as SF22
        import p3_dd_tree as P3
        f = SF22.FORMATS[fmt]
        W = mx.load(ST["prov"]["weights"])
        for name, m in mods.items():
            if name == "se_gatew":
                out[name] = eq(m.weight, W["se_gatew.weight"])
                continue
            bits = f["router"] if name == "router" else f["lin"]
            w0 = W[f"{name}.weight"]
            deq = mx.dequantize(m.weight, m.scales, m.biases, group_size=64, bits=bits).astype(w0.dtype)
            out[name] = eq(deq, P3.fq(mx, w0, bits))
        for (hn, sub), m in hcs.items():
            w0 = ST["orig_hc"][(hn, sub)]
            deq = mx.dequantize(m.weight, m.scales, m.biases, group_size=64, bits=f["hc"]).astype(w0.dtype)
            out[f"{hn}.{sub}"] = eq(deq, P3.fq(mx, w0, f["hc"]))
    out["top_k"] = int(L0.mlp.top_k) == int(ST["prov"]["top_k"])
    return out
