"""Fit12 S1 DF-TF (Sun 15 Nov 01:3x; WEEK98 ~/FableGod/deepdraft/LEVERS_A.md 'S1', NewIdeas 01:1x; line written before the run):
DFlash2 (l120_ft, the 12 GB engine's live drafter: d7b_live 4-bit, bound to B12's embedding + lm_head) at BLOCK 32, offline on
the 125B BANK (~/datasets/deepdraft/bank: raw = the runtime-path taps {5,15,25,35,45} x 2560, ids = the 125B's greedy text,
tk[t, 0] = ids[t + 1] checked 1.0 on the generation). DFlash2's config is is_causal false (bidirectional in-block attention), so
the teacher-forced read takes one pass per block position j: block = [ids[a .. a + j - 1], MASK x (32 - j)], the prediction at j
(its top-1) vs ids[a + j]. The native control = the all-masked block (j = 0's pass), scored as kept = 1 + its leading matches.
Anchors: a = gen - 1, then every 32 tokens while a + 31 <= the last id (block position 0 = the committed token, context = the taps
of positions < a, as the engine's proposer). The drafter's context cache is filled once per anchor; the 31 teacher-forced passes
reuse it (no new context), self-checked against the native pass at j = 1.
Sets: MBPP 100 (dlog__mbpp*), HE+ 24 (tf_code_HumanEval*). Bands by depth d = j - 1: 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30.
LINE (LEVERS_A S1): PASS if teacher-forced HE+ 23-30 >= 95.0 and within 3 pt of its own depth 0, and MBPP 23-30 >= 85;
BETWEEN if HE+ 23-30 is 90-95; KILL if < 90. The native kept/32 should reproduce L1-20's ~10.5 (+/- 0.5).
Usage: python -B s1tf.py OUT_DIR (writes s1tf.json + s1tf.txt)"""
import glob, json, os, sys, time
OUT = sys.argv[1]
ENV = {"MTPLX_PREFILL_CHUNK_SIZE": "512", "BIGMOE_TIER_PF_GROUP": "64", "BIGMOE_RESET_GEN": "0", "EVALSUITE_SHIP": "1",
       "B12_LATE3": "2,10", "B12_NORM": "mass", "BIGMOE_PF_GROUP": "64", "BIGMOE_ASYNC_EVERY": "1",
       "B12_TRIMS": "1", "B12_PFR": "1", "B12_ALLWAIT": "0", "B12_CACHE_MB": "64",
       "BIGMOE_LOOP_GUARD": "0", "MTPLX_NGRAM_PREWARM": "off", "MTPLX_COMPILED_VERIFY": "0", "MTPLX_QWEN4EXP_COMPILE": "0",
       "MTPLX_COMPILE_AR_FORWARD": "0", "MTPLX_COMPILED_TARGET_PREFIX": "0", "BIGMOE_SPARE": "0", "BIGMOE_PF_NOPUBLISH": "1",
       "MTPLX_FAMILY_CAPTURE_COMMIT": "1", "MTPLX_DROP_EVENTS": "1", "MTPLX_DEFER_REPAIR_EVAL": "1",
       "B12_CM": "47", "B12_CL": "32"}                        # fit_onecall.py's process env (the engine as it runs)
for k, v in ENV.items():
    os.environ.setdefault(k, v)
BT = os.path.expanduser("~/Fit12/pinned/8db7/bt_b12.py")
sys.path.insert(0, os.path.expanduser("~/Breakthrough/scripts")); sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
sys.path.insert(0, os.path.dirname(BT)); sys.path.insert(0, os.path.expanduser("~/Fit12/scripts"))
import numpy as np
import mlx.core as mx
import bt_b12 as B
import t3_twopass as T3
T3.apply(B)
t0 = time.time()
rt, _ = B.load_b12(0.10, mtp=False)
draft, DF = T3.load_draft(B, rt.model)
MASK, SW, BL = int(draft.config.mask_token_id), int(draft.config.sliding_window), 32
# (Sun 02:4x) the teacher-forced passes add NO context rows: dflash_model_mlx's attention cannot reshape an empty context
# (-1 on a 0-size array), so in THIS process only (Leap's file is untouched) a pass with an empty context runs the same math with
# the cached context keys: same projections, norms, RoPE offset (cache.offset = the block's position after the native pass's
# update) and the same sliding / bidirectional mask. All DFlash2 layers are sliding_attention (RotatingKVCache): after one
# update of S <= SW - 1 rows the cache holds them in order. Self-checked: j = 1 must equal the native pass.
_Att = type(draft.layers[0].self_attn)
_orig_att = _Att.__call__


def _att0(self, x, x_ctx, rope, cache):
    if x_ctx.shape[1] > 0:
        return _orig_att(self, x, x_ctx, rope, cache)
    Bq, L, _ = x.shape
    q = self.q_norm(self.q_proj(x).reshape(Bq, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
    pk = self.k_norm(self.k_proj(x).reshape(Bq, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    pv = self.v_proj(x).reshape(Bq, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = rope(q, offset=cache.offset)
    pk = rope(pk, offset=cache.offset)
    keys, values = cache.keys, cache.values
    ctx_len = keys.shape[2]
    keys = mx.concatenate([keys, pk], axis=2)
    values = mx.concatenate([values, pv], axis=2)
    mask = None
    if self.is_sliding:
        query = ctx_len + mx.arange(L)[:, None]
        key = mx.arange(ctx_len + L)[None]
        context = (key < ctx_len) & (query - key < self.sliding_window)
        block = key >= ctx_len
        if self.is_causal:
            block = block & (key <= query)
        mask = context | block
    elif self.is_causal:
        from mlx_lm.models.base import create_causal_mask
        mask = create_causal_mask(L, offset=ctx_len)
    out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
    return self.o_proj(out.transpose(0, 2, 1, 3).reshape(Bq, L, -1))


_Att.__call__ = _att0
print(f"[s1tf] engine + drafter loaded in {time.time() - t0:.0f} s; MASK {MASK}, SW {SW}, block {BL}, causal {draft.config.is_causal}", flush=True)
BANK = os.path.expanduser("~/datasets/deepdraft/bank")
SETS = {"mbpp": sorted(glob.glob(f"{BANK}/dlog__mbpp*.npz")), "humaneval": sorted(glob.glob(f"{BANK}/tf_code_HumanEval*.npz"))}
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
empty = mx.zeros((1, 0, 12800), mx.bfloat16)
res, selfchk = {}, [0, 0]
HV, NV, AN = {}, {}, {}                                     # per set: TF hit vectors, native hit vectors, anchor ids
HEAD = int(os.environ.get("S1TF_HEAD", "0"))               # > 0: a head test of that many anchors (the orchestrator 02:35)
for name, files in SETS.items():
    tf = np.zeros((31,), np.int64); nat = np.zeros((31,), np.int64); n_anch = 0; kept = []
    for f in files:
        z = np.load(f)
        ids, raw, g = z["ids"].astype(np.int64), z["raw"], int(z["gen"])
        a = g - 1
        while a + BL - 1 <= len(ids) - 1:
            start = max(0, a - (SW - 1))
            cache = draft.make_cache()
            for c in cache:
                c.offset = start
            ctx = mx.array(raw[start:a])[None].astype(mx.bfloat16)
            blk0 = mx.array([[int(ids[a])] + [MASK] * (BL - 1)])
            p0 = mx.argmax(draft.compute_logits(draft.hidden_states(blk0, ctx, cache, 1)), axis=-1)[0]   # positions 1..31
            p0 = np.array(p0).tolist()
            truth = ids[a + 1:a + BL].tolist()
            m = [int(x == y) for x, y in zip(p0, truth)]
            nat += np.array(m)
            k = 1
            while k <= 31 and m[k - 1]:
                k += 1
            kept.append(k)
            hv = []                                         # (NewIdeas 02:25) the per-anchor TF hit vector, j = 1..31
            for j in range(1, BL):                          # teacher forcing: true tokens at 0..j-1, MASK at j..31
                blk = mx.array([ids[a:a + j].tolist() + [MASK] * (BL - j)])
                h = draft.hidden_states(blk, empty, cache, j)[:, :1]
                pj = int(mx.argmax(draft.compute_logits(h), axis=-1)[0, 0].item())
                hv.append(int(pj == int(ids[a + j])))
                tf[j - 1] += hv[-1]
                if j == 1:                                  # self-check: j = 1 is the native block through the reused cache
                    selfchk[0] += 1; selfchk[1] += int(pj == p0[0])
            HV.setdefault(name, []).append(hv); NV.setdefault(name, []).append(m); AN.setdefault(name, []).append(f"{os.path.basename(f)}:{a}")
            n_anch += 1
            a += BL
            if HEAD and n_anch >= HEAD:                     # (Sun 02:4x) head test: the first HEAD anchors, then the check
                print(f"[s1tf] HEAD {HEAD}: self-check j=1 == native {selfchk[1]} / {selfchk[0]}; TF hits {HV[name]}; native hits "
                      f"{NV[name]}; kept {kept}", flush=True)
                sys.exit(0 if selfchk[0] and selfchk[1] == selfchk[0] else 3)
    band = lambda v: {f"{lo}-{hi}" if hi > lo else f"{lo}": round(100.0 * v[lo:hi + 1].sum() / (n_anch * (hi - lo + 1)), 2) for lo, hi in BANDS}
    H = np.array(HV[name], np.int64)                        # [anchors, 31]
    lead = np.array([next((i for i, x in enumerate(r) if not x), 31) for r in H])   # TF leading run of hits
    elig = np.stack([lead >= d for d in range(31)], 1)      # depth d counts when the TF hits at depths < d are all True
    hitc = H.astype(bool) & elig
    strict = {f"{lo}-{hi}" if hi > lo else f"{lo}": (round(100.0 * hitc[:, lo:hi + 1].sum() / elig[:, lo:hi + 1].sum(), 2)
              if elig[:, lo:hi + 1].sum() else None) for lo, hi in BANDS}
    strict_n = {f"{lo}-{hi}" if hi > lo else f"{lo}": int(elig[:, lo:hi + 1].sum()) for lo, hi in BANDS}
    res[name] = dict(files=len(files), anchors=n_anch, strict=strict, strict_n=strict_n, tf_kept32=round(float(1 + lead.mean()), 3),
                     tf_bands_info=band(tf), native_bands_info=band(nat), kept32=round(float(np.mean(kept)), 3),
                     tf_depth=[round(100.0 * x / n_anch, 2) for x in tf])
    print(f"[s1tf] {name}: {len(files)} files, {n_anch} anchors | STRICT {strict} (n {strict_n}) | TF kept/32 {res[name]['tf_kept32']} | "
          f"native kept/32 {res[name]['kept32']} | TF bands (info) {res[name]['tf_bands_info']}", flush=True)
np.savez(os.path.join(OUT, "s1tf_hits.npz"), **{f"{n}_tf": np.array(v, np.int8) for n, v in HV.items()},
         **{f"{n}_native": np.array(v, np.int8) for n, v in NV.items()}, **{f"{n}_anchors": np.array(v) for n, v in AN.items()})
he, mb = res["humaneval"]["strict"], res["mbpp"]["strict"]
d0 = res["humaneval"]["tf_depth"][0]
v = ("NO DATA" if he["23-30"] is None or mb["23-30"] is None else
     "PASS" if he["23-30"] >= 95.0 and mb["23-30"] >= 85.0 else "BETWEEN" if he["23-30"] >= 90.0 else "KILL")
res["verdict"] = v; res["selfcheck_j1_equal_native"] = selfchk
json.dump(res, open(os.path.join(OUT, "s1tf.json"), "w"), indent=1)
txt = (f"S1 DF-TF (block 32, DFlash2 l120_ft as the 12 GB engine runs it, offline on the BANK; STRICT line per NewIdeas 02:25): "
       f"self-check j=1 == native {selfchk[1]} / {selfchk[0]}\n" + "\n".join(
       f"{n}: {r['anchors']} anchors | STRICT conditional {r['strict']} (eligible {r['strict_n']}) | TF kept/32 {r['tf_kept32']} | "
       f"native kept/32 {r['kept32']} (L1-20 ~10.5; t30b chain 13.8 HE+) | bands, information only: TF {r['tf_bands_info']}, native "
       f"{r['native_bands_info']}" for n, r in res.items() if isinstance(r, dict) and "strict" in r)
       + f"\nLINE (strict): HE+ 23-30 {he['23-30']}, MBPP 23-30 {mb['23-30']} -> {v} (PASS >= 95 and >= 85; BETWEEN HE+ 90-95; KILL < 90)\n")
open(os.path.join(OUT, "s1tf.txt"), "w").write(txt)
print(txt, flush=True)
