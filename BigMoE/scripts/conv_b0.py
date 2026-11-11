#!/usr/bin/env python3
"""B0 of the conversation cache (docs/CONV_CACHE_DESIGN.md): the real state sizes, and restore + suffix prefill vs a cold
prefill, on the real cache classes.

One QSA attention layer and one GDN layer of qwen4_exp (mtplx, the engine's model code), built from models/bare/config.json
with random weights (seed 0, bf16), each with its own cache from the classes make_cache() uses (QSACache, ArraysCache).
Inputs are a function of the token ids (a seeded table), so a prefix gives the same rows in every run. Light GPU (< 2 GB):
no expert, no model load. What it settles before B1 on the real engine:
  sizes   allocated and logical bytes per piece at 2,048 / 12,302 / 20,480 / 61,440 tokens (12,302 = the recorded Claude
          Code turn, Leap's agent/sessions_real_1110.jsonl);
  replay  the recorded shape: turn 1 (12,302 tokens, stable end 10,019) prefilled in 1,024-row chunks with checkpoints at
          the prompt-end edge (12,288) and the stable edge (9,216), then decode-style writes (16-row verify, rollback,
          commit 5) for 194 tokens; turn 2 (12,496 tokens: turn 1 + reply + new user) restored at 12,288 on the live
          cache; a new conversation (10,159 shared tokens) restored at 9,216. Each warm run is compared with a cold
          prefill of the same ids: every chunk output and every final state array, sha256 of the bytes;
  costs   checkpoint clone / restore / trim for 36 GDN layers at the real shapes, the DFlash2 taps tail, the hash chain.
usage: ~/BigMoE/.venv/bin/python scripts/conv_b0.py [out.json]"""
import hashlib, json, os, sys, time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache

from mtplx import cache_state as CS
from mtplx.models import qwen4_exp as Q

ROOT = os.path.expanduser("~/BigMoE")
CHUNK = 1024
cfg = json.load(open(os.path.join(ROOT, "models/bare/config.json")))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
RATIO = args.indexer_compress_ratio or 4
H = args.hidden_size
N_QSA = sum(1 for t in args.layer_types if t != "linear_attention")
N_GDN = len(args.layer_types) - N_QSA

mx.random.seed(0)
att = Q.Attention(args)
gdn = Q.GatedDeltaNet(args)
att.set_dtype(mx.bfloat16)
gdn.set_dtype(mx.bfloat16)
NE = 4099
E = (mx.random.normal((NE, H)) * 0.5).astype(mx.bfloat16)
mx.eval(att.parameters(), gdn.parameters(), E)


def x_of(ids):
    return E[mx.array(np.asarray(ids, dtype=np.int64) % NE)][None]


def leaf_bytes(x):
    if x.dtype == mx.bfloat16:
        x = x.view(mx.uint16)
    return np.asarray(x).tobytes()


def sha(x):
    return hashlib.sha256(leaf_bytes(x)).hexdigest()[:16]


def state_hashes(ca, cg):
    off = ca.offset
    nb = min(ca.pooled_len, off // RATIO)
    k, v, raw, pooled = ca.state
    out = {"kv.keys": sha(k), "kv.values": sha(v), "raw_keys": sha(raw), "pooled": sha(pooled),
           "pooled_f32": sha(ca.pooled_f32_view(nb)), "offset": off, "pooled_len": ca.pooled_len}
    for i, s in enumerate(cg.state):
        out[f"gdn[{i}]"] = None if s is None else sha(s)
    return out


def new_caches():
    return Q.QSACache(RATIO), ArraysCache(size=2)


def forward(ids, ca, cg):
    x = x_of(ids)
    ya, yg = att(x, ca), gdn(x, None, cg)
    mx.eval(ya, yg, [s for s in ca.state if s is not None], [s for s in cg.state if s is not None])
    return ya, yg


def prefill(ids, ca, cg, start=0, edges=()):
    """1,024-row chunks from `start` (a multiple of CHUNK), as the loop's prefill(); checkpoints at `edges`"""
    outs, ckpt = {}, {}
    for a in range(start, len(ids), CHUNK):
        ya, yg = forward(ids[a:a + CHUNK], ca, cg)
        outs[a] = (sha(ya), sha(yg))
        e = a + CHUNK
        if e in edges and e < len(ids):
            s = CS.snapshot_untrimmable_cache([ca, cg])      # eager clone of the GDN state; the KV stays live (trimmable)
            mx.eval([x for st in s.states if st is not None for x in st if x is not None])
            ckpt[e] = s
    return outs, ckpt


def restore(ca, cg, c, snap):
    ca.trim(ca.offset - c)
    CS.restore_cache([ca, cg], snap)                         # clone=True: the checkpoint stays intact for later restores


def decode(ca, cg, rng, n, win=16, acc=5):
    """decode-style writes past the prompt: verify `win` rows, roll back, commit `acc` (the engine's verify / rollback)"""
    done = []
    while len(done) < n:
        snap = CS.snapshot_untrimmable_cache_lazy([ca, cg])
        rows = rng.integers(0, 248320, win).tolist()
        forward(rows, ca, cg)
        CS.rollback_after_verify([ca, cg], snap, win)
        k = min(acc, n - len(done))
        forward(rows[:k], ca, cg)
        done += rows[:k]
    return done


def edge_le(n):
    return (n // CHUNK) * CHUNK


def compare(name, ids, warm_outs, warm_state, c):
    ca, cg = new_caches()
    cold_outs, _ = prefill(ids, ca, cg)
    cold_state = state_hashes(ca, cg)
    chunks = sorted(warm_outs)
    bad_chunks = [a for a in chunks if warm_outs[a] != cold_outs.get(a)]
    bad_state = [k for k in cold_state if cold_state[k] != warm_state.get(k)]
    ok = not bad_chunks and not bad_state
    return dict(case=name, N=len(ids), restored_at=c, prefilled=len(ids) - c, chunks_compared=len(chunks),
                chunks_differ=bad_chunks, states_compared=len(cold_state), states_differ=bad_state, PASS=ok)


def sizes(N):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 248320, N).tolist()
    ca, cg = new_caches()
    t0 = time.perf_counter()
    prefill(ids, ca, cg)
    dt = time.perf_counter() - t0
    off = ca.offset
    nb = min(ca.pooled_len, off // RATIO)
    kv_alloc = ca.kv.nbytes
    raw_alloc = ca.raw_keys.nbytes
    pooled_alloc = ca.pooled.nbytes
    mirror_alloc = 0 if ca.pooled_f32_t is None else ca.pooled_f32_t.nbytes
    k, v, raw, pooled = ca.state
    kv_log = k.nbytes + v.nbytes
    idx_log = raw.nbytes + pooled.nbytes + ca.pooled_f32_view(nb).nbytes
    gdn_b = sum(s.nbytes for s in cg.state if s is not None)
    return dict(N=N, prefill_s_1attn_1gdn=round(dt, 2),
                per_qsa_layer=dict(kv_alloc=kv_alloc, kv_logical=kv_log, kv_rows_alloc=int(ca.kv.keys.shape[2]),
                                   index_alloc=raw_alloc + pooled_alloc + mirror_alloc, index_logical=idx_log,
                                   raw_rows_alloc=int(ca.raw_keys.shape[1]), pooled_blocks_alloc=int(ca.pooled.shape[1])),
                per_gdn_layer=dict(bytes=gdn_b, shapes=[list(s.shape) + [str(s.dtype)] for s in cg.state if s is not None]),
                model=dict(kv_logical_B_per_token=round(N_QSA * kv_log / N, 1),
                           index_logical_B_per_token=round(N_QSA * idx_log / N, 1),
                           qsa_alloc_GB=round(N_QSA * (kv_alloc + raw_alloc + pooled_alloc + mirror_alloc) / 1e9, 3),
                           qsa_logical_GB=round(N_QSA * (kv_log + idx_log) / 1e9, 3),
                           gdn_MB_per_checkpoint=round(N_GDN * gdn_b / 1e6, 2)))


def costs():
    """checkpoint clone, restore and trim at the real shapes for all 36 GDN layers; the taps tail; the hash chain"""
    Hv, Kd, Vd = args.linear_num_value_heads, args.linear_key_head_dim, args.linear_value_head_dim
    conv_dim = 2 * args.linear_num_key_heads * Kd + Hv * Vd
    layers = []
    for _ in range(N_GDN):
        c = ArraysCache(size=2)
        c[0] = mx.random.normal((1, args.linear_conv_kernel_dim - 1, conv_dim)).astype(mx.bfloat16)
        c[1] = mx.random.normal((1, Hv, Vd, Kd)).astype(mx.float32)
        layers.append(c)
    mx.eval([c.state for c in layers])
    gdn_bytes = sum(s.nbytes for c in layers for s in c.state)
    res = {"gdn_checkpoint_MB": round(gdn_bytes / 1e6, 2)}
    tc, tr = [], []
    for _ in range(7):
        t0 = time.perf_counter()
        snap = CS.snapshot_untrimmable_cache(layers)
        mx.eval([x for st in snap.states for x in st])
        tc.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        CS.restore_cache(layers, snap)
        mx.eval([c.state for c in layers])
        tr.append(time.perf_counter() - t0)
    res["gdn_clone_ms_median"] = round(1e3 * sorted(tc)[3], 2)
    res["gdn_restore_ms_median"] = round(1e3 * sorted(tr)[3], 2)
    tap_dim = 5 * H                                          # TAPS 5/15/25/35/45, each the 4-way mean of the hc stream
    T = mx.random.normal((12302, tap_dim)).astype(mx.float16)
    mx.eval(T)
    t0 = time.perf_counter()
    tail = T[12288 - 2047:12288] + mx.zeros((), dtype=mx.float16)
    mx.eval(tail)
    res["taps_tail_MB"] = round(tail.nbytes / 1e6, 2)
    res["taps_tail_clone_ms"] = round(1e3 * (time.perf_counter() - t0), 2)
    res["taps_full_B_per_token_during_prefill"] = tap_dim * 2
    ids = np.random.default_rng(2).integers(0, 248320, 61440).astype(np.int32)
    t0 = time.perf_counter()
    h, chain = b"", []
    for a in range(0, len(ids) - CHUNK + 1, CHUNK):
        h = hashlib.sha256(h + ids[a:a + CHUNK].tobytes()).digest()
        chain.append(h)
    res["hash_chain_ms_61440"] = round(1e3 * (time.perf_counter() - t0), 3)
    return res


def replay():
    """the recorded Claude Code shape: turn 1, decode, turn 2 on the live cache, then a new conversation on the stable edge"""
    rng = np.random.default_rng(3)
    sys_end, n1, g = 10019, 12302, 5                         # g: the generation prompt's tokens (<|im_start|>assistant ...)
    ids1 = rng.integers(0, 248320, n1).tolist()
    e_prompt, e_stable = edge_le(n1 - g), edge_le(sys_end)
    ca, cg = new_caches()
    _, ck = prefill(ids1, ca, cg, edges={e_prompt, e_stable})
    reply = decode(ca, cg, rng, 150)                         # the reply; Leap's next render: prompt 1 + reply + user
    ids2 = ids1 + reply + rng.integers(0, 248320, 12496 - n1 - 150).tolist()
    out = []
    c = e_prompt                                             # the deepest kept edge <= the shared prefix (n1 - g)
    restore(ca, cg, c, ck[c])
    w_outs, ck2 = prefill(ids2, ca, cg, start=c, edges={edge_le(len(ids2) - g)})
    out.append(compare("turn 2 on the live cache, restored at the prompt-end edge", ids2, w_outs, state_hashes(ca, cg), c))
    decode(ca, cg, rng, 100)
    ids3 = ids1[:10159] + rng.integers(0, 248320, 12300 - 10159).tolist()   # a new conversation, same system + tools
    c = e_stable
    restore(ca, cg, c, ck[c])
    w_outs, _ = prefill(ids3, ca, cg, start=c)
    out.append(compare("new conversation on the same live cache, restored at the stable edge", ids3, w_outs,
                       state_hashes(ca, cg), c))
    ca2, cg2 = new_caches()                                  # control: the checkpoint restored into a fresh container
    prefill(ids1[:e_stable], ca2, cg2)
    snap = ck[e_stable]
    restore(ca2, cg2, e_stable, snap)
    w_outs, _ = prefill(ids3, ca2, cg2, start=e_stable)
    out.append(compare("control: stable-edge checkpoint restored again (it must be unchanged by the reuse above)", ids3,
                       w_outs, state_hashes(ca2, cg2), e_stable))
    ca3, cg3 = new_caches()                                  # negative control: the wrong checkpoint (stable edge's GDN
    prefill(ids1, ca3, cg3)                                  # state under the prompt-end edge's KV) must FAIL
    restore(ca3, cg3, e_prompt, ck[e_stable])
    w_outs, _ = prefill(ids2, ca3, cg3, start=e_prompt)
    r_ = compare("negative control: wrong checkpoint (must differ)", ids2, w_outs, state_hashes(ca3, cg3), e_prompt)
    r_["PASS"] = not r_["PASS"]                              # PASS here = the comparison caught it
    out.append(r_)
    return out


if __name__ == "__main__":
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data/conv/b0.json")
    t0 = time.time()
    R = {"what": "B0: conversation-cache state sizes + restore mechanics (1 QSA + 1 GDN layer, random weights)",
         "mlx": mx.__version__, "layers": {"qsa": N_QSA, "gdn": N_GDN}, "chunk": CHUNK}
    R["replay"] = replay()
    print(json.dumps(R["replay"], indent=1), flush=True)
    R["costs"] = costs()
    print(json.dumps(R["costs"], indent=1), flush=True)
    R["sizes"] = [sizes(n) for n in (2048, 12302, 20480, 61440)]
    print(json.dumps(R["sizes"], indent=1), flush=True)
    R["peak_GB"] = round(mx.get_peak_memory() / 1e9, 3)
    R["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(R, open(dst, "w"), indent=1)
    print("PASS" if all(r["PASS"] for r in R["replay"]) else "FAIL", "peak", R["peak_GB"], "GB, wall", R["wall_s"], "s")
