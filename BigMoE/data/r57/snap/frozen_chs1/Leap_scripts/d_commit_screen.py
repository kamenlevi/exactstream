"""D-COMMIT, mission 29's design D screen (Leap, written Mon 2026-11-02 23:0x for Tue after 15:00, the one big job by day;
pre-registered in ~/Leap/TABLE_350.md 'D-COMMIT'; ~/FableGod/DRAFT_HIDDEN.md design D): hide the commit replay behind the
next draft by putting the draft on a second MLX stream. R-11's harness (r11_screen.py: CA's resident 12-layer slice +
the DFlash2 L1-20 drafter, 16-row checks, keep fixed at 10), arms:
  SB  the engine's cycle (draft, read, check, acceptance read, commit replay submitted on its own)
  G1  acceptance on the GPU: one sync per cycle (the draft tokens)
  D   G1 with the next draft built and submitted on a second GPU stream right after the check, so the commit replay
      (default stream) and the draft run concurrently
Exactness: D changes only which stream runs which kernels; checked directly by drafting one block from the same state on
the default stream and on the second stream (tokens equal) before the timing.
Lines: PASS D >= 0.6 ms per cycle faster than G1 at 12 layers (~2.4 ms at 48: the commit replay's ~3 ms mostly hidden);
KILL < 0.2 ms. Scale to 48 layers: x4 for the per-layer commit part.
Usage: [BigMoE venv] d_commit_screen.py    (writes data/d_commit_screen.txt; ~20.5 GB active: one big job)"""
import glob, os, sys, time
sys.argv = [sys.argv[0], "r11"]                          # ca_anat's MODE: load only, no run
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
import ca_anat as CA
import numpy as np
import mlx.core as mx
import mlx.nn as nn

Q, CS, model, D = CA.Q, CA.CS, CA.model, CA.D
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))

# ---------------------------------------------------------------- the drafter, as CA.draft_times
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_third_party"))
import dflash_model_mlx as DF
class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)
w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
tgt = Target()
tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales",
                  "biases")] + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales",
                  "biases")])
del w
draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
draft.bind(tgt)
draft.load_weights(os.path.join(D, "l120_ft.safetensors"), strict=False)
mx.eval(draft.parameters(), tgt.parameters())
MASK = int(draft.config.mask_token_id)
F = int(draft.fc.weight.shape[-1])                      # context feature width (target layers x hidden)

def select(block, hidden):
    logits = draft.compute_logits(hidden)
    sel = draft.candidate_selector
    cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
    unary = mx.take_along_axis(logits, cand, axis=-1)
    hp = sel.hidden_projection(hidden)
    pred, path = block[:, 0], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)

R, BS = 16, 16
K = CA.KEEP[R]                                           # 10 = accepted + 1 per check (L1-20)
MASKS = mx.array([[MASK] * (BS - 1)])
say(f"R-11: 12-layer slice ({CA.KINDS}) + DFlash2 drafter (L1-20, block {BS}); {R}-row checks, keep fixed at {K}; "
    f"context features {F} wide; {mx.get_active_memory() / 1e9:.1f} GB active")

cache = CA.prefill(1024)
mx.synchronize()
dcache = draft.make_cache()
ST = dict(pos=1024, dpos=0)

def dbuild(tok, ctx):
    """one block's draft graph (lazy); tok: int or a [1, 1] array; the block rows are trimmed back off (lazy)"""
    first = mx.array([[tok]]) if isinstance(tok, int) else tok
    blk = mx.concatenate([first, MASKS], axis=1)
    d = select(blk, draft.hidden_states(blk, ctx, dcache, 1))
    ST["dpos"] += ctx.shape[1]
    if (trim := dcache[0].offset - ST["dpos"]) > 0:
        DF._trim_recent_cache(dcache, trim)
    return d

def check(variant):
    x = mx.array([CA.ids[ST["pos"]:ST["pos"] + R]])
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    p0 = Q._PLE_STAGE_SECONDS[0]
    lg = CA.forward(variant, x, cache)                   # "A" = the engine's eager check, "M" = async every 4 layers
    return lg, snap, Q._PLE_STAGE_SECONDS[0] - p0

def commit(snap):
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=K, verified_tokens=R)
    ST["pos"] += K
    return bool(ok)

def feats(lg):
    return lg[:, :K, :F].astype(mx.bfloat16)

def target_states():
    return CA.leaves(CA.states(cache))

def block_S(st, n, cheap, rec):
    """the engine's cycle (cheap: SB)"""
    tA = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        d = dbuild(st["tok"], st["ctx"])
        t1 = time.perf_counter()
        dl = d.tolist()                                  # sync 1: the draft tokens
        t2 = time.perf_counter()
        lg, snap, ple = check("M" if cheap else "A")
        t3 = time.perf_counter()
        g = mx.argmax(lg[0], axis=-1).tolist()           # sync 2: the acceptance read
        t4 = time.perf_counter()
        a = 0
        while a < BS - 1 and dl[0][a] == g[a]:
            a += 1
        ok = commit(snap)
        if cheap:
            mx.async_eval(*target_states())              # the commit replay goes out on its own (A7)
        st["tok"], st["ctx"] = g[K - 1], feats(lg)
        t5 = time.perf_counter()
        rec["cyc"].append(dict(bd=t1 - t0, ed=t2 - t1, bc=t3 - t2, ple=ple, ec=t4 - t3, bm=t5 - t4, wall=t5 - t0))
        rec["ok"].append(ok); rec["paused"] |= os.path.exists(PAUSE)
    mx.eval(st["ctx"], *target_states())
    mx.synchronize()
    st["tok_arr"] = mx.array([[st["tok"]]])
    return (time.perf_counter() - tA) / n

def block_P(st, n, sync, rec):
    """G1 (sync: one read per cycle) or O (no read)"""
    d = dbuild(st["tok_arr"], st["ctx"])
    mx.eval(d)
    mx.synchronize()
    acc = None
    tA = time.perf_counter()
    for _ in range(n):
        if sync:
            dl = d.tolist()                              # the one sync: draft tokens (+ the previous acceptance)
            if acc is not None:
                acc.item()
        lg, snap, ple = check("M")
        g = mx.argmax(lg[0], axis=-1)
        acc = mx.sum(mx.cumprod((g[:BS - 1] == d[0]).astype(mx.int32)))   # acceptance on the GPU
        ok = commit(snap)
        d = dbuild(g[K - 1:K][None], feats(lg))
        mx.async_eval(d, acc, *target_states())
        rec["cyc"].append(dict(ple=ple)); rec["ok"].append(ok); rec["paused"] |= os.path.exists(PAUSE)
    d.tolist()
    mx.synchronize()
    wall = (time.perf_counter() - tA) / n
    st["tok_arr"], st["ctx"] = g[K - 1:K][None], feats(lg)
    mx.eval(st["tok_arr"], st["ctx"])
    st["tok"] = int(st["tok_arr"].item())
    return wall

S2 = mx.new_stream(mx.gpu)

def block_D(st, n, rec):
    """G1 with the next draft on a second GPU stream (design D): the commit replay and the draft run concurrently"""
    d = dbuild(st["tok_arr"], st["ctx"])
    mx.eval(d)
    mx.synchronize()
    acc = None
    tA = time.perf_counter()
    for _ in range(n):
        dl = d.tolist()                                  # the one sync: draft tokens (+ the previous acceptance)
        if acc is not None:
            acc.item()
        lg, snap, ple = check("M")
        g = mx.argmax(lg[0], axis=-1)
        acc = mx.sum(mx.cumprod((g[:BS - 1] == d[0]).astype(mx.int32)))
        ok = commit(snap)
        mx.async_eval(acc, *target_states())             # the commit replay, default stream
        with mx.stream(S2):
            d = dbuild(g[K - 1:K][None], feats(lg))      # the next draft, second stream
            mx.async_eval(d)
        rec["cyc"].append(dict(ple=ple)); rec["ok"].append(ok); rec["paused"] |= os.path.exists(PAUSE)
    d.tolist()
    mx.synchronize()
    wall = (time.perf_counter() - tA) / n
    st["tok_arr"], st["ctx"] = g[K - 1:K][None], feats(lg)
    mx.eval(st["tok_arr"], st["ctx"])
    st["tok"] = int(st["tok_arr"].item())
    return wall

ARMS = ["SB", "G1", "D"]
def run_block(arm, st, n):
    rec = dict(cyc=[], ok=[], paused=os.path.exists(PAUSE))
    while os.path.exists(PAUSE):                         # quiet mode / heat pause: wait at a block boundary
        time.sleep(5)
    if arm == "SB":
        wall = block_S(st, n, True, rec)
    elif arm == "G1":
        wall = block_P(st, n, True, rec)
    else:
        wall = block_D(st, n, rec)
    return wall, rec

# first state: one engine cycle's worth (a check, its features and correction token)
lg, snap, _ = check("A")
g0 = mx.argmax(lg[0], axis=-1).tolist()
commit(snap)
st = dict(tok=g0[K - 1], ctx=feats(lg))
mx.eval(st["ctx"], *target_states())
st["tok_arr"] = mx.array([[st["tok"]]])

# exactness: the same draft from the same state on the default stream and on the second stream
dsnap = [(c.offset,) for c in dcache]
d_def = dbuild(st["tok_arr"], st["ctx"]); mx.eval(d_def)
DF._trim_recent_cache(dcache, dcache[0].offset - dsnap[0][0]); ST["dpos"] -= st["ctx"].shape[1]
with mx.stream(S2):
    d_s2 = dbuild(st["tok_arr"], st["ctx"]); mx.eval(d_s2)
DF._trim_recent_cache(dcache, dcache[0].offset - dsnap[0][0]); ST["dpos"] -= st["ctx"].shape[1]
say(f"  exactness: the draft on the second stream == the default stream: {d_def.tolist() == d_s2.tolist()}")
for arm in ARMS:                                         # warm-up, discarded
    run_block(arm, st, 3)
ROUNDS, N = 5, 8
W = {a: [] for a in ARMS}
C = {a: [] for a in ARMS}
OK, dropped = [], 0
t_run = time.time()
r = 0
while r < ROUNDS + dropped and r < ROUNDS + 2:
    order = ARMS[r % 3:] + ARMS[:r % 3]
    got = {}
    for arm in order:
        got[arm] = run_block(arm, st, N)
    if any(rec["paused"] for _, rec in got.values()):
        dropped += 1
        say(f"  round {r}: a PAUSE came during it, dropped")
    else:
        for arm, (wall, rec) in got.items():
            W[arm].append(1e3 * wall); C[arm].extend(rec["cyc"]); OK.extend(rec["ok"])
    r += 1
say(f"  {len(W['SB'])} rounds x {N} cycles per arm kept ({dropped} dropped), {time.time() - t_run:.0f} s; "
    f"commits landed {all(OK)}; context now {ST['pos']} tokens, drafter cache {dcache[0].offset}")

# ---------------------------------------------------------------- report
m = {a: med(W[a]) for a in ARMS}
say("  per-cycle wall (median of rounds; 12 layers + the drafter): " +
    ", ".join(f"{a} {m[a]:.2f} ms [min {min(W[a]):.2f}]" for a in ARMS))
pair = lambda a, b: med([x - y for x, y in zip(W[a], W[b])])
gain = pair("G1", "D")
say(f"  paired: SB - G1 {pair('SB', 'G1'):.2f}, G1 - D {gain:.2f}, SB - D {pair('SB', 'D'):.2f} ms per cycle")
v = "PASS" if gain >= 0.6 else "KILL" if gain < 0.2 else "BETWEEN"
say(f"D-COMMIT verdict (PASS D >= 0.6 ms per cycle faster than G1 at 12 layers, KILL < 0.2): {v} ({gain:.2f} ms; "
    f"~{4 * gain:.1f} ms at 48 layers for the per-layer commit part)")
open(os.path.join(D, "d_commit_screen.txt"), "w").write("\n".join(lines) + "\n")
raise SystemExit(0 if all(OK) else 1)
