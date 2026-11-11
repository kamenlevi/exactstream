"""R-11b (Leap, pre-registered Tue 2026-10-27 20:10 EET in ~/Leap/TABLE_350.md 'R-11b'): R-11 found S - O = 1.63 ms
against 5.75 ms of direct CPU spans per cycle, so the evals themselves may be CPU-bound (MLX's C++ encoding). This is
R-11's S cycle with the main thread's CPU time measured inside each eval (draft, check): eval CPU / eval wall near
100 % = the stage is bound by encoding, not by the GPU. Arms S and O only, 5 rounds x 8 cycles.
R-11 (the source of this copy):
the GPU's idle time per draft -> check -> accept -> commit cycle at 16 rows, on CA's resident 12-layer slice
(ca_anat.py, imported read-only) plus the DFlash2 drafter (L1-20 weights, block 16; CA's draft path).
Arms do the same work per cycle (one draft, one check, one commit replay, keep fixed at 10):
  S   the engine's order: draft built, its tokens read (sync), PLE staging + eager check build, acceptance read (sync),
      commit built lazily (its replay lands in the next check)
  SB  S with the cheap exact variants: the check built async every 4 layers (CA's M / b5's B) and the commit replay
      submitted on its own right after the acceptance read (A7: +12.9 % on the 35B loop)
  G1  acceptance on the GPU (oracle keep): check, commit and the next draft go out as one submission; one sync per
      cycle (the draft tokens, which PLE staging needs on the CPU)
  O   no sync at all: the GPU-busy bound (any sync hidden inside the engine's calls shows up in its PLE time)
Every check reads the text's own tokens (same routing in every arm). The drafter's context features are a slice of
the check's logits: a real GPU dependency, values irrelevant for timing.
S's idle, directly: the CPU spans between a read returning and the next submission (draft build, PLE + check build,
accept + commit build), cross-checked by S - O. 48 layers: drafter and PLE parts once, per-layer builds x4.
Lines (QUEUE R-11): lever if the 48-layer idle >= 8 ms per cycle; KILL < 3 ms.
Usage: [BigMoE venv] r11_screen.py    (writes data/r11_screen.txt)"""
import glob, os, sys, time
sys.argv = [sys.argv[0], "r11"]                          # ca_anat's MODE: load only, no run
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
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
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_third_party"))
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
        c1 = time.thread_time()
        dl = d.tolist()                                  # sync 1: the draft tokens
        t2, c2 = time.perf_counter(), time.thread_time()
        lg, snap, ple = check("M" if cheap else "A")
        t3 = time.perf_counter()
        c3 = time.thread_time()
        g = mx.argmax(lg[0], axis=-1).tolist()           # sync 2: the acceptance read
        t4, c4 = time.perf_counter(), time.thread_time()
        a = 0
        while a < BS - 1 and dl[0][a] == g[a]:
            a += 1
        ok = commit(snap)
        if cheap:
            mx.async_eval(*target_states())              # the commit replay goes out on its own (A7)
        st["tok"], st["ctx"] = g[K - 1], feats(lg)
        t5 = time.perf_counter()
        rec["cyc"].append(dict(bd=t1 - t0, ed=t2 - t1, bc=t3 - t2, ple=ple, ec=t4 - t3, bm=t5 - t4, wall=t5 - t0,
                               cd=c2 - c1, cc=c4 - c3))
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

ARMS = ["S", "O"]
def run_block(arm, st, n):
    rec = dict(cyc=[], ok=[], paused=os.path.exists(PAUSE))
    while os.path.exists(PAUSE):                         # quiet mode / heat pause: wait at a block boundary
        time.sleep(5)
    if arm in ("S", "SB"):
        wall = block_S(st, n, arm == "SB", rec)
    else:
        wall = block_P(st, n, arm == "G1", rec)
    return wall, rec

# first state: one engine cycle's worth (a check, its features and correction token)
lg, snap, _ = check("A")
g0 = mx.argmax(lg[0], axis=-1).tolist()
commit(snap)
st = dict(tok=g0[K - 1], ctx=feats(lg))
mx.eval(st["ctx"], *target_states())
st["tok_arr"] = mx.array([[st["tok"]]])

for arm in ARMS:                                         # warm-up, discarded
    run_block(arm, st, 3)
ROUNDS, N = 5, 8
W = {a: [] for a in ARMS}
C = {a: [] for a in ARMS}
OK, dropped = [], 0
t_run = time.time()
r = 0
while r < ROUNDS + dropped and r < ROUNDS + 2:
    order = ARMS[r % 4:] + ARMS[:r % 4]
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
say(f"  {len(W['S'])} rounds x {N} cycles per arm kept ({dropped} dropped), {time.time() - t_run:.0f} s; "
    f"commits landed {all(OK)}; context now {ST['pos']} tokens, drafter cache {dcache[0].offset}")

# ---------------------------------------------------------------- report
m = {a: med(W[a]) for a in ARMS}
say("  per-cycle wall (median of rounds; 12 layers + the drafter): " +
    ", ".join(f"{a} {m[a]:.2f} ms [min {min(W[a]):.2f}]" for a in ARMS))
pair = lambda a, b: med([x - y for x, y in zip(W[a], W[b])])
say(f"  paired: S - O {pair('S', 'O'):.2f} ms per cycle")
c = {k: 1e3 * med([x[k] for x in C["S"]]) for k in ("bd", "ed", "bc", "ple", "ec", "bm", "wall")}
idle12 = 1e3 * med([x["bd"] + x["bc"] + x["bm"] for x in C["S"]])
say(f"  S components (ms, medians): draft build {c['bd']:.2f}, draft eval + read {c['ed']:.2f}, PLE + check build "
    f"{c['bc']:.2f} (PLE {c['ple']:.2f}), check eval + read {c['ec']:.2f}, accept + commit build {c['bm']:.2f}; "
    f"cycle {c['wall']:.2f}")
say("  PLE staging per check (ms, medians): " + ", ".join(f"{a} {1e3 * med([x['ple'] for x in C[a]]):.2f}"
                                                         for a in ARMS) + "  (O >> S: a hidden sync in PLE staging)")
gap1 = c["bd"] + 4 * c["bm"]
gap2 = c["ple"] + 4 * (c["bc"] - c["ple"])
idle48 = gap1 + gap2
say(f"  S idle, direct (CPU spans with nothing queued): {idle12:.2f} ms per cycle at 12 layers (S - O measured "
    f"{pair('S', 'O'):.2f})")
say(f"  48 layers: idle ~{idle48:.2f} ms per cycle = after the acceptance read {gap1:.2f} (draft build + 4 x commit "
    f"build: a GPU-side accept removes it) + after the draft read {gap2:.2f} (PLE + 4 x check build: needs the draft "
    f"tokens on the CPU while PLE stages on the CPU)")
cd, cc = (1e3 * med([x[k] for x in C["S"]]) for k in ("cd", "cc"))
say(f"  draft eval: {c['ed']:.2f} ms wall, {cd:.2f} ms of main-thread CPU ({cd / c['ed']:.0%}); check eval: "
    f"{c['ec']:.2f} ms wall, {cc:.2f} ms CPU ({cc / c['ec']:.0%})")
say(f"R-11b: GPU-side busy per cycle at most ~{c['ed'] + c['ec'] - 0:.2f} ms; the CPU path (builds + encoding) "
    f"~{c['bd'] + c['bc'] + c['bm'] + cd + cc:.2f} ms of the {c['wall']:.2f} ms cycle")
for name, cpu, wall in (("draft", cd, c["ed"]), ("check", cc, c["ec"])):
    say(f"  {name} eval: {'ENCODING-BOUND' if cpu >= 0.8 * wall else 'not encoding-bound'} "
        f"(CPU {cpu / wall:.0%} of its wall; line 80 %)")

# ---------------------------------------------------------------- dispatch counts (one untimed S cycle, graphs exported)
import collections, re
VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided",
         "Flatten", "Unflatten", "StopGradient", "Depends", "Split"}
def count_dot(path):
    cnt = collections.Counter()
    for line in open(path):
        m = re.search(r'label\s*=\s*"([A-Z][A-Za-z0-9_]*)', line)
        if m:
            cnt[m.group(1)] += 1
    return cnt
NK = {}
try:
    d = dbuild(st["tok"], st["ctx"])
    mx.export_to_dot(os.path.join(D, "r11b_draft.dot"), d)
    dl = d.tolist()
    lg, snap, _ = check("A")
    mx.export_to_dot(os.path.join(D, "r11b_check.dot"), lg, *target_states())
    g = mx.argmax(lg[0], axis=-1).tolist()
    commit(snap)
    st["tok"], st["ctx"] = g[K - 1], feats(lg)
    mx.eval(st["ctx"], *target_states())
    for name in ("draft", "check"):
        cnt = count_dot(os.path.join(D, f"r11b_{name}.dot"))
        NK[name] = sum(v for k_, v in cnt.items() if k_ not in VIEWS)
        say(f"  {name} graph: ~{NK[name]} kernel-launching primitives ({sum(cnt.values())} nodes; top: " +
            ", ".join(f"{k_} {v}" for k_, v in cnt.most_common(8)) + ")")
    for name, cpu, wall in (("draft", cd, c["ed"]), ("check", cc, c["ec"])):
        n_ = max(NK[name], 1)
        say(f"  {name}: {1e3 * cpu / n_:.1f} us of eval CPU per primitive; an indirect command buffer replayed per cycle "
            f"(~1.5 us per dispatch, Leap's Sun 05:50 floor) ~{n_ * 1.5e-3:.2f} ms; encoding on 4 threads ~{cpu / 4:.2f} ms "
            f"(from {cpu:.2f}); x4 at 48 layers for the check")
except Exception as exc:
    say(f"  dispatch count failed: {str(exc)[:300]}")

# ---------------------------------------------------------------- MoonShots' R-24: per-row routing of real drafted checks
ROUTE, ROUTE_ON, ORIG_ARGP = [], [False], mx.argpartition
def argp(a, kth, axis=-1, **kw):
    out = ORIG_ARGP(a, kth=kth, axis=axis, **kw)
    if ROUTE_ON[0] and a.shape[-1] == 512 and kth == -10:
        ROUTE.append(out)
    return out
mx.argpartition = argp
try:
    zp = sorted(glob.glob(os.path.join(D, "dfl2d_code", "*.npz")))[0]
    z = np.load(zp)
    f, sid, off, gen = z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])
    rc = model.make_cache()
    pre = [int(t) for t in sid[:gen - off]]
    for a0 in range(0, len(pre), 1024):
        lg0 = model(mx.array([pre[a0:a0 + 1024]]), cache=rc)
        mx.eval(lg0, CA.states(rc))
    dc = draft.make_cache()
    start = max(off, gen - (draft.config.sliding_window - 1))
    for c_ in dc:
        c_.offset = start
    fed, a, recs = start, gen, []
    while len(recs) < 20 and a - off + 16 < len(sid):
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(sid[a - off])] + [MASK] * (BS - 1)])
        dr = np.array(select(blk, draft.hidden_states(blk, ctx, dc, 1))[0]).tolist()
        fed = a
        true = sid[a - off + 1:a - off + BS]
        k = 0
        while k < BS - 1 and dr[k] == true[k]:
            k += 1
        if (trim := dc[0].offset - a) > 0:
            DF._trim_recent_cache(dc, trim)
        x = mx.array([[int(sid[a - off])] + dr[:BS - 1]])
        snap = CS.snapshot_untrimmable_cache_lazy(rc)
        ROUTE.clear(); ROUTE_ON[0] = True
        lg = CA.forward("A", x, rc)
        mx.argmax(lg[0], axis=-1).tolist()
        ROUTE_ON[0] = False
        rt = np.array(mx.stack([r_[..., -10:].reshape(R, 10) for r_ in ROUTE])).astype(np.int16)
        model.commit_verified_window(rc, snap.states, keep_tokens=k + 1, verified_tokens=R)
        mx.eval(CA.states(rc))
        recs.append((k, rt))
        a = a + k + 1
    ks = np.array([k for k, _ in recs])
    routes = np.stack([rt for _, rt in recs])                # [checks, MoE layers, 16 rows, top-10]
    np.savez(os.path.join(D, "r24_routes.npz"), routes=routes, accepted=ks, layers=np.arange(routes.shape[1]),
             sample=os.path.basename(zp), context=gen - off)
    u_all = [len(set(routes[i, l].ravel())) for i in range(len(recs)) for l in range(routes.shape[1])]
    u_kept = [len(set(routes[i, l, :ks[i] + 1].ravel())) for i in range(len(recs)) for l in range(routes.shape[1])]
    say(f"  R-24 dump (MoonShots): {len(recs)} real drafted checks ({os.path.basename(zp)}, {gen - off} context tokens, "
        f"12-layer slice = {routes.shape[1]} MoE layers); accepted drafts {ks.mean():.2f} per check (keep {ks.mean() + 1:.2f}); "
        f"expert union per layer: all 16 rows {np.mean(u_all):.1f}, kept rows only {np.mean(u_kept):.1f} "
        f"({1 - np.mean(u_kept) / np.mean(u_all):.0%} of the union comes only from rejected rows); data/r24_routes.npz")
except Exception as exc:
    say(f"  R-24 dump failed: {str(exc)[:300]}")
mx.argpartition = ORIG_ARGP
open(os.path.join(D, "r11b_screen.txt"), "w").write("\n".join(lines) + "\n")
raise SystemExit(0 if all(OK) else 1)
