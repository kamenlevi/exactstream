"""FRS-1 (Leap, Mon 2026-11-02; for Fable mission 18 item 1): a frequency-ranked drafter vocabulary (FR-Spec, Zhao et al.
2025, arXiv 2502.14856: the draft head scores only the most frequent tokens; the target still checks every token, so
outputs stay exact and only kept per check can move). On the Neural Engine the head is the biggest piece of a batched
call (int8 8.4 ms for 45 rows; int4 falls back to the CPU), so a top-N head is the head lever: 32K rows = 1/7.6 of it.
1. Ranking: counts of the 125B's generated tokens in TrainLab's captures (d02a without its tf_gsm_* / tf_code_HumanEval_* /
   agent* files, which are the replay's test samples; d15_multi; e2_x4 = MBPP), ties by token id. Saved to data/frspec_rank.npy.
2. Coverage of the replay sets' generated tokens by the top N.
3. Kept per check on r84's replay (stock MLX DFlash2 L1-20, temperature 0) with the head masked to the top N (= a top-N
   head: the top-16 candidates and the chain only see those tokens), N = full / 65,536 / 32,768 / 16,384; code 24
   samples, math the first 30 of 60.
Usage: [BigMoE venv] frspec_walk.py -> ~/Leap/data/frspec_walk.txt"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
TL = os.path.expanduser("~/TrainLab/data")
V = 248320
OUT = os.path.join(D, "frspec_walk.txt")
NS = (65536, 32768, 16384)


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


say(f"FRS-1 {time.strftime('%a %H:%M')}")
cnt = np.zeros(V, np.int64)
nf = ntok = 0
for f in sorted(glob.glob(f"{TL}/d02a/*.npz")):
    b = os.path.basename(f)
    if b.startswith("tf_gsm_") or b.startswith("tf_code_HumanEval_") or b.startswith("agent"):
        continue
    z = np.load(f)
    g = z["ids"][int(z["gen"]) - int(z["off"]):]
    cnt += np.bincount(g, minlength=V)
    nf += 1
    ntok += g.size
for d in ("d15_multi", "e2_x4"):
    for f in sorted(glob.glob(f"{TL}/{d}/*.npz")):
        z = np.load(f)
        g = z["ids"][z["asst"].astype(bool)]
        cnt += np.bincount(g, minlength=V)
        nf += 1
        ntok += g.size
rank = np.lexsort((np.arange(V), -cnt))
np.save(os.path.join(D, "frspec_rank.npy"), rank)
say(f"ranking from {nf} capture files, {ntok} generated tokens, {int((cnt > 0).sum())} distinct")
TEST = {}
SETS = os.environ.get("FRS_SETS", "code,math").split(",")
SRC = {"code": (os.path.join(D, "dfl2d_code"), "*.npz", 24), "math": (os.path.join(D, "dfl2d_math"), "*.npz", 30),
       "agent": (os.path.join(TL, "d02a"), "agent*.npz", 12)}
for name in SETS:
    folder, pat, nmax = SRC[name]
    smp = []
    for p in sorted(glob.glob(os.path.join(folder, pat)))[:nmax]:
        z = np.load(p)
        smp.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"], int(z["off"]), int(z["gen"])))
    TEST[name] = smp
    gen = np.concatenate([s[1][s[3] - s[2]:] for s in smp])
    say(f"  {name}: {gen.size} generated tokens; coverage " + ", ".join(
        f"top {n}: {100 * np.isin(gen, rank[:n]).mean():.2f} %" for n in NS))

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
from ane_probe import mlx_drafter
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party"))
import dflash_model_mlx as DF
d, tgt = mlx_drafter()
MASK = int(d.config.mask_token_id)
NEG = {}
for n in NS:
    m = np.full(V, -1e9, np.float32)
    m[rank[:n]] = 0
    NEG[n] = mx.array(m).astype(mx.bfloat16)
LIM = [None]
SESS = [None]                                  # mx bf16 (V,) 0 / -1e9: tokens already in the sample's context
orig = d.compute_logits


def compute_logits(hidden):
    lg = orig(hidden)
    if LIM[0] is None:
        return lg
    return lg + (NEG[LIM[0]] if SESS[0] is None else mx.maximum(NEG[LIM[0]], SESS[0]))
d.compute_logits = compute_logits


def walk(sample):
    f, ids, off, gen = sample
    cache = d.make_cache()
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, out, n = start, gen, [], len(ids)
    seen = np.full(V, -1e9, np.float32)
    while a < off + n - 1:
        if USE_SESS[0]:
            seen[ids[:a - off + 1]] = 0                # the prompt window + every token up to the anchor
            SESS[0] = mx.array(seen).astype(mx.bfloat16)
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * 15])
        dr = np.array(d.propose(blk, ctx, cache, 0.0, 1)[0][0]).tolist()
        fed = a
        true = ids[a - off + 1:a - off + 16]
        k = 0
        while k < len(true) and dr[k] == true[k]:
            k += 1
        out.append(k + 1)
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + k + 1
    return out


WINDOWS = [int(x) for x in os.environ.get("FRS_WINDOW", "2048").split(",")]


def set_window(w):
    """The drafter's attention window (trained: 2048): cache size and the sliding mask of every layer."""
    d.config.sliding_window = w
    for l_ in d.layers:
        l_.self_attn.sliding_window = w


USE_SESS = [os.environ.get("FRS_SESSION") == "1"]
ARMS = [int(x) for x in os.environ.get("FRS_ARMS", "0," + ",".join(map(str, NS))).split(",")]
say(f"arms {ARMS} (0 = full head), session extension {'ON (top N + tokens already in the context)' if USE_SESS[0] else 'off'}")
for name, smp in TEST.items():
    base = None
    for w in WINDOWS:
        set_window(w)
        for n in [None if x == 0 else x for x in ARMS]:
            LIM[0] = n
            t0 = time.time()
            kept = [k for s in smp for k in walk(s)]
            km = float(np.mean(kept))
            base = base or km
            say(f"  {name} window {w} head {'full' if n is None else f'top {n}'}{' + session' if USE_SESS[0] and n else ''}: "
                f"kept per check {km:.3f} ({len(kept)} checks, {time.time() - t0:.0f} s) -> {100 * (km / base - 1):+.2f} %")
