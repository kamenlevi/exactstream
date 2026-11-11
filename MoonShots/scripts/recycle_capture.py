"""M-06 RECYCLE capture + oracle (MoonShots Thu 17:3x; exact; GPU in one quiet lock): does the 125B's own verify output
after DFlash2's first miss already hold the next block?
Phase A: MTP Head's D-02b walk (functions copied from ~/TrainLab/sticky/d02b_walk.py, read-only: DFlash2 L1-20 bf16,
  block 16, the official 1-step loop, R-06's exact two-stage top-16 head) over Leap's dfl2d_code (24 HumanEval+) and
  dfl2d_math (60 GSM8K) samples, now keeping every check's 15 drafts; per check (accepted, checkable, anchor) as stored.
Phase B: the Eval suite's 'ref' variant (exact 4-bit, sorted routing; read-only import, no BIGMOE_* flags) teacher-forced
  over branch sequences ids[:anchor + 1] + the check's 15 drafts, for N sampled checks per domain with a first miss and a
  next check; per row j = 0..15 (position anchor + j) the argmax and its probability = the 16 verify rows.
Oracle (NewIdeas' lines, M-06 row): a check with acc accepted drafts has its first miss at draft acc + 1; kept_r = the
  matching run of rows acc + 1 .. 15 (the recycled tail, predicting positions anchor + acc + 2 ..) against the 125B's
  tokens; kept_f = the next check's accepted count (DFlash2's fresh draft from anchor + acc + 1, the walk's own).
  gain = E[max(kept_r, kept_f)] - E[kept_f].
KILL: < 0.3 tokens per check on both code and math. PASS: >= +0.8 on either (then a selection rule, e.g. the tail's p1).
usage: [BigMoE venv] recycle_capture.py [N=100] -> stdout, data/recycle.txt, data/recycle_<domain>.npz"""
import glob, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
EV = os.path.expanduser("~/EvalSuite")
sys.path.insert(0, EV)
from evalsuite import stages as SG, model as M, engine as E     # first: evalsuite.env sets the engine's env
sys.path.insert(0, os.path.expanduser("~/Leap/third_party"))
import dflash_model_mlx as DF

LD = os.path.expanduser("~/Leap/data")
OUT = os.path.expanduser("~/MoonShots/data")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
V, K = 248320, 16
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


# ---------------------------------------------------------------- Phase A: MTP Head's walk, drafts kept
class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(V, 2560, group_size=64, bits=4)
        self.lm_head = nn.QuantizedLinear(2560, V, bias=False, group_size=64, bits=4)


def load_drafter():
    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    tgt = Target()
    tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                     + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    del w
    mx.eval(tgt.parameters())
    d = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    d.load_weights(os.path.join(LD, "l120_ft.safetensors"), strict=False)
    d.bind(tgt)
    mx.eval(d.parameters())
    return d, tgt


def head2s(d, hid):
    lg = d.compute_logits(hid)
    B, s = 128, lg.shape
    blk = lg.reshape(*s[:-1], s[-1] // B, B)
    tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
    sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, B)), axis=-2).reshape(*s[:-1], K * B)
    j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
    c = mx.take_along_axis(tb, j // B, axis=-1) * B + j % B
    return c, mx.take_along_axis(lg, c, axis=-1)


def chain(d, blk, unary, cand, hp):
    sel = d.candidate_selector
    pred, path = blk[:, 0], []
    for p in range(hp.shape[1]):
        sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                  * sel.successor_codebook(cand[:, p]), axis=-1)
        pred = mx.take_along_axis(cand[:, p], mx.argmax(sc, axis=-1)[:, None], axis=-1)[:, 0]
        path.append(pred)
    return mx.stack(path, axis=1)


def tail(d, hid, blk):
    cand, unary = head2s(d, hid)
    return chain(d, blk, unary, cand, d.candidate_selector.hidden_projection(hid))


def walk(d, sample, bs=16):
    """D-02b's walk_stats for one sample, plus each check's drafts -> [(accepted, checkable, anchor, drafts)]"""
    f, ids, off, gen = sample
    cache = d.make_cache()
    MASK = int(d.config.mask_token_id)
    start = max(off, gen - (d.config.sliding_window - 1))
    for c in cache:
        c.offset = start
    fed, a, out, n = start, gen, [], len(ids)
    while a < off + n - 1:
        ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
        blk = mx.array([[int(ids[a - off])] + [MASK] * (bs - 1)])
        dr = np.array(tail(d, d.hidden_states(blk, ctx, cache, 1), blk)[0]).tolist()
        fed = a
        true = ids[a - off + 1:a - off + bs]
        k = 0
        while k < len(true) and dr[k] == true[k]:
            k += 1
        out.append((k, len(true), a, dr))
        if (trim := cache[0].offset - a) > 0:
            DF._trim_recent_cache(cache, trim)
        a = a + k + 1
    return out


t0 = time.time()
d, tgt = load_drafter()
doms = {}
for dom in ("code", "math"):
    smp = []
    for p in sorted(glob.glob(os.path.join(LD, f"dfl2d_{dom}", "*.npz"))):
        z = np.load(p)
        assert int(z["off"]) == 0, p
        smp.append((z["f"].reshape(z["f"].shape[0], -1), z["ids"].astype(np.int64), int(z["off"]), int(z["gen"])))
    walks = [walk(d, s) for s in smp]
    stored = np.load(os.path.join(os.path.expanduser("~/TrainLab/data"), f"d02b_walk_{dom}.npz"))
    same = sum(int(np.array_equal(np.array([(k, m, a) for k, m, a, _ in w_]), stored[f"{i}_checks"])) for i, w_ in enumerate(walks))
    kept = np.mean([k + 1 for w_ in walks for k, _, _, _ in w_])
    say(f"Phase A {dom}: {len(smp)} samples, {sum(len(w_) for w_ in walks)} checks, kept per check {kept:.2f}; walks equal to "
        f"D-02b's stored ones: {same} / {len(smp)} ({time.time() - t0:.0f} s)")
    doms[dom] = ([s[1] for s in smp], walks)          # token ids only from here on (the features are Phase A's)
    del smp
del d, tgt
mx.clear_cache()

# ---------------------------------------------------------------- Phase B: the 125B's verify rows on the drafted blocks
rng = np.random.default_rng(0)
items, meta = [], []
for dom, (smp, walks) in doms.items():
    elig = [(i, c) for i, w_ in enumerate(walks) for c in range(len(w_) - 1) if w_[c][0] < 15 and w_[c][1] == 15]
    pick = sorted(rng.choice(len(elig), min(N, len(elig)), replace=False).tolist())
    for j in pick:
        i, c = elig[j]
        k, m, a, dr = walks[i][c]
        ids = smp[i]
        items.append(dict(key=f"{dom}_{i}_{a}", set=dom, ids=[int(x) for x in ids[:a + 1]] + [int(x) for x in dr[:15]],
                          pos=list(range(a, a + 16))))
        meta.append((dom, i, c))
v = M.variant("ref")
t1 = time.time()
rt, st = M.load(v, C=128)
M.configure(v)
say(f"Phase B: 'ref' loaded in {time.time() - t1:.0f} s; {len(items)} branch sequences, {sum(len(it['ids']) for it in items):,} tokens")


def head(i, pos, logits):
    lp = E.logsoftmax(logits)
    return {"top1": mx.argmax(lp, axis=-1).astype(mx.int32), "p1": mx.exp(mx.max(lp, axis=-1))}


res = SG.run_batches(rt, items, head, label="M-06 verify rows")
st.gate.check()
say(f"Phase B done ({time.time() - t1:.0f} s)")

for dom in doms:
    smp, walks = doms[dom]
    R = []
    agree, rows = 0, 0
    for n_, (dm, i, c) in enumerate(meta):
        if dm != dom:
            continue
        k, m, a, dr = walks[i][c]
        ids = smp[i]
        top1, p1 = res[n_]["top1"], res[n_]["p1"]
        agree += int(np.sum(top1[:k + 1] == ids[a + 1:a + k + 2])); rows += k + 1
        tail_ = top1[k + 1:16]                             # rows k + 1 .. 15 -> positions a + k + 2 ..
        true = ids[a + k + 2:a + k + 2 + len(tail_)]
        r = 0
        while r < len(true) and tail_[r] == true[r]:
            r += 1
        fk = walks[i][c + 1][0]
        R.append((k, r, fk, len(tail_), float(np.mean(p1[k + 1:16])) if len(tail_) else 0.0, float(p1[k + 1]) if len(tail_) else 0.0))
    R = np.array(R)
    kf, kr = R[:, 2], R[:, 1]
    gain = np.maximum(kr, kf).mean() - kf.mean()
    say(f"{dom}: {len(R)} checks with a first miss; rows 0..acc argmax == the 125B's tokens {agree} / {rows}; "
        f"E[kept_f] {kf.mean():.2f}, E[kept_r] {kr.mean():.2f} (tail {R[:, 3].mean():.1f} rows), "
        f"oracle E[max] - E[kept_f] = {gain:+.2f}; kept_r > kept_f on {100 * np.mean(kr > kf):.0f} % "
        f"(mean excess there {np.mean((kr - kf)[kr > kf]) if (kr > kf).any() else 0:.2f})")
    for lo, hi in ((0, 3), (3, 8), (8, 15)):
        s = (R[:, 0] >= lo) & (R[:, 0] < hi)
        if s.any():
            say(f"    acc {lo}-{hi - 1}: {s.sum():3d} checks, E[kept_f] {kf[s].mean():5.2f}, E[kept_r] {kr[s].mean():5.2f}, "
                f"oracle gain {np.maximum(kr[s], kf[s]).mean() - kf[s].mean():+.2f}")
    np.savez(os.path.join(OUT, f"recycle_{dom}.npz"), R=R, meta=np.array([(i, c) for dm, i, c in meta if dm == dom]))
say(f"({time.time() - t0:.0f} s)")
open(os.path.join(OUT, "recycle.txt"), "w").write("\n".join(lines) + "\n")
