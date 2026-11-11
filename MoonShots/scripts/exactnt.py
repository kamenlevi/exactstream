"""M-15 EXACT-NT (MoonShots, Sun 2026-11-01 09:0x; QUEUE row M-15, lines written first): can RANKED #21's near-tie acceptance be
made EXACT, and how much of R-61N's chat gain survives?
Construction (speculative cascades, Narasimhan et al., ICLR 2025, arXiv 2405.19261, adapted to a block drafter): the served
model's own one-token rule is "take T when T != the 125B's top-1 and p_125B(T) >= rho x p_top1, else the top-1", where T = the
drafter's depth-1 token from its normal block call started at this position (L1-20, cand21's selector at t 0.0). T depends only
on the prefix (target features up to the position), so the live loop can stay bit-exact vs this rule's own AR decode. In the
loop a block's drafted token d_j is accepted iff d_j == the rule's token: a first miss is forgiven only if d_j == T (R-61N
forgives whenever p_d >= rho x p_top1), and a greedy hit (d_j == top-1) becomes a miss where T is another near-tie.
Replay: R-61N's chat chain as recorded (~/TrainLab/data/r61n/chatnt_b16.jsonl, read-only) + the 125B's top-32 per position from
D-02a's captures + a fresh drafter block at EVERY chat position for T (stage 1, cached in data/exactnt_T.npz). Accounting =
nt_analyze's: LB = +1 token per forgiven block; MOD = after a forgiven token the run continues with the replay's conditional
survival and later misses forgiven at the rule's rate (an estimate). R-61N's own numbers are reproduced as a check.
Lines (QUEUE M-15): PASS = chat MOD >= +15 % at rho 0.25 net of rule-made misses; KILL = < +8 %; BETWEEN otherwise.
usage: [BigMoE venv] python exactnt.py (EXNT_DEV=gpu default, as R-61N ran; EXNT_ANALYZE_ONLY=1 for stage 2 only)
-> stdout + data/exactnt.txt"""
import os, sys, glob, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
BS = 16
SRC = os.path.expanduser(f"~/TrainLab/data/r61n/chatnt_b{BS}.jsonl")
CAP = os.path.expanduser("~/TrainLab/data/d02a")
TF = os.path.expanduser("~/MoonShots/data/exactnt_T.npz")
OUT = os.path.expanduser("~/MoonShots/data/exactnt.txt")
DOMS = ("tf_ifeval", "tf_multi")
RHOS = (0.5, 0.25, 0.1)
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def wait_pause():
    while os.path.exists(PAUSE):
        time.sleep(20)


keys = sorted(os.path.basename(p)[:-4] for d in DOMS for p in glob.glob(f"{CAP}/{d}_*.npz"))
TS = dict(np.load(TF)) if os.path.exists(TF) else {}
todo = [k for k in keys if k not in TS]
t0 = time.time()
if todo and os.environ.get("EXNT_ANALYZE_ONLY") != "1":
    # ---------------------------------------------------------- stage 1: the drafter's fresh depth-1 token at every position
    wait_pause()
    import mlx.core as mx, mlx.nn as nn
    if os.environ.get("EXNT_DEV", "gpu") == "cpu":
        mx.set_default_device(mx.cpu)
    sys.path.insert(0, os.path.expanduser("~/Leap/third_party"))
    import dflash_model_mlx as D

    class Target(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.QuantizedEmbedding(248320, 2560, group_size=64, bits=4)
            self.lm_head = nn.QuantizedLinear(2560, 248320, bias=False, group_size=64, bits=4)

    w = mx.load(os.path.expanduser("~/BigMoE/models/bare/model-00016-of-00017.safetensors"))
    tgt = Target()
    tgt.load_weights([(f"embed_tokens.{s}", w[f"language_model.model.embed_tokens.{s}"]) for s in ("weight", "scales", "biases")]
                     + [(f"lm_head.{s}", w[f"language_model.lm_head.{s}"]) for s in ("weight", "scales", "biases")])
    del w
    draft = D.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    draft.bind(tgt)
    draft.load_weights(os.path.expanduser("~/Leap/data/l120_ft.safetensors"), strict=False)
    mx.eval(draft.parameters(), tgt.parameters())
    MASK = int(draft.config.mask_token_id)
    sel = draft.candidate_selector

    def first_token(block, hidden, logits):
        """cand21's selector at temperature 0, step p = 0 only: identical to r61n_chatnt.select_conf's path[0]"""
        cand = mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:]
        unary = mx.take_along_axis(logits, cand, axis=-1)
        hp = sel.hidden_projection(hidden)
        sc = unary[:, 0] + mx.sum(sel.predecessor_codebook(block[:, 0])[:, None] * hp[:, 0, None]
                                  * sel.successor_codebook(cand[:, 0]), axis=-1)
        s = mx.argmax(sc, axis=-1)
        return mx.take_along_axis(cand[:, 0], s[:, None], axis=-1)[:, 0]

    say(f"stage 1: fresh blocks at every position of {len(todo)} chat answers (model load {time.time() - t0:.0f} s)")
    for key in todo:
        wait_pause()
        z = np.load(f"{CAP}/{key}.npz")
        f = z["f"].reshape(z["f"].shape[0], -1); ids = z["ids"]; off, gen = int(z["off"]), int(z["gen"]); n = len(ids)
        T = np.full(n, -1, np.int64)                                   # T[q - off] = fresh token for position q
        cache = draft.make_cache()
        start = max(off, gen - (draft.config.sliding_window - 1))
        for c in cache:
            c.offset = start
        fed, a = start, gen
        while a < off + n - 1:                                         # r61n_chatnt's loop with acc = 1
            ctx = mx.array(f[fed - off:a - off][None]).astype(mx.bfloat16)
            blk = mx.array([[int(ids[a - off])] + [MASK] * (BS - 1)])
            h = draft.hidden_states(blk, ctx, cache, 1)
            t = first_token(blk, h, draft.compute_logits(h)); mx.eval(t)
            T[a + 1 - off] = int(t[0])
            fed = a
            if (trim := cache[0].offset - a) > 0:
                D._trim_recent_cache(cache, trim)
            a += 1
        TS[key] = T
        np.savez(TF, **TS)
        print(f"{key}: {int((T >= 0).sum())} positions, {time.time() - t0:.0f} s", flush=True)

# -------------------------------------------------------------------------------- stage 2: the chain under each rule
caps = {}
for key in keys:
    z = np.load(f"{CAP}/{key}.npz")
    caps[key] = (z["tk"], z["lp"].astype(np.float64), z["ids"], int(z["off"]))
ALL = [json.loads(l) for l in open(SRC)]
ALL = [r for r in ALL if r["dom"] in DOMS and r["key"] in TS]
say(f"M-15 EXACT-NT: R-61N's chat chain (block {BS}, L1-20 + cand21's selector at t 0.0, teacher-forced on D-02a), "
    f"{len(ALL)} blocks from {len({r['key'] for r in ALL})} answers; fresh T at every position ({time.time() - t0:.0f} s)")

# per block and depth: top-1, p_top1, T, p_125B(T) (0 if outside the 125B's top-32; flagged if that could matter)
amb = 0
for r in ALL:
    tk, lp, ids, off = caps[r["key"]]
    T = TS[r["key"]]
    r["T"], r["pT"], r["t1s"], r["txt"] = [], [], [], []
    for j in range(r["nb"]):
        i = r["a"] - off + j                                           # row i predicts the token at i + 1
        q = i + 1
        Tq = int(T[q])
        wh = np.where(tk[i] == Tq)[0]
        pT = float(np.exp(lp[i, wh[0]])) if len(wh) else 0.0
        if not len(wh) and np.exp(lp[i, -1]) >= min(RHOS) * np.exp(lp[i, 0]):
            amb += 1
        r["T"].append(Tq); r["pT"].append(pT); r["t1s"].append(int(tk[i, 0])); r["txt"].append(int(ids[q]))
    r["d0"] = r["t1s"][0] if r["hit"][0] else r.get("dt")             # the chain's own depth-1 token
self_ok = sum(r["d0"] == r["T"][0] for r in ALL)
greedy_ok = sum(all(r["t1s"][j] == int(caps[r["key"]][2][r["a"] - caps[r["key"]][3] + j + 1]) for j in range(r["nb"]))
                for r in ALL)
say(f"self-check: the chain's depth-1 token == fresh T at the same position in {self_ok} / {len(ALL)} blocks "
    f"({100 * self_ok / len(ALL):.2f} %); captured top-1 == the greedy text in {100 * greedy_ok / len(ALL):.2f} % of blocks; "
    f"T outside the 125B's top-32 where it could matter: {amb} positions")

npos = sum(r["nb"] for r in ALL); ndiff = sum(t != x for r in ALL for t, x in zip(r["t1s"], r["txt"]))
say(f"positions where the text token != the capture's top-1: {ndiff} / {npos} ({100 * ndiff / npos:.2f} %); the replay "
    f"treats the text token as the greedy one (R-61N's convention)")
miss = [r for r in ALL if r["k"] < r["nb"]]
agree = [r for r in miss if r["dt"] == r["T"][r["k"]]]
fix = [r for r in miss if r["T"][r["k"]] == r["txt"][r["k"]]]
d0 = [r for r in miss if r["k"] == 0]
d1 = [r for r in miss if r["k"] >= 1]
say(f"first misses {len(miss)}: block token == fresh T at {100 * len(agree) / len(miss):.1f} % (depth 1: "
    f"{100 * sum(r['dt'] == r['T'][0] for r in d0) / max(1, len(d0)):.1f} % of {len(d0)}; depth >= 2: "
    f"{100 * sum(r['dt'] == r['T'][r['k']] for r in d1) / max(1, len(d1)):.1f} % of {len(d1)}); fresh T == the text token "
    f"(the fresh block would have hit) at {100 * len(fix) / len(miss):.1f} %")


def run(recs, rho, exact, rng):
    """tokens per pass (LB, MOD), forgiven share of misses, rule-made misses per block; nt_analyze's accounting"""
    nb_max = BS - 1
    reach, hits = np.zeros(nb_max), np.zeros(nb_max)
    ev = []                                                          # per block: (kind, depth)
    for r in recs:
        kind, jj = "full", r["nb"]
        for j in range(r["nb"]):
            reach[j] += 1
            # R-61N's convention: the greedy token is the TEXT token (it differs from the capture's top-1 at ~2 % of positions,
            # near-ties between the capture's and the generation's numerics), probabilities from the capture
            nt = r["T"][j] != r["txt"][j] and r["pT"][j] >= rho * r["p1"][j]
            if j < r["k"]:
                if exact and nt:
                    kind, jj = "rule", j; break
                hits[j] += 1
                continue
            if exact:
                forgiven = nt and r["dt"] == r["T"][j]
            else:
                forgiven = r["pd"] >= rho * r["p1"][j]
            kind, jj = ("forg" if forgiven else "miss"), j
            break
        ev.append((kind, jj))
    surv = hits / np.maximum(1, reach)
    nmiss = sum(k in ("rule", "miss", "forg") for k, _ in ev)
    f = sum(k == "forg" for k, _ in ev) / max(1, nmiss)
    base = np.mean([r["acc"] for r in recs])
    lb = np.mean([min(j + 1, r["rem"]) if k == "rule" else
                  r["acc"] + (1 if k == "forg" and r["k"] + 2 <= r["rem"] else 0) for r, (k, j) in zip(recs, ev)])
    tot, RUNS = 0.0, 200
    for _ in range(RUNS):
        for r, (k, j) in zip(recs, ev):
            if k == "rule":
                tot += min(j + 1, r["rem"]); continue
            if k != "forg":
                tot += r["acc"]; continue
            j += 1
            while j < r["nb"]:
                if rng.random() < surv[j] or rng.random() < f:
                    j += 1
                else:
                    break
            tot += min(j + 1, r["rem"])
    mod = tot / (RUNS * len(recs))
    rule = sum(k == "rule" for k, _ in ev) / len(recs)
    return base, lb, mod, f, rule


GROUPS = [("chat (ifeval + multi)", DOMS), ("ifeval", ("tf_ifeval",)), ("multi", ("tf_multi",))]
verdict = None
for name, doms in GROUPS:
    recs = [r for r in ALL if r["dom"] in doms]
    say(f"\n== {name}: {len(recs)} blocks; exact greedy tokens per pass {np.mean([r['acc'] for r in recs]):.2f}")
    say("   rho   rule        forgiven  rule-made misses / block  tokens per pass LB        MOD")
    for rho in RHOS:
        for exact in (False, True):
            base, lb, mod, f, rule = run(recs, rho, exact, np.random.default_rng(0))
            say(f"   {rho:<5} {'EXACT-NT' if exact else 'R-61N   '}   {100 * f:5.1f} %   {rule:10.3f}              "
                f"{lb:.2f} ({100 * (lb / base - 1):+5.1f} %)  {mod:.2f} ({100 * (mod / base - 1):+5.1f} %)")
            if exact and rho == 0.25 and name.startswith("chat"):
                verdict = mod / base - 1
call = "PASS" if verdict >= 0.15 else ("KILL" if verdict < 0.08 else "BETWEEN")
say(f"\nLines (QUEUE M-15): EXACT-NT chat MOD at rho 0.25 {100 * verdict:+.1f} % (PASS >= +15 %, KILL < +8 %) -> {call}. "
> (a quote from the project owner, kept private)
say(f"({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
