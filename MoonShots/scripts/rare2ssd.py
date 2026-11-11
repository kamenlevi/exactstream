"""M-12 RARE2 step 2 (MoonShots, Sun 2026-11-01; NewIdeas GO 07:4x, NewIdeas' lines): how many SSD bytes per kept token does
the rare-expert 2-bit rule save under the live cache rule?
Rule (step 1, data/rare2.txt): per layer the rare-20 % experts use a 2-bit tier (MSB-2) at ranks 2-9; ranks 0-1 stay 4-bit, other
experts as REF2-T. Here the rare set = the bottom 102 experts per layer by popularity on the OTHER prompts (leave-one-prompt-out).
Replay = BigMoE's scripts/seed45_lru.py, the live rule LRU + popularity prior (victim key = last-use stamp + 30 x popularity),
CHAINED over cap45's six prompts (res0 forced in per prompt, prompts 2-6 counted), at C 224 / 256 / 278 = K 246 / 287 / 315
resident MSB-3 slots per layer. Residency is unchanged by the rule (no extra slots credited for the smaller rare records).
Ranks per row: recomputed from cap45's captured MoE inputs x with the real 8-bit router (softmax order); an id outside the
recomputed top-10 counts as rank 9.
Bytes (BigMoE DESIGN_3bit.md record parts): baseline miss = MSB-3 record 2.16 MB. RARE2: a rare expert needed in a check only at
ranks 2-9 reads MSB-2 (2-bit codes 1,228,800 B + s3 + b2 = 1.54 MB) and stays "2-bit resident"; a rare expert needed at rank
0-1 reads MSB-2 + the bit-1 plane (+ b3: 0.77 MB) = 2.30 MB; a hit on a 2-bit resident needed at rank 0-1 pays the bit-1 plane
(0.77 MB). LSB traffic is identical in both arms and left out (so the shares below are of MSB-side bytes).
Lines (NewIdeas): PASS = >= 10 % fewer SSD MB per kept token at C 256; KILL = < 5 %; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python rare2ssd.py (CPU, one core, ~2 min) -> stdout + data/rare2ssd.txt"""
import heapq, json, os, struct, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
MD = os.path.expanduser("~/BigMoE/models/bare")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E, R = 48, 512, 16
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
MSB3, MSB2, BIT1 = 2162688 / 1e6, (1228800 + 153600 + 153600) / 1e6, (614400 + 153600) / 1e6
KS = ((224, 246), (256, 287), (278, 315))
PRIOR = 30.0
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")
t0 = time.time()
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
heads = {}


def tensor(name):
    f = os.path.join(MD, idx[name])
    if f not in heads:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            heads[f] = (8 + n, json.loads(fh.read(n)))
    start, h = heads[f]
    m = h[name]
    dt = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}[m["dtype"]]
    return np.memmap(f, dtype=dt, mode="r", offset=start + m["data_offsets"][0], shape=tuple(m["shape"]))


def f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def router(l):
    pre = f"language_model.model.layers.{l}.mlp.gate."
    w, s, b = (tensor(pre + k) for k in ("weight", "scales", "biases"))
    bits = w.shape[-1] * 32 // (s.shape[-1] * 64)
    sh = np.arange(0, 32, bits, dtype=np.uint32)
    q = ((np.asarray(w)[..., None] >> sh) & np.uint32((1 << bits) - 1)).reshape(w.shape[0], -1).astype(np.float32)
    q = q.reshape(w.shape[0], -1, 64)
    return (q * f32(s)[..., None] + f32(b)[..., None]).reshape(w.shape[0], -1)


WR = [router(l) for l in range(NL)]
P, agree = [], [0, 0]
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, xoff, ids, X = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"], z["ids"], z["x"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"need": {}, "minr": {}, "kept": None})
        if s not in order:
            order.append(s)
        if l in c["need"]:
            c["kept"] = r
            continue
        rid = ids[roff[n]:roff[n] + r].astype(np.int64)                   # [r, 10] captured (order not trusted)
        lg = f32(X[xoff[n]:xoff[n] + r]) @ WR[l].T                         # [r, 512]
        rank = np.argsort(np.argsort(-lg, axis=1), axis=1)                # rank of every expert per row
        rr = np.minimum(np.take_along_axis(rank, rid, 1), 9)
        agree[0] += int((np.take_along_axis(rank, rid, 1) < 10).sum()); agree[1] += rid.size
        need = np.unique(rid)
        mr = {int(e): int(rr[rid == e].min()) for e in need}
        c["need"][l] = frozenset(mr); c["minr"][l] = mr
    cks = [checks[s] for s in order if len(checks[s]["need"]) == NL and checks[s]["kept"] is not None]
    P.append((name, cks, [frozenset(np.nonzero(z["res0"][l] >= 0)[0].tolist()) for l in range(NL)]))
POP = []
for i in range(len(P)):
    pop, n = np.zeros((NL, E)), 0
    for j, (_, cks, _) in enumerate(P):
        if j != i:
            for c in cks:
                for l in range(NL):
                    pop[l, list(c["need"][l])] += 1
            n += len(cks)
    POP.append(pop / n)
RARE = [[set(np.argsort(POP[p][l], kind="stable")[:102].tolist()) for l in range(NL)] for p in range(len(P))]
say(f"M-12 RARE2 step 2: cap45, {sum(len(c) for _, c, _ in P)} live checks over {len(P)} prompts, chained; live rule LRU + "
    f"{PRIOR:.0f} x popularity; recomputed router top-10 contains {100 * agree[0] / agree[1]:.2f} % of the captured ids "
    f"({time.time() - t0:.0f} s)")


def replay(K):
    """per prompt: [checks, 6] = misses, rare misses, rare misses needed only at ranks 2-9, base MB, RARE2 MB, upgrades"""
    out = [np.zeros((len(c), 6)) for _, c, _ in P]
    for l in range(NL):
        st, two, i = {}, set(), 0
        for p, (_, cks, res0) in enumerate(P):
            pop, rare = POP[p][l], RARE[p][l]
            seq = [(-1, res0[l], None)] + [(t, c["need"][l], c["minr"][l]) for t, c in enumerate(cks)]
            for t, need, mr in seq:
                i += 1
                if t >= 0:
                    o = out[p][t]
                    for e in need:
                        lowonly = mr[e] >= 2
                        if e not in st:
                            o[0] += 1; o[3] += MSB3
                            if e in rare:
                                o[1] += 1
                                if lowonly:
                                    o[2] += 1; o[4] += MSB2; two.add(e)
                                else:
                                    o[4] += MSB2 + BIT1; two.discard(e)
                            else:
                                o[4] += MSB3
                        elif e in two and not lowonly:
                            o[4] += BIT1; o[5] += 1; two.discard(e)
                else:                                                    # prefill: full records, not counted
                    for e in need:
                        two.discard(e)
                for e in need:
                    st[e] = float(i)
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need),
                                             key=lambda e: st[e] + PRIOR * pop[e]):
                        del st[e]; two.discard(e)
        if os.path.exists(PAUSE):
            while os.path.exists(PAUSE):
                time.sleep(20)
    return out


def tot(o, col):
    return sum(o[p][:, col].sum() for p in range(1, len(P)))


KEPT = sum(sum(c["kept"] for c in P[p][1]) for p in range(1, len(P)))
say("prompts 2-6 pooled, per kept token:")
say("  C    K    misses  rare  rare ranks 2-9 only  upgrades  MSB-side MB base  RARE2   saved")
saved = {}
for C, K in KS:
    o = replay(K)
    m, rm, rl, b, r2, up = (tot(o, j) / KEPT for j in (0, 1, 2, 3, 4, 5))
    saved[C] = 1 - r2 / b
    say(f"  {C}  {K}  {m:7.1f} {rm:6.1f} {rl:12.1f} {up:16.2f} {b:10.1f} {r2:8.1f} {100 * saved[C]:6.1f} %")
s = saved[256]
call = "PASS" if s >= 0.10 else ("KILL" if s < 0.05 else "BETWEEN")
say(f"Lines (NewIdeas): RARE2 saves {100 * s:.1f} % of SSD MB per kept token at C 256 (MSB side; PASS >= 10 %, KILL < 5 %) -> "
    f"{call}. With LSB reads in the denominator the share is smaller; slots freed by 1.54 MB rare records are not credited.")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/rare2ssd.txt"), "w").write("\n".join(lines) + "\n")
