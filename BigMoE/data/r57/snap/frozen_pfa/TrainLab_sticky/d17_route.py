"""D-17 MTPDEEP router read-out (MTP Head, Thu 10-29; REPORT 'D-17'; NewIdeas' row frames it as the per-round router's
oracle). Both drafters at EVERY position of the same texts:
- the native MTP chain from the true prefix: data/d17/<key>.npz (D-05's 8 code / gsm / ifeval / multi prompts) and
  ~/Chat1000/data/agent125 (12 agent turns); mtp[p][j] guesses out[p + j];
- DFlash2 L1-20, block 16, 1 step: TL_WALK_EVERY walks data/d02b_walk_d17e_<dom>.npz (a check at every anchor a);
  its check at anchor a drafts out[a - gen + 1 ..], the same tokens as the MTP chain at p = a - gen + 1.
Walks over each answer (cut at its first end token), tokens per pass = accepted + 1:
DFlash2 alone; MTP alone capped at K = 7 / 15; the per-round oracle (the better of the two at each pass); and a
confidence router (MTP when the 125B's top-1 p on the bonus token is below a threshold, else DFlash2; the threshold
picked in-sample, so it is optimistic). The 125B's p comes from the D-02a-style captures (data/d17f, data/d02a).
Lines (NewIdeas' D-17 row): PASS oracle >= +0.5 kept over DFlash2 on chat or agent; KILL < +0.2 on all three
(ifeval / multi / agent).
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d17_route.py -> data/d17_route.txt"""
import glob, os
import numpy as np
from tokenizers import Tokenizer

H = os.path.expanduser
TD = H("~/TrainLab/data")
tk = Tokenizer.from_file(H("~/BigMoE/models/bare/tokenizer.json"))
END = {tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def texts(dom):
    """[(out, mtp, kD by output position, p1 of the 125B on out[q] by output position q)]"""
    W = np.load(os.path.join(TD, f"d02b_walk_d17e_{dom}.npz"))
    res = []
    for s, nm in enumerate(W["names"]):
        nm = str(nm)
        if dom == "agent":
            z = np.load(H("~/Chat1000/data/agent125/") + nm[len("agent_"):]); P = np.load(os.path.join(TD, "d02a", nm))
        else:
            z = np.load(os.path.join(TD, "d17", nm)); P = np.load(os.path.join(TD, "d17f", nm))
        out, mtp = [int(t) for t in z["out"]], z["mtp"]
        gen, off = int(W[f"{s}_gen"]), int(W[f"{s}_off"])
        ids = W[f"{s}_ids"]
        n = min(len(out), off + len(ids) - gen)
        assert [int(t) for t in ids[gen - off:gen - off + n]] == out[:n], nm
        kD = {}
        for k, m, a in W[f"{s}_checks"]:
            kD[int(a) - gen + 1] = int(k)
        pof, lp = int(P["off"]), P["lp"].astype(np.float32)
        p1 = {}
        for q in range(n):
            r = gen + q - 1 - pof                         # the 125B's row predicting out[q]
            if 0 <= r < lp.shape[0]:
                p1[q] = float(np.exp(lp[r, 0]))
        e = next((i for i, t in enumerate(out[:n]) if t in END), n - 1)
        res.append((out[:e + 1], mtp, kD, p1))
    return res

def kM(out, mtp, p, K):
    n, L = 0, len(out)
    while n < K and p + n < L and mtp[p][n] >= 0 and mtp[p][n] == out[p + n]:
        n += 1
    return n

def walk(T, choose):
    tok = passes = mtp_passes = 0
    for out, mtp, kD, p1 in T:
        p, L = 1, len(out)
        while p < L:
            d = min(kD.get(p, 0), L - p)
            use_m, k = choose(out, mtp, p, d, p1)
            tok += k + 1; passes += 1; mtp_passes += use_m
            p += k + 1
    return tok / max(passes, 1), mtp_passes / max(passes, 1)

say("D-17 router read-out: tokens per pass on the same answers (both drafters at every position)")
for dom in ("code", "gsm", "ifeval", "multi", "agent"):
    if not os.path.exists(os.path.join(TD, f"d02b_walk_d17e_{dom}.npz")):
        say(f"  {dom}: no every-position DFlash2 walk"); continue
    T = texts(dom)
    dfl, _ = walk(T, lambda o, m, p, d, p1: (0, d))
    row = [f"DFlash2 {dfl:.2f}"]
    for K in (7, 15):
        mt, _ = walk(T, lambda o, m, p, d, p1, K=K: (1, kM(o, m, p, K)))
        orc, share = walk(T, lambda o, m, p, d, p1, K=K: (1, kM(o, m, p, K)) if kM(o, m, p, K) > d else (0, d))
        row.append(f"MTP K{K} {mt:.2f}, oracle K{K} {orc:.2f} (MTP on {share:.0%} of passes, +{orc - dfl:.2f})")
    best = None
    for th in (0.3, 0.5, 0.7, 0.9, 0.99):
        r, share = walk(T, lambda o, m, p, d, p1, th=th: (1, kM(o, m, p, 15)) if p1.get(p - 1, 1.0) < th else (0, d))
        if best is None or r > best[0]:
            best = (r, th, share)
    row.append(f"router (MTP K15 when p1(bonus) < {best[1]}) {best[0]:.2f} (MTP on {best[2]:.0%}, +{best[0] - dfl:.2f}, "
               f"in-sample threshold)")
    say(f"  {dom} ({len(T)} answers): " + "; ".join(row))
say("Lines (NewIdeas): PASS oracle >= +0.5 kept over DFlash2 on chat (multi) or agent; KILL < +0.2 on ifeval, multi and agent.")
open(os.path.join(TD, "d17_route.txt"), "w").write("\n".join(lines) + "\n")
