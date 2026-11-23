"""D-17 MTPDEEP read-out (MTP Head, Thu 10-29; REPORT 'D-17'): the native MTP chain drafted from the true prefix at every
position (data/d17: D-05's 8 code / gsm / ifeval / multi prompts; ~/Chat1000/data/agent125: 12 agent turns). Per domain:
the conditional rate at depths 1-15 (depth j right given depths < j right, over every position with 15 tokens ahead),
and a walk that keeps the chain capped at K = 3 / 5 / 7 / 15 drafts (tokens per pass = accepted + 1, the next pass at
p + accepted + 1, the end token included). DFlash2 L1-20's kept per check on the same answers comes from its D-02b /
D-02a walks (the texts are the same prompts; this decode's text may differ in places). End tokens cut each answer.
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d17_read.py -> data/d17_read.txt"""
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

def cut(out):
    e = next((i for i, t in enumerate(out) if t in END), len(out) - 1)
    return e + 1                                     # positions 0..e (the end token is a draft target too)

def domain_stats(items):
    reach, right = np.zeros(16), np.zeros(16)
    tau = {K: [0, 0] for K in (3, 5, 7, 15)}
    for out, mtp in items:
        L = cut(out)
        for p in range(1, L):
            ok = True
            for j in range(15):
                if p + j >= L or mtp[p][j] < 0:
                    break
                reach[j + 1] += 1
                if mtp[p][j] == out[p + j]:
                    right[j + 1] += 1
                else:
                    break
        for K in tau:
            p = 1
            while p < L:
                n = 0
                while n < K and p + n < L and mtp[p][n] >= 0 and mtp[p][n] == out[p + n]:
                    n += 1
                tau[K][0] += n + 1; tau[K][1] += 1
                p += n + 1
    rates = [right[j] / reach[j] if reach[j] else float("nan") for j in range(1, 16)]
    return rates, {K: v[0] / max(v[1], 1) for K, v in tau.items()}

def dflash_subset(walk, keys):
    W = np.load(os.path.join(TD, walk))
    kept = n = 0
    for s, nm in enumerate(W["names"]):
        if str(nm)[:-4] in keys:
            ch = W[f"{s}_checks"]; kept += sum(int(k) + 1 for k, m, a in ch); n += len(ch)
    return kept / max(n, 1), n

say("D-17 MTPDEEP: the native MTP chain from the true prefix (conditional rate per depth; tokens per pass at a K cap)")
groups = {}
for p in sorted(glob.glob(os.path.join(TD, "d17", "*.npz"))):
    z = np.load(p)
    groups.setdefault(str(z["domain"]), []).append((os.path.basename(p)[:-4], [int(t) for t in z["out"]], z["mtp"]))
ag = []
for p in sorted(glob.glob(H("~/Chat1000/data/agent125/*.npz"))):
    z = np.load(p)
    ag.append(("agent_" + os.path.basename(p)[:-4], [int(t) for t in z["out"]], z["mtp"]))
groups["agent"] = ag
WALK = {"code": "d02b_walk_code.npz", "gsm": "d02b_walk_math.npz", "ifeval": "d02b_walk_d02a_ifeval.npz",
        "multi": "d02b_walk_d02a_multi.npz", "agent": "d02b_walk_d02a_agent.npz"}
for dom in ("code", "gsm", "ifeval", "multi", "agent"):
    if dom not in groups:
        continue
    items = [(o, m) for _, o, m in groups[dom]]
    rates, tau = domain_stats(items)
    dk, dn = dflash_subset(WALK[dom], {k for k, _, _ in groups[dom]})
    say(f"  {dom} ({len(items)} answers): MTP chain tokens per pass K3 {tau[3]:.2f} / K5 {tau[5]:.2f} / K7 {tau[7]:.2f} / "
        f"K15 {tau[15]:.2f}; DFlash2 L1-20 block 16 on the same answers {dk:.2f} ({dn} checks)")
    say("    MTP conditional rate, depths 1-15: " + " ".join(f"{r:.3f}" for r in rates))
open(os.path.join(TD, "d17_read.txt"), "w").write("\n".join(lines) + "\n")
