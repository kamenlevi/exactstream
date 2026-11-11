"""D-14 probes for the design study (MTP Head, Thu 10-29; CPU only, stored data). Three questions about DFlash2 L1-20
(block 16, 1 step; walks data/d02b_walk_{code,math}.npz and data/d02b_walk_d02a_{ifeval,multi,agent}.npz):
A. depth-1 hit rate split by the 125B's confidence on that token (sure p1 >= 0.99 / tie p1 - p2 <= 0.1 / mid): does
   the drafter miss even tokens the 125B is sure of (a knowledge or coverage gap) on IFEval / multi / agent?
B. multi per language (bg de fr es ru zh ja ar): kept per check, depth-1 rate, the 125B's sure share, depth-1 hits on
   sure tokens.
C. agent: is the true continuation at a first miss copyable from the context (its 4-gram occurs earlier), and if so
   only beyond the drafter's 2048-token window? Same split for all depth-1 positions as the base rate. Full sequences
   from ~/Chat1000/data/agent125 (prompt + out).
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d14_probe.py -> data/d14_probe.txt"""
import collections, os
import numpy as np

H = os.path.expanduser
TD = H("~/TrainLab/data")
SETS = [("code", "d02b_walk_code.npz", H("~/Leap/data/l118_tgt")), ("math", "d02b_walk_math.npz", TD + "/d02a"),
        ("ifeval", "d02b_walk_d02a_ifeval.npz", TD + "/d02a"), ("multi", "d02b_walk_d02a_multi.npz", TD + "/d02a"),
        ("agent", "d02b_walk_d02a_agent.npz", TD + "/d02a")]
WIN, NG = 2048, 4
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def cat(lp_row):
    p1, p2 = np.exp(float(lp_row[0])), np.exp(float(lp_row[1]))
    return "sure" if p1 >= 0.99 else "tie" if p1 - p2 <= 0.1 else "mid"

def load(wf, pdir):
    W = np.load(os.path.join(TD, wf))
    out = []
    for s, nm in enumerate(W["names"]):
        nm = str(nm)
        P = np.load(os.path.join(pdir, nm))
        out.append(dict(name=nm, ids=W[f"{s}_ids"], off=int(W[f"{s}_off"]), gen=int(W[f"{s}_gen"]),
                        checks=W[f"{s}_checks"], tk=P["tk"], lp=P["lp"].astype(np.float32),
                        poff=int(P["off"]) if "off" in P.files else 0))
    return out

def latest(hay, needle, end):
    """latest start s < end - len(needle) + 1 of needle in hay[:end] (int arrays), or -1"""
    hb, nb = hay[:end].astype(np.int32).tobytes(), needle.astype(np.int32).tobytes()
    e = len(hb)
    while True:
        i = hb.rfind(nb, 0, e)
        if i < 0:
            return -1
        if i % 4 == 0:
            return i // 4
        e = i + len(nb) - 1

say("D-14 probes (DFlash2 L1-20, block 16, 1 step). A: depth-1 hit rate by the 125B's confidence on the token")
say("(sure p1 >= 0.99 / tie p1 - p2 <= 0.1 / mid), on-path positions only; n = depth-1 trials in that class.")
data = {}
for dom, wf, pdir in SETS:
    S = load(wf, pdir)
    data[dom] = S
    hit = collections.Counter(); tri = collections.Counter()
    for x in S:
        for k, m, a in x["checks"]:
            k, m, a = int(k), int(m), int(a)
            if m < 1:
                continue
            p = a + 1; r = p - 1 - x["poff"]
            if not 0 <= r < x["tk"].shape[0] or int(x["tk"][r, 0]) != int(x["ids"][p - x["off"]]):
                continue
            c = cat(x["lp"][r]); tri[c] += 1; hit[c] += int(k >= 1)
    say(f"  {dom:6s} " + "  ".join(f"{c} {hit[c] / max(tri[c], 1):.3f} (n {tri[c]})" for c in ("sure", "mid", "tie")))

say("")
say("B: multi per language (24 answers, 3 per language): kept per check, depth-1 rate, the 125B's sure share of the")
say("completion tokens, depth-1 hit rate on sure tokens")
per = collections.defaultdict(lambda: dict(kept=0, n=0, d1=0, d1n=0, sure=0, tok=0, sh=0, sn=0))
for x in data["multi"]:
    lang = "".join(ch for ch in x["name"].split("_")[2] if ch.isalpha())
    g = per[lang]
    for k, m, a in x["checks"]:
        k, m, a = int(k), int(m), int(a)
        g["kept"] += k + 1; g["n"] += 1
        if m >= 1:
            g["d1n"] += 1; g["d1"] += int(k >= 1)
            r = a - x["poff"]
            if 0 <= r < x["tk"].shape[0] and int(x["tk"][r, 0]) == int(x["ids"][a + 1 - x["off"]]) and cat(x["lp"][r]) == "sure":
                g["sn"] += 1; g["sh"] += int(k >= 1)
    for p in range(x["gen"] + 1, x["off"] + len(x["ids"])):
        r = p - 1 - x["poff"]
        if 0 <= r < x["tk"].shape[0]:
            g["tok"] += 1; g["sure"] += int(np.exp(float(x["lp"][r, 0])) >= 0.99)
for lang, g in per.items():
    say(f"  {lang}: kept {g['kept'] / max(g['n'], 1):.2f} ({g['n']} checks), depth-1 {g['d1'] / max(g['d1n'], 1):.3f}, "
        f"125B sure share {g['sure'] / max(g['tok'], 1):.2f}, depth-1 on sure tokens {g['sh'] / max(g['sn'], 1):.3f} (n {g['sn']})")

say("")
say(f"C: agent, is the true {NG}-gram at the position copyable from the context before it: W = its latest earlier")
say(f"occurrence lies inside the drafter's {WIN}-token window, B = only beyond it, N = nowhere earlier")
cnt = {"first miss": collections.Counter(), "depth-1 hit": collections.Counter(), "depth-1 miss": collections.Counter()}
for x in data["agent"]:
    z = np.load(H("~/Chat1000/data/agent125/") + x["name"][len("agent_"):])
    full = np.concatenate([z["prompt"], z["out"]]).astype(np.int64)
    assert (full[x["off"]:x["off"] + len(x["ids"])] == x["ids"]).all(), x["name"]
    def where(p):
        if p + NG > len(full):
            return None
        s = latest(full, full[p:p + NG], p)
        return "N" if s < 0 else "W" if s >= p - WIN else "B"
    for k, m, a in x["checks"]:
        k, m, a = int(k), int(m), int(a)
        if m >= 1:
            w = where(a + 1)
            if w:
                cnt["depth-1 hit" if k >= 1 else "depth-1 miss"][w] += 1
        if k < m:
            w = where(a + k + 1)
            if w:
                cnt["first miss"][w] += 1
for key, c in cnt.items():
    n = sum(c.values())
    say(f"  {key:12s}: n {n:4d}  W {c['W'] / max(n, 1):.2f}  B {c['B'] / max(n, 1):.2f}  N {c['N'] / max(n, 1):.2f}")
open(os.path.join(TD, "d14_probe.txt"), "w").write("\n".join(lines) + "\n")
