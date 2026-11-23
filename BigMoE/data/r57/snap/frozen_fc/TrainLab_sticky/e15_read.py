"""E1 / E5 read-out (MTP Head, Sun 11-01 night; Fable ~/FableGod/FINDINGS.md MISSION 4 sec 4 E1 / E5, MISSION 3 sec 4).
CPU only, on walks (sticky/d02b_walk.py, block 16, official 1-step loop) of the held-out D-02a answers: multi (24, 3 per
language) and IFEval (30), none of which any training set here contains (DLOG = MGSM-seeded prompts). Baseline = L1-20's
own walks of the same answers (data/d02b_walk_d02a_{multi,ifeval}.npz, D-02a). Per arm (E15_ARMS=tag,tag,...: the walk tags,
files data/d02b_walk_<tag>_{multi,ifeval}.npz and _<tag>.txt):
- kept per check; depth-1 hit rate on on-path tokens by the 125B's confidence (D-14 A: sure p1 >= 0.99 / tie p1 - p2 <= 0.1
  / mid), per set; multi per language (D-14 B); code / math kept (forgetting) from the walk's txt.
Lines (Fable's, written before these runs):
- E1 (coverage from data): PASS = de / ru / ja sure-token depth-1 hit (pooled) 0.67-0.73 -> >= 0.85 AND multi kept +0.5;
  KILL = the pooled de / ru / ja sure-hit gain < +0.05; else BETWEEN.
- E5 (borrowed experts): PASS = multi sure-token depth-1 hit 0.62 -> >= 0.75 OR IFEval mid-token hit 0.565 -> >= 0.62
  (MISSION 4); KILL = < +0.1 on both the multi sure-hit and IFEval kept (MISSION 3); else BETWEEN. E5 is also shown against
  E1 (same data, full-body tune, no experts) when both exist: the experts' own share.
Output: data/e15_read.txt (appended)."""
import collections, os, re
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def cat(lp_row):
    p1, p2 = np.exp(float(lp_row[0])), np.exp(float(lp_row[1]))
    return "sure" if p1 >= 0.99 else "tie" if p1 - p2 <= 0.1 else "mid"

def stats(wf):
    """per key ('all', each language for multi): kept / checks, depth-1 hits / trials by class"""
    W = np.load(os.path.join(TD, wf))
    g = collections.defaultdict(lambda: collections.Counter())
    for s, nm in enumerate(W["names"]):
        nm = str(nm)
        P = np.load(os.path.join(TD, "d02a", nm))
        tk, lp, poff = P["tk"], P["lp"].astype(np.float32), int(P["off"]) if "off" in P.files else 0
        ids, off = W[f"{s}_ids"], int(W[f"{s}_off"])
        keys = ["all"] + (["".join(c for c in nm[:-4].split("_")[2] if c.isalpha())] if nm.startswith("tf_multi_") else [])
        for k, m, a in W[f"{s}_checks"]:
            k, m, a = int(k), int(m), int(a)
            for key in keys:
                g[key]["kept"] += k + 1; g[key]["n"] += 1
            if m < 1:
                continue
            r = a - poff
            if not 0 <= r < tk.shape[0] or int(tk[r, 0]) != int(ids[a + 1 - off]):
                continue
            c = cat(lp[r])
            for key in keys:
                g[key][c + "_n"] += 1; g[key][c + "_h"] += int(k >= 1)
    return g

kp = lambda c: c["kept"] / max(c["n"], 1)
hr = lambda c, cl: c[cl + "_h"] / max(c[cl + "_n"], 1)
def pooled(g, langs, cl="sure"):
    h = sum(g[l][cl + "_h"] for l in langs); n = sum(g[l][cl + "_n"] for l in langs)
    return h / max(n, 1), n

def forget(tag):
    f = os.path.join(TD, f"d02b_walk_{tag}.txt")
    out = {}
    if os.path.exists(f):
        for line in open(f):
            m = re.match(r"\s*(code|math): .*kept per check ([0-9.]+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out

ARMS = [a for a in os.environ.get("E15_ARMS", "e1").split(",") if a]
B = {st: stats(f"d02b_walk_d02a_{st}.npz") for st in ("multi", "ifeval")}
R = {}
for arm in ARMS:
    if all(os.path.exists(os.path.join(TD, f"d02b_walk_{arm}_{st}.npz")) for st in ("multi", "ifeval")):
        R[arm] = {st: stats(f"d02b_walk_{arm}_{st}.npz") for st in ("multi", "ifeval")}
say(f"E1 / E5 read-out: held-out D-02a answers (multi 24, IFEval 30), block 16; L1-20 vs {', '.join(R) or '(no arm walked yet)'}")
cols = ["L1-20"] + list(R)
G = {"L1-20": B, **R}
for st in ("multi", "ifeval"):
    say(f"  {st}: kept per check " + " | ".join(f"{c} {kp(G[c][st]['all']):.2f}" for c in cols))
    for cl in ("sure", "mid", "tie"):
        say(f"    depth-1 hit, {cl:4s}: " + " | ".join(f"{c} {hr(G[c][st]['all'], cl):.3f} (n {G[c][st]['all'][cl + '_n']})" for c in cols))
langs = sorted(k for k in B["multi"] if k != "all")
say("  multi by language: kept / sure-token depth-1 hit")
for l in langs:
    say(f"    {l}: " + " | ".join(f"{c} {kp(G[c]['multi'][l]):.2f} / {hr(G[c]['multi'][l], 'sure'):.3f}" for c in cols))
for arm in R:
    fg = forget(arm)
    if fg:
        say(f"  {arm} forgetting (code / math kept; L1-20 9.57 / 7.00, D-13's 21-step drift 9.40 / 6.71): "
            + " / ".join(f"{k} {v:.2f}" for k, v in sorted(fg.items())))
DRJ = ("de", "ru", "ja")
b_drj, n_drj = pooled(B["multi"], DRJ)
for arm, g in R.items():
    a_drj, _ = pooled(g["multi"], DRJ)
    dk = kp(g["multi"]["all"]) - kp(B["multi"]["all"])
    ds = hr(g["multi"]["all"], "sure") - hr(B["multi"]["all"], "sure")
    dmid = hr(g["ifeval"]["all"], "mid") - hr(B["ifeval"]["all"], "mid")
    dki = kp(g["ifeval"]["all"]) - kp(B["ifeval"]["all"])
    if arm.startswith("e1"):
        v = "PASS" if a_drj >= 0.85 and dk >= 0.5 else "KILL" if a_drj - b_drj < 0.05 else "BETWEEN"
        say(f"  {arm} (E1 line): de/ru/ja sure-token depth-1 hit {b_drj:.3f} -> {a_drj:.3f} ({a_drj - b_drj:+.3f}; n {n_drj}), "
            f"multi kept {dk:+.2f} -> {v}")
    else:
        v = ("PASS" if hr(g["multi"]["all"], "sure") >= 0.75 or hr(g["ifeval"]["all"], "mid") >= 0.62
             else "KILL" if ds < 0.1 and dki < 0.1 else "BETWEEN")
        say(f"  {arm} (E5 line): multi sure-token depth-1 hit {hr(B['multi']['all'], 'sure'):.3f} -> "
            f"{hr(g['multi']['all'], 'sure'):.3f} ({ds:+.3f}); IFEval mid-token hit {hr(B['ifeval']['all'], 'mid'):.3f} -> "
            f"{hr(g['ifeval']['all'], 'mid'):.3f} ({dmid:+.3f}); IFEval kept {dki:+.2f} -> {v}")
        e1 = next((x for x in R if x.startswith("e1")), None)
        if e1:
            say(f"    vs {e1} (same data, no experts): multi sure-hit {hr(g['multi']['all'], 'sure') - hr(R[e1]['multi']['all'], 'sure'):+.3f}, "
                f"IFEval mid-hit {hr(g['ifeval']['all'], 'mid') - hr(R[e1]['ifeval']['all'], 'mid'):+.3f}, "
                f"multi kept {kp(g['multi']['all']) - kp(R[e1]['multi']['all']):+.2f}")
open(os.path.join(TD, "e15_read.txt"), "a").write("\n".join(lines) + "\n\n")
