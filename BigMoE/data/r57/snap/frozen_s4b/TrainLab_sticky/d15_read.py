"""D-15a read-out (MTP Head, Thu 10-29; REPORT 'D-15a'): L1-20 vs the multilingual rapid tune (L) on the 8 held-out
answers (one per language): kept per 16-row check and the depth-1 hit rate on tokens the 125B is >= 0.99 sure of, per
language and overall; plus code / math kept for forgetting (vs D-13's U drift 9.40 / 6.71).
Inputs: data/d02b_walk_d02a_multi.npz (L1-20, all 24), data/d02b_walk_d15L_mheld.npz, data/d02b_walk_d15L.txt.
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d15_read.py -> data/d15_read.txt"""
import collections, os
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def stats(Z, keep):
    g = collections.defaultdict(lambda: [0, 0, 0, 0])            # kept, checks, sure depth-1 hits, sure depth-1 trials
    for s, nm in enumerate(Z["names"]):
        nm = str(nm)
        if nm not in keep:
            continue
        P = np.load(os.path.join(TD, "d02a", nm)); lang = "".join(c for c in nm[len("tf_multi_"):-4] if c.isalpha())
        ids, off = Z[f"{s}_ids"], int(Z[f"{s}_off"])
        for k, m, a in Z[f"{s}_checks"]:
            k, m, a = int(k), int(m), int(a)
            for key in (lang, "all"):
                g[key][0] += k + 1; g[key][1] += 1
            r = a - int(P["off"])
            if m >= 1 and 0 <= r < P["tk"].shape[0] and int(P["tk"][r, 0]) == int(ids[a + 1 - off]) \
                    and np.exp(float(P["lp"][r, 0])) >= 0.99:
                for key in (lang, "all"):
                    g[key][2] += int(k >= 1); g[key][3] += 1
    return g

TAG = os.environ.get("D15_TAG", "d15L")                  # E5a (11-01): the same read-out for other arms' mheld walks
T = np.load(os.path.join(TD, f"d02b_walk_{TAG}_mheld.npz"))
keep = {str(x) for x in T["names"]}
b, t = stats(np.load(os.path.join(TD, "d02b_walk_d02a_multi.npz")), keep), stats(T, keep)
say(f"{'D-15a LANGPROBE' if TAG == 'd15L' else TAG}: 8 held-out multilingual answers; kept per check and depth-1 hits on >= 0.99-sure tokens")
say("lang | L1-20 kept | tuned kept | L1-20 sure-hit | tuned sure-hit")
for k in sorted(b, key=lambda x: (x == "all", x)):
    say(f"{k:4s} | {b[k][0] / max(b[k][1], 1):.2f} | {t[k][0] / max(t[k][1], 1):.2f} | "
        f"{b[k][2] / max(b[k][3], 1):.3f} ({b[k][3]}) | {t[k][2] / max(t[k][3], 1):.3f} ({t[k][3]})")
txt = open(os.path.join(TD, f"d02b_walk_{TAG}.txt")).read()
for line in txt.splitlines():
    if "kept per check" in line and ("code" in line or "math" in line):
        say("forgetting check (vs D-13 U 9.40 / 6.71): " + line.strip()[:110])
open(os.path.join(TD, "d15_read.txt" if TAG == "d15L" else f"d15_read_{TAG}.txt"), "w").write("\n".join(lines) + "\n")
