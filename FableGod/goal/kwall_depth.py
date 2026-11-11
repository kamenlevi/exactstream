"""kwall_depth.py: the drafter's accuracy by depth and by the class of the token (every-position walks). python -I."""
import importlib.util, numpy as np, sys
spec = importlib.util.spec_from_file_location("kwr", "kwall_replay.py"); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
sets = K.load_sets()
INERT = ("comment", "docstring")
for label, S in sets:
    acc = {g: np.zeros(15) for g in ("code", "comment", "docstring", "other")}
    cnt = {g: np.zeros(15) for g in acc}
    sure = {g: [np.zeros(15), np.zeros(15)] for g in ("sure", "mid", "low")}
    for x in S:
        ids, gen = x["ids"], x["gen"]
        for i, (k, m, a) in enumerate(x["ch"]):
            m, a = int(m), int(a)
            if a < gen:
                continue
            for j in range(m):
                p = a + 1 + j
                c = x["fine"][p][0]
                g = c if c in INERT else "other" if c in ("prose", "fence") else "code"
                hit = int(x["tt"][i][j]) == ids[p]
                acc[g][j] += hit; cnt[g][j] += 1
                if g == "code":
                    p1 = K.p1_of(x, p)
                    if p1 is not None:
                        b = "sure" if p1 >= 0.99 else "mid" if p1 >= 0.5 else "low"
                        sure[b][0][j] += hit; sure[b][1][j] += 1
    print(f"\n=== {label}: teacher-forced accuracy of DFlash2 by depth (the pick on the right prefix)")
    for g in acc:
        if cnt[g].sum():
            print(f"  {g:10s} n at depth 1 {int(cnt[g][0]):5d}: " + " ".join(f"{v:.2f}" for v in acc[g] / np.maximum(cnt[g], 1)))
    for b in sure:
        if sure[b][1].sum():
            print(f"  code, 125B {b:4s} (p1 >= 0.99 / 0.5-0.99 / < 0.5) n {int(sure[b][1][0]):5d}: " + " ".join(f"{v:.2f}" for v in sure[b][0] / np.maximum(sure[b][1], 1)))
    tot = sum(sure[b][1][0] for b in sure)
    if tot:
        print("  share of code tokens: " + ", ".join(f"{b} {100 * sure[b][1][0] / tot:.1f} %" for b in sure))
