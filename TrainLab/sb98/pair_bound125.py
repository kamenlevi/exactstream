"""WEEK98 S21 PAIR stage 0 on the 125B (MTP Head; added after the sandbox bound, which s30's lower confidence may
understate): the same perfect-pairing level bound on t30b's TOPK dumps (BigMoE ceiling dir, read-only;
topk_t30b_{heplus24,mbppbank}.npz: top-8 probs along the true path per anchor and depth). Walk = certify98's (R0 = 1).
Pessimistic = pairs only inside the accepted run; optimistic = at every depth (teacher-forced p1). D in (12, 20, 31),
tau in (0.95, 0.9). Usage: BigMoE venv python -I -B pair_bound125.py OUT.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite/tools"))
import certify98 as C98                                     # noqa: E402

CE = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
R = {}
for st, tag in (("mbpp", "mbppbank"), ("he", "heplus24")):
    tz = np.load(f"{CE}/topk_t30b_{tag}.npz")
    dep, right, tp = tz["depth"].astype(int), tz["right"].astype(int), tz["top_p"].astype(np.float32)
    s0 = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(s0[1:], len(dep))
    W = []
    for s, t in zip(s0, en):
        dw = dep[s:t]; n0 = int((dw == 0).sum())
        rt = np.full((n0 + 1, 31), -1, np.int8); P1 = np.zeros((n0 + 1, 31), np.float32)
        for j in range(31):
            m = dw == j; c = int(m.sum())
            rt[:c, j] = right[s:t][m]; P1[:c, j] = tp[s:t][m][:, 0]
        W.append((rt, P1))
    for mode in ("pess", "opt"):
        for tau in (0.95, 0.9):
            for D in (12, 20, 31):
                saved = []
                for rt, P1 in W:
                    p = C98.R0
                    while p < rt.shape[0]:
                        acc = int(np.argmin(np.append(rt[p] == 1, False)))
                        j, pairs = 1, 0
                        while j <= D:
                            if (mode == "opt" or j <= acc) and j + 1 <= D and P1[p, j - 1] >= tau:
                                pairs += 1; j += 2
                            else:
                                j += 1
                        saved.append(pairs / D); p += acc + 1
                R[f"{st}|{mode}|tau{tau}|D{D}"] = float(np.mean(saved))
    print(st, " ".join(f"{k.split('|', 1)[1]} {100 * v:.1f}%" for k, v in R.items() if k.startswith(st) and ("D20" in k or "D12" in k)))
ok = all(R[f"{st}|pess|tau0.95|D20"] >= 0.20 for st in ("mbpp", "he"))
print("125B stage-0 line (pessimistic, tau 0.95, D 20, both sets >= 20 %): " + ("GO" if ok else "KILL"))
json.dump(R, open(sys.argv[1], "w"), indent=1)
