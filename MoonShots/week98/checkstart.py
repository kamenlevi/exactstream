"""WEEK98 K (MoonShots, Sun 15 Nov 2026), as pre-registered in LEVERS_A_MOONSHOTS.md: the audit at the decode walk's check
starts. kept/32 under the walk (certify98's kept_walk logic from row 1, 32-row checks) with P audit passes per check (one
repair per pass), and walk-strict accuracy at depths 0 / 1 / 2. Layout: certify98.windows() (EvalSuite, imported read-only)
on cells files whose `right` is the head's or the audited outcome. K0 must reproduce certify98's t30b kept/32 (MBPP 6.924,
HE+24 13.769). K2's depth-0 threshold is chosen on MBPP windows 0-49 only, then tested on 50-99 and HE+.
usage: ~/BigMoE/.venv/bin/python -I checkstart.py"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/EvalSuite/tools")); sys.path.insert(0, H("~/MoonShots/week98"))
import certify98 as CF                                    # noqa: E402
import chooser as CH                                      # noqa: E402

TMP = "/private/tmp/claude-501/-Users-kamen-MoonShots/4311ccb7-fdc5-4dac-ace8-b31665971f5f/scratchpad"
CELLS = {"mbpp": H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz"),
         "heplus": H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus24.npz")}


def per_cell(name, rows_tag):
    """(depth, head right, auditor right, auditor margin) in the cells file's own order"""
    if name == "mbpp":
        c = CH.mbpp_cells()
    else:
        Z = np.load(CELLS["heplus"], allow_pickle=True)
        c = dict(depth=Z["depth"], right=Z["right"].astype(bool), fi=Z["fi"], row=Z["row"],
                 an=Z["row"] - Z["depth"] - 1, files=[os.path.basename(str(f)) for f in Z["files"]])
    c = CH.attach_2b(c, H(f"~/MoonShots/week98/rows{rows_tag}_{name}.npz"), lambda t: 5)
    return c


def mats(name, right_arr, tag):
    Z = dict(np.load(CELLS[name], allow_pickle=True))
    Z["right"] = right_arr.astype(np.int8)
    p = os.path.join(TMP, f"cells_{name}_{tag}.npz"); np.savez(p, **{k: Z[k] for k in ("depth", "right", "m", "fi") if k in Z})
    return [rt for rt, _ in CF.windows(p)]


def walk(H_, C_, P, wsel):
    """kept per check and walk-strict num / den at depths 0-2; H_ / C_: per window [n0, 31] head / audited (1 / 0 / -1)"""
    kept, num, den = [], np.zeros(3), np.zeros(3)
    for w in wsel:
        h, c = H_[w], C_[w]; p = CF.R0
        while p < h.shape[0]:
            k, used = 0, 0
            while k < 31:
                hv, cv = h[p, k], c[p, k]
                if hv == -1:
                    if k < 3:
                        den[k] += 1
                    break
                if P is not None and used >= P:
                    ok = hv == 1
                elif hv == 1 and cv == 1:
                    ok = True
                elif hv == 0 and cv == 1:
                    ok = True; used += 1
                else:
                    ok = False
                if k < 3:
                    den[k] += 1; num[k] += ok
                if not ok:
                    break
                k += 1
            kept.append(k + 1); p += k + 1
    return dict(kept32=round(float(np.mean(kept)), 3),
                strict_0_1_2=[round(100 * float(n / max(d, 1)), 2) for n, d in zip(num, den)], checks=len(kept))


def main():
    res = {}
    for aud in ("2b", "4b"):
        for name in ("mbpp", "heplus"):
            c = per_cell(name, aud)
            Hm = mats(name, c["right"], "head")
            nw = len(Hm); wtr = range(min(50, nw)) if name == "mbpp" else range(nw); wte = range(50, nw) if name == "mbpp" else range(nw)
            for tau0 in (4.0, 2.0, 1.0, 0.5):
                sched = dict(CH.SCHED); sched[0] = tau0
                tau = np.array([sched[int(d)] for d in c["depth"]], np.float32)
                comb = np.where(c["m2b"] >= tau, c["r2b"], c["right"])
                Cm = mats(name, comb, f"{aud}_t{tau0}")
                for P in (1, 2):
                    for split, ws in (("all", range(nw)), ("train", wtr), ("test", wte)):
                        if name == "heplus" and split != "all":
                            continue
                        key = f"{aud} {name} {split} tau0={tau0:g} P={P}"
                        res[key] = walk(Hm, Cm, P, ws)
            for split, ws in (("all", range(nw)), ("train", wtr), ("test", wte)):
                if name == "heplus" and split != "all":
                    continue
                res[f"head {name} {split}"] = walk(Hm, Hm, 0, ws)
    for k, v in res.items():
        print(f"{k:42s} kept/32 {v['kept32']:7.3f}  walk-strict d0/d1/d2 {v['strict_0_1_2']}  checks {v['checks']}")
    json.dump(res, open(H("~/MoonShots/week98/checkstart.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
