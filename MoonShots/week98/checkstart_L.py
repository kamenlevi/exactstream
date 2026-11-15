"""WEEK98 L (MoonShots, Sun 15 Nov 2026), as pre-registered in LEVERS_A_MOONSHOTS.md: a correction cache within the
answer, along the decode walk (certify98's kept_walk logic, 32-row checks, from row 1). When a check ends at a miss, it
remembers (previous token, drafted token) -> the 125B's token. At later cells of the same window, a draft whose key matches
is replaced by the cached correction. The drafted token on a miss is the 125B's top-32 entry at the cell's rank (the
cells carry rank on every cell; rank 99 = outside the top-32 -> unknown, no key).
LINE: LIVE if it fixes >= 10 % of the check-start depth-0 misses while breaking <= 0.5 % of the hits it touches.
usage: ~/BigMoE/.venv/bin/python -I checkstart_L.py"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98"))
import chooser as CH                                      # noqa: E402

BANK = H("~/datasets/deepdraft/bank")


def windows_of(name):
    """per window: file name, and per (anchor index, depth): right, row, rank, in the cells file's order"""
    if name == "mbpp":
        c = CH.mbpp_cells(); Z = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_mbpp100.npz"))
        rank = Z["rank"].astype(np.int64); files = c["files"]; fi, row, depth, right = c["fi"], c["row"], c["depth"], c["right"]
    else:
        Z = np.load(H("~/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus24.npz"), allow_pickle=True)
        rank = Z["rank"].astype(np.int64); files = [os.path.basename(str(f)) for f in Z["files"]]
        fi, row, depth, right = Z["fi"], Z["row"], Z["depth"], Z["right"].astype(bool)
    st = np.flatnonzero((depth == 0) & np.append(True, depth[:-1] != 0)); en = np.append(st[1:], len(depth))
    out = []
    for s, t in zip(st, en):
        n0 = int((depth[s:t] == 0).sum())
        R = np.full((n0, 31), -1, np.int8); RW = np.zeros((n0, 31), np.int64); RK = np.full((n0, 31), 99, np.int64)
        for j in range(31):
            idx = np.arange(s, t)[depth[s:t] == j]
            R[:len(idx), j] = right[idx].astype(np.int8); RW[:len(idx), j] = row[idx]; RK[:len(idx), j] = rank[idx]
        out.append((files[int(fi[s])], R, RW, RK))
    return out


def main():
    res = {}
    for name in ("mbpp", "heplus"):
        kept, num, den = [], np.zeros(3), np.zeros(3)
        cs_miss = cs_fixed = touched_hits = broken = fixes = 0
        for nm, R, RW, RK in windows_of(name):
            z = np.load(os.path.join(BANK, nm)); ids = np.asarray(z["ids"]); tk = np.asarray(z["tk"])
            cache = {}; p = 1
            while p < R.shape[0]:
                k = 0
                while k < 31:
                    h = R[p, k]
                    if h == -1:
                        if k < 3:
                            den[k] += 1
                        break
                    r = int(RW[p, k]); truth = int(ids[r + 1]); prev = int(ids[r])
                    d = truth if h == 1 else (int(tk[r][RK[p, k]]) if 0 <= RK[p, k] < 32 else -1)
                    ok, used = bool(h == 1), d
                    if d != -1 and (prev, d) in cache and cache[(prev, d)] != d:
                        used = cache[(prev, d)]; ok = used == truth
                        if h == 1:
                            touched_hits += 1; broken += int(not ok)
                        elif ok:
                            fixes += 1
                    if k == 0 and h == 0:
                        cs_miss += 1; cs_fixed += int(ok)
                    if k < 3:
                        den[k] += 1; num[k] += ok
                    if not ok:
                        if used != -1:
                            cache[(prev, used)] = truth          # the 125B's correction at this row (the bonus token)
                        break
                    k += 1
                kept.append(k + 1); p += k + 1
        res[name] = dict(kept32=round(float(np.mean(kept)), 3), strict_0_1_2=[round(100 * float(a / max(b, 1)), 2) for a, b in zip(num, den)],
                         checkstart_depth0_misses=cs_miss, fixed=cs_fixed, fixed_share=round(cs_fixed / max(cs_miss, 1), 3),
                         hits_touched=touched_hits, hits_broken=broken, fixes_any_depth=fixes,
                         line=("LIVE" if cs_fixed / max(cs_miss, 1) >= 0.10 and broken <= 0.005 * max(touched_hits, 1) else "not live"))
        print(name, res[name])
    json.dump(res, open(H("~/MoonShots/week98/checkstart_L.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
