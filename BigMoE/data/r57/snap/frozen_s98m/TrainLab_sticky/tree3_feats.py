"""tree3_feats (MTP Head, Tue 11-03, for MoonShots' M-23b): the MTP head's per-draft features from data/tree3_map, in the
map's coordinates. Coordinates (README there): a check at anchor A (the last known token) drafts depth j = 0..14, which is
the token at A + 1 + j = map row A + j. Every answer position is an anchor, so any loop's (A, j) can be looked up.
Features for the DFlash2 draft at (A, j) (teacher-forced along the true path, exact up to the first miss):
  flag   1.0 if DFlash2's own draft != the MTP head's top-1 there, else 0.0
  p1     the MTP head's top-1 probability
  pdraft the MTP head's probability of DFlash2's drafted token (its top-3 probability if the draft is in its top-3, else 0)
Usage: T = load("code"); f = feats(T, "tf_code_HumanEval_10", A, J) -> [J, 3] (depths 0..J-1; NaN rows where missing).
Self-test: python tree3_feats.py (checks the coordinates against the map's tokens and DFlash2's acceptance)."""
import os, glob
import numpy as np
ROOT = os.path.expanduser("~/TrainLab/data/tree3_map")


def load(st):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, st, "*.npz"))):
        z = np.load(f)
        A = z["A"]
        out[os.path.basename(f)[:-4]] = dict(at={int(a): c for c, a in enumerate(A)}, own=z["dfl_own"], t3=z["mtp_t3"],
                                             p3=z["mtp_p3"].astype(np.float32), true=z["true"], tokens=z["tokens"])
    return out


def feats(T, win, A, J):
    w = T[win]; c = w["at"].get(int(A))
    f = np.full((J, 3), np.nan, np.float32)
    if c is None:
        return f
    for j in range(min(J, w["own"].shape[1])):
        t3 = w["t3"][c, j]
        if t3[0] < 0:
            continue
        d = int(w["own"][c, j])
        hit = np.nonzero(t3 == d)[0]
        f[j] = (float(d != int(t3[0])), float(w["p3"][c, j, 0]), float(w["p3"][c, j, hit[0]]) if len(hit) else 0.0)
    return f


if __name__ == "__main__":
    for st in ("code", "ifeval", "multi"):
        T = load(st); n = ok = 0; fl = fm = nf = 0
        for win, w in T.items():
            for A, c in list(w["at"].items())[:400]:
                for j in range(15):
                    t = int(w["true"][c, j])
                    if t < 0:
                        break
                    n += 1; ok += int(w["tokens"][A + 1 + j]) == t       # coordinates: depth j at anchor A = token A + 1 + j
                    f = feats(T, win, A, j + 1)[j]
                    if np.isnan(f[0]):
                        continue
                    nf += 1; fl += f[0]; fm += f[0] * (int(w["own"][c, j]) != t)
        print(f"{st}: coordinate check {ok}/{n}; MTP disagrees with DFlash2's draft on {fl / max(nf, 1):.3f} of drafts, "
              f"and the draft is wrong on {fm / max(fl, 1):.3f} of those")
