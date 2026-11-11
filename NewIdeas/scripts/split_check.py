"""SPLIT MASK BIT-CHECK (NewIdeas, Thu 5 Nov 2026 07:3x; CPU only). Fable C's audit test 1 arms in the map's qom_0_live_bq form
(dm_oea_masks.py qom_<thr>_split<cf>: drafted rows with cf >= the split keep ranks 0-1 + union members only; loads unchanged).
  A. qom_0_split1.1_live_bq == qom_0_live_bq bit for bit (the addition is neutral when no row is strict);
  B. each split arm vs qom_0_live_bq: ranks 0-1 identical; experts dropped / added (added = information: strict rows stop
     touching residents, so the LRU drifts a little for later rows);
  C. nesting: rows the 0.99 split changed that the 0.9 split did not (information; ~0 expected).
Writes ~/NewIdeas/data/dm_masks/check_split.log."""
import os, sys
import numpy as np
D = os.path.expanduser("~/NewIdeas/data/dm_masks")
F = sys.argv[1] if len(sys.argv) > 1 else "bq"            # Thu 07:3x: "ball" = the DRAFT17 form (twin qom_0_live_ball)


def ld(a, nm): return np.load(os.path.join(D, a, nm))["keep"]


names = sorted(x for x in os.listdir(os.path.join(D, f"qom_0_split0.9_live_{F}")) if x.endswith(".npz"))
A = 0; r01 = 0; S = {a: [0, 0, 0] for a in (f"qom_0_split0.99_live_{F}", f"qom_0_split0.9_live_{F}")}; nest = 0
for nm in names:
    b = ld(f"qom_0_live_{F}", nm); A += int(np.array_equal(b, ld(f"qom_0_split1.1_live_{F}", nm)))
    ch = {}
    for a in S:
        x = ld(a, nm); r01 += int(not np.array_equal(x[:, :, :2], b[:, :, :2]))
        S[a][0] += int((b & ~x).sum()); S[a][1] += int((x & ~b).sum()); ch[a] = (x != b).any(axis=(1, 2)); S[a][2] += int(ch[a].sum())
    nest += int((ch[f"qom_0_split0.99_live_{F}"] & ~ch[f"qom_0_split0.9_live_{F}"]).sum())
L = [f"A: qom_0_split1.1_live_{F} == qom_0_live_{F} bit for bit in {A} / {len(names)} windows"]
for a, (dr, ad, rw) in S.items():
    L.append(f"B: {a} vs qom_0_live_{F}: rows changed {rw}; experts dropped {dr}, added {ad} (LRU drift; information)")
L.append(f"B: windows with a ranks 0-1 mismatch: {r01} (must be 0)")
L.append(f"C: rows the 0.99 split changed but the 0.9 split did not: {nest} (information)")
ok = A == len(names) and r01 == 0
L.append(f"{len(names)} windows -> {'OK' if ok else 'FAIL'}")
open(os.path.join(D, "check_split.log" if F == "bq" else f"check_split_{F}.log"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
