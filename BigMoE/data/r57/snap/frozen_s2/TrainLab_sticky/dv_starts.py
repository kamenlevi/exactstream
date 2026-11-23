"""DIVERSE2 (MTP Head, Mon 11-02; REPORT 'DIVERSE2'): L1-20's live block starts on its own training data, simulated on the
CPU from the conf-out pass's right / wrong per anchor and depth (data/hpd/conf_l120_right/<key>.npy, [n, 31], NaN off-path):
the official loop at block 16 starts at the first answer anchor, accepts the drafts up to the first teacher-forced miss (up
to it the own and teacher-forced chains agree), and starts the next block right after it (a + k + 1). Writes the block
starts per window to data/hpd/starts_l120/<key>.npy and prints their share of the anchors."""
import glob, os
import numpy as np
D = os.path.expanduser("~/TrainLab/data/hpd/conf_l120_right")
OUT = os.path.expanduser("~/TrainLab/data/hpd/starts_l120"); os.makedirs(OUT, exist_ok=True)
L120 = os.path.expanduser("~/Leap/data/l120")
tot_st = tot_an = 0
for p in sorted(glob.glob(os.path.join(D, "*.npy"))):
    key = os.path.basename(p)[:-4]
    rt = np.load(p)
    z = np.load(os.path.join(L120, key + ".npz"))
    asst = z["asst"].astype(bool); n = rt.shape[0]
    anchors = [a for a in range(n) if asst[a] and asst[a + 1]]
    st, a = [], (anchors[0] if anchors else n)
    while a < n:
        row = rt[a, :15]
        if np.isnan(row[0]):                         # off-path or past the answer: move to the next anchor
            nxt = [x for x in anchors if x > a]
            if not nxt:
                break
            a = nxt[0]; continue
        st.append(a)
        k = 0
        while k < 15 and not np.isnan(row[k]) and row[k] == 1:
            k += 1
        a = a + k + 1
    np.save(os.path.join(OUT, key + ".npy"), np.array(st, np.int64))
    tot_st += len(st); tot_an += len(anchors)
print(f"dv_starts: {tot_st} live block starts over {tot_an} anchors ({100 * tot_st / max(tot_an, 1):.1f} %) in "
      f"{len(glob.glob(os.path.join(D, '*.npy')))} windows -> {OUT}")
