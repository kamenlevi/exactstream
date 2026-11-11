"""DAY9 bytes side (Breakthrough, Mon 2026-11-02): the map's capture in MoonShots' replay format (MoonShots 10:4x).
One npz per window in out/moonshots/<window>.npz:
- tokens int32 [T]; gen int = the first answer position (ids[gen] is the first answer token; the first check's anchor is
  row gen - 1, the last prompt token); domain str; drafter 'dfl2' (DFlash2 L1-20, the shipped default on every type).
- ids int16 [T - 1, 48, 10]: every row's top-10 expert ids (row j predicts token j + 1), RANK-SORTED by the block's own
  router (the bare model's 8-bit gate, D-02a configuration), teacher-forced along the 125B's greedy answer.
- conf float32 [T - 1]: DFlash2's cf for the draft row j verifies (dm_gate.py; NaN = none: prompt rows, and the bonus
  row after 15 kept drafts).
- checks int32 [M, 3]: (first row, rows after the t 0.3 cut, kept), the cut applied inside the HPD-0 walk's checks;
  checks_walk int32 [M, 3]: (first row, rows uncut = 1 + checkable drafts, kept uncut); dcf float32 [M, 15]: each
  check's 15 draft cf (own picks, also past the first miss) to rebuild either. The walk's anchors follow the UNCUT
  acceptance: where the cut ends a check before its first miss (code 8.4 % of checks, 2.2 % of kept rows; IFEval 6.8 /
  4.2 %; multi 4.1 / 3.3 %) the live engine would start its next check earlier.
- p1 float32 [T - 1]: the 125B's own top-1 probability (baseline), NaN outside the measured rows (rows >= gen)."""
import json, os
import numpy as np
import dm_gate
OUT = dm_gate.OUT; TD = dm_gate.TD
os.makedirs(os.path.join(OUT, "moonshots"), exist_ok=True)
G = dm_gate.load(); base = json.load(open(os.path.join(OUT, "baseline.json")))
n_ok = 0
for dom, wf in dm_gate.WALK.items():
    Z, M = np.load(os.path.join(TD, wf)), np.load(os.path.join(TD, dm_gate.MAPW[dom]))
    zn = [str(x) for x in Z["names"]]
    for s, nm in enumerate([str(x) for x in M["names"]][:12]):
        i = zn.index(nm); ids = Z[f"{i}_ids"].astype(np.int32); T = len(ids); gen0 = int(Z[f"{i}_gen"])
        R = np.load(os.path.join(OUT, "routing", nm))
        assert np.array_equal(R["tokens"], ids) and R["ids"].shape == (T - 1, 48, 10), nm
        ch, dcf = Z[f"{i}_checks"], Z[f"{i}_dcf"].astype(np.float32)
        cut = []
        for (k, m, A), c in zip(ch, dcf):
            run = np.cumprod(c[:m]); L = int(np.argmax(run < 0.3)) if (run < 0.3).any() else int(m)
            cut.append((A, L + 1, min(k, L) + 1))
        b = base[nm]; a = b["a"]; p1 = np.full(T - 1, np.nan, np.float32)
        p1[a:a + len(b["lp"])] = np.exp(np.array(b["lp"])[:, 0])
        np.savez(os.path.join(OUT, "moonshots", nm), tokens=ids, gen=np.int64(gen0 + 1), domain=dom, drafter="dfl2",
                 ids=R["ids"].astype(np.int16), conf=G[nm].astype(np.float32), checks=np.array(cut, np.int32),
                 checks_walk=np.array([(A, m + 1, k + 1) for k, m, A in ch], np.int32), dcf=dcf, p1=p1)
        n_ok += 1
print(f"exported {n_ok} windows to {os.path.join(OUT, 'moonshots')}")
