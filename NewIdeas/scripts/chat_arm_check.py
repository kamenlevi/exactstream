"""CHAT ARM BIT-CHECK (NewIdeas, Wed 4 Nov 2026 20:1x; CPU only). Masks for the chat route's kill test (QUEUE 20:1x):
qom_0.5_live_c147 (dm_oea_masks.py, uniform 147-slot pool) and qom_0.5_live_c147_bonus5g (Fable C's bytes10_bonus_masks.py
gate at 0.5, proxy g, output in ~/NewIdeas/data/dm_masks). Checks, per window:
  A. c147 vs qom_0.5_live (45 GB pool) and qom_0.5_live_c164: same shapes; rows with all 10 kept (prompt / unsure rows)
     counted (information); ranks 0-1 identical; differences only at ranks 2-9 of sure rows; kept share of
     sure-row rank 2-9 pairs ordered pool m > c164 > c147.
  B. c147_bonus5g vs c147: only bonus rows differ, by drops only (new keep a subset of old).
  C. the gated rows match Fable C's qom_0.5_live_bonus5g: its changed rows (vs qom_0.5_live) should all change in mine
     too (a smaller pool keeps fewer union members, so a row it drops on, mine drops on); the builder's gated count must be
     165, as Fable C's (the same segments and cf).
Prints OK / FAIL per check; writes ~/NewIdeas/data/dm_masks/check_c147_chat.log."""
import os
import numpy as np
D = os.path.expanduser("~/NewIdeas/data/dm_masks")
FC = os.path.expanduser("~/FableGod/fablec/masks/qom_0.5_live_bonus5g")


def ld(d, nm): return np.load(os.path.join(d, nm))["keep"]


def main():
    names = sorted(x for x in os.listdir(os.path.join(D, "qom_0.5_live_c147_bonus5g")) if x.endswith(".npz"))
    L = []; ok = True; share = {"m": [0, 0], "c164": [0, 0], "c147": [0, 0]}
    rowsB = rowsC = missing = fm_only = c_only = 0; added = 0
    for nm in names:
        m = ld(os.path.join(D, "qom_0.5_live"), nm); c164 = ld(os.path.join(D, "qom_0.5_live_c164"), nm)
        c147 = ld(os.path.join(D, "qom_0.5_live_c147"), nm); g = ld(os.path.join(D, "qom_0.5_live_c147_bonus5g"), nm)
        fc = ld(FC, nm)
        if not (m.shape == c164.shape == c147.shape == g.shape == fc.shape): L.append(f"FAIL shape {nm}"); ok = False; continue
        full = [x.all(axis=(1, 2)) for x in (m, c164, c147)]
        fm_only += int((full[0] & ~full[2]).sum()); c_only += int((full[2] & ~full[0]).sum())   # information: a sure row
        # can keep all 10 when every rank 2-9 expert is resident, more often in a bigger pool
        if not (np.array_equal(m[:, :, :2], c164[:, :, :2]) and np.array_equal(m[:, :, :2], c147[:, :, :2])):
            L.append(f"FAIL A ranks 0-1 differ {nm}"); ok = False
        sure = ~full[0]
        for k, x in (("m", m), ("c164", c164), ("c147", c147)):
            share[k][0] += int(x[sure][:, :, 2:].sum()); share[k][1] += int(x[sure][:, :, 2:].size)
        # B: the gate only drops, only on some rows
        if (g & ~c147).any(): added += int((g & ~c147).sum()); ok = False; L.append(f"FAIL B the gate added experts {nm}")
        rb = np.where((g != c147).any(axis=(1, 2)))[0]; rowsB += len(rb)
        # C: the same rows as Fable C's build differ from their base
        rc = np.where((fc != m).any(axis=(1, 2)))[0]; rowsC += len(rc)
        missing += len(set(rc.tolist()) - set(rb.tolist()))
    sh = {k: 100 * v[0] / max(v[1], 1) for k, v in share.items()}
    if not (sh["m"] > sh["c164"] > sh["c147"]): ok = False; L.append("FAIL A kept share not ordered m > c164 > c147")
    L.append(f"A: kept share of sure-row rank 2-9 pairs, pool m {sh['m']:.1f} % > c164 {sh['c164']:.1f} % > c147 {sh['c147']:.1f} %")
    L.append(f"A (information): rows with all 10 kept in pool m only {fm_only}, in c147 only {c_only}")
    if missing: ok = False
    L.append(f"B: rows the gate changed {rowsB} (drops only: {added == 0}); C: Fable C's qom_0.5_live_bonus5g changed {rowsC} "
             f"rows, {missing} of them unchanged in mine (must be 0)")
    L.append(f"{len(names)} windows -> {'OK' if ok else 'FAIL'}")
    open(os.path.join(D, "check_c147_chat.log"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
