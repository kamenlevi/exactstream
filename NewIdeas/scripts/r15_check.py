"""R15 / R15b MASK BIT-CHECK (NewIdeas, Thu 5 Nov 2026 07:0x; CPU only). Cloud3's C3-R15 (the chat route + its unsure
rows at top-8 + union members + resident) and C3-R15b (+ REACH-CUT 0.5, resident keep), built with dm_oea_masks.py
(parse() gained qom_<thr>_u<k>[_reach<tau>]) and Fable C's bonus gate at 0.5 (proxy g), as the measured route
qom_0.5_live_c147_bonus5g was. Checks per window:
  A. qom_0.5_u10_live_c147 == qom_0.5_live_c147 bit for bit (the parse addition is neutral at u 10);
  B. u8 vs the route's base: ranks 0-1 identical; on rows the base keeps whole (unsure rows), u8 drops only ranks 8-9;
     every row's change counted (information: sure rows move a little through the smaller unions and the LRU pool);
  C. u8 + reach 0.5 vs u8: ranks 0-1 identical; rows that changed keep their top-2 (information: how many rows the
     reach cut takes, and their kept ranks 2-9);
  D. each bonus gate vs its base: drops only; the gated rows are the route's 165 (the same rows its gate changed).
Writes ~/NewIdeas/data/dm_masks/check_r15.log."""
import os
import numpy as np
D = os.path.expanduser("~/NewIdeas/data/dm_masks")


def ld(a, nm): return np.load(os.path.join(D, a, nm))["keep"]


def main():
    names = sorted(x for x in os.listdir(os.path.join(D, "qom_0.5_u8_reach0.5_live_c147_bonus5g")) if x.endswith(".npz"))
    ok = True; A = 0; Bdrop89 = Bbad = Brows = Badd = Bsure = 0; Crows = Ccut = Cbad = 0; Dbad = 0; Dsame = 0
    for nm in names:
        base, u10 = ld("qom_0.5_live_c147", nm), ld("qom_0.5_u10_live_c147", nm)
        u8, rc = ld("qom_0.5_u8_live_c147", nm), ld("qom_0.5_u8_reach0.5_live_c147", nm)
        A += int(np.array_equal(base, u10))
        if not (np.array_equal(base[:, :, :2], u8[:, :, :2]) and np.array_equal(u8[:, :, :2], rc[:, :, :2])): Bbad += 1
        whole = base.all(axis=(1, 2)); ch = (base != u8).any(axis=(1, 2)); Brows += int(ch.sum())
        lost = base & ~u8
        Bdrop89 += int(lost[whole][:, :, 8:].sum()); Bbad += int(lost[whole][:, :, :8].sum())   # unsure rows: ranks 8-9 only
        Badd += int((u8 & ~base).sum()); Bsure += int((ch & ~whole).sum())
        chc = (u8 != rc).any(axis=(1, 2)); Crows += int(chc.sum())
        Ccut += int(((u8 & ~rc)[:, :, 2:]).sum()); Cbad += int((rc & ~u8)[:, :, :2].sum())
        rg = set(np.where((ld("qom_0.5_live_c147_bonus5g", nm) != base).any(axis=(1, 2)))[0].tolist())
        for a in ("qom_0.5_u8_live_c147", "qom_0.5_u8_reach0.5_live_c147"):
            b, g = ld(a, nm), ld(a + "_bonus5g", nm)
            Dbad += int((g & ~b).sum())
            Dsame += int(set(np.where((g != b).any(axis=(1, 2)))[0].tolist()) <= rg)
    L = [f"A: qom_0.5_u10_live_c147 == qom_0.5_live_c147 bit for bit in {A} / {len(names)} windows",
         f"B: u8 vs the route's base: rows changed {Brows} ({Bsure} of them sure rows, via unions / the pool); unsure rows "
         f"dropped {Bdrop89} rank 8-9 experts and {Bbad} others (ranks 0-7 / 0-1 mismatches; must be 0); experts added {Badd} "
         "(information: a lighter pool keeps other residents)",
         f"C: u8 + reach 0.5 vs u8: rows changed {Crows}; rank 2-9 experts the cut dropped {Ccut}; ranks 0-1 lost {Cbad} (must be 0)",
         f"D: bonus gates drop only: {Dbad == 0} (added {Dbad}); gated rows within the route's gated rows in {Dsame} / "
         f"{2 * len(names)} window-arms"]
    ok = A == len(names) and Bbad == 0 and Cbad == 0 and Dbad == 0 and Dsame == 2 * len(names)
    L.append(f"{len(names)} windows -> {'OK' if ok else 'FAIL'}")
    open(os.path.join(D, "check_r15.log"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
