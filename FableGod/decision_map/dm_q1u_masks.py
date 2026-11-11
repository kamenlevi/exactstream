"""Q1-UNION keep masks (Breakthrough, Mon 2026-11-02 23:2x; Fable C's AUDIT_350.md section 4, the orchestrator 23:2x;
pre-registered SCOREBOARD a6dd04e). On drafter-sure rows (the realizable gate dm_gate: cf >= theta) keep ranks 0-1 and every
rank 2-9 expert already in the CHECK's union at that layer: all ranks of the block's unsure rows, ranks 0-1 of its sure
rows. Blocks = the HPD-0 walks' 16-row checks (rows A .. A + m); the rows after a check's first miss enter the union with
the wrong-draft stand-in (row p + 97's routing) and their own drafts' cf (dcf). Unsure rows, prompt rows and rows outside
the walk keep all 10. Out: out/q1u_cf<thr>/<window>.npz keep [n - 1, 48, 10] bool, rank-sorted as DAY9's routing capture;
with --resident also ORs in the LRU-C-256-resident experts (Q1-on-miss variant: not written unless asked).
Usage (BigMoE venv): python dm_q1u_masks.py 0.99 0.9"""
import json, os, sys
import numpy as np
import dm_gate
OUT = os.path.expanduser(os.environ.get("DM_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")))
base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()

import re
LIVE_T = 0.3
def live_segments(k, m, A, dc):
    """NewIdeas' live_segments (~/NewIdeas/scripts/dm_oea_masks.py, Tue 10:3x), copied: the live engine's running-confidence
    cut (t 0.3) on one walk check: drafts kept while the product of their cf >= 0.3; a cut before the first miss ends the
    check at the kept drafts' bonus row and the next check starts at the next row with the walk's remaining drafts."""
    segs, a0, kk, dd = [], A, k, [float(x) for x in dc[:m]]
    while True:
        L = int((np.cumprod(dd) >= LIVE_T).sum()) if dd else 0
        if L >= kk or L == len(dd):
            segs.append((kk, L, a0, dd)); return segs
        segs.append((L, L, a0, dd[:L]))
        a0, kk, dd = a0 + L + 1, kk - L - 1, dd[L + 1:]

def parse(arm):
    """q1u_cf<thr> / q1u <float> -> (thr, 2, 10); oea_k<k> -> (None, k, k); hyb_<thr>_u<k> -> (thr, 2, k) (NewIdeas' rules,
    ~/NewIdeas/scripts/dm_oea_masks.py: sure rows (cf >= thr) k0 ks, the others k0 ku, every row adds union members)"""
    m = re.fullmatch(r"oea_k(\d+)", arm)
    if m: return None, int(m.group(1)), int(m.group(1))
    m = re.fullmatch(r"hyb_([0-9.]+)_u(\d+)", arm)
    if m: return float(m.group(1)), 2, int(m.group(2))
    raise ValueError(arm)

def masks(thr, arm=None, t3=False, live=None, tau=None):
    """arm None: Q1-UNION at thr (out tag q1u_cf0<thr>); else an OEA / hybrid arm; t3: the arm's sure rows also drop
    ranks 0-1 from the mask (they come from records3 at 3 bits through dm_t3, the gate ("t3", thr)): tag <arm>_t3"""
    ks, ku = 2, 10
    if arm is not None: thr, ks, ku = parse(arm)
    if live is None: live = os.path.exists(os.path.join(OUT, "LIVE_CUT"))   # FLOOR night: touch <DM_OUT>/LIVE_CUT before step B
    tag = ("q1u_cf0" + str(thr)[2:] if arm is None else arm) + ("_t3" if t3 else "")
    if os.environ.get("MASK_TAG"): tag = os.environ["MASK_TAG"]   # e.g. q1u_0.9_live for a side-by-side check
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    st = {}
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids); g = np.nan_to_num(G[nm], nan=-1.0)
            keep = np.ones((N, 48, 10), bool); kept = drop = 0
            for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
              for (k, m, A, dc) in (live_segments(k0_, m0_, A0_, dc0_) if live else [(k0_, m0_, A0_, dc0_)]):
                last, fm = min(A + m, N - 1), min(A + k, N - 1)
                if A > N - 1: continue                            # a live segment past the window's end
                blk = []                                          # (routing row, k0)
                for j_, p in enumerate(range(A, last + 1)):
                    if p <= fm: c = -1.0 if (live and k == m and p == A + m) else g[p]   # a live cut's bonus row: exact
                    else: c = dc[p - A] if p - A < m else -1.0
                    k0c = ku if (thr is None or c < thr) else ks
                    if tau is not None and k0c == ku and (np.r_[1.0, np.cumprod(np.array(dc, float))][j_] if j_ <= len(dc) else 0.0) < tau:
                        k0c = ks                                  # REACH-CUT: an unlikely-reached unsure row keeps ks + members
                    blk.append((p if p <= fm else (p + 97) % N, k0c))
                for l in range(48):
                    u = set()
                    for p, k0 in blk: u.update(ids[p, l, :k0].tolist())
                    for j, p in enumerate(range(A, fm + 1)):
                        k0 = blk[j][1]
                        if k0 >= 10: continue
                        inu = np.array([e in u for e in ids[p, l, k0:].tolist()])
                        keep[p, l, k0:] = inu; kept += int(inu.sum()); drop += int((~inu).sum())
                        if t3 and k0 == ks and thr is not None: keep[p, l, :2] = False   # this check's sure rows (a live bonus row is not)
            np.savez(os.path.join(OUT, tag, nm), keep=keep)
            d = st.setdefault(dom, [0, 0]); d[0] += kept; d[1] += drop
    for dom, (kp, dr) in st.items():
        print(f"{tag} {dom}: of Q1's dropped rank 2-9 pairs on sure rows, {100 * kp / max(kp + dr, 1):.1f} % are union members "
              f"(kept), {kp + dr} pairs", flush=True)

if __name__ == "__main__":       # python dm_q1u_masks.py 0.99 0.9 | oea_k8 hyb_0.9_u8 ... | hyb_0.9_u8:t3
    for t in sys.argv[1:] or ["0.99", "0.9"]:
        a, _, x = t.partition(":")
        if re.fullmatch(r"[0-9.]+", a): masks(float(a), t3=(x == "t3"))
        else: masks(None, a, t3=(x == "t3"))
