"""Gated bonus row masks for screen 1 of BYTES10.md section 5 (Fable C, Wed 4 Nov 2026). CPU, < 1 min per arm.
Takes an existing live-cut arm's keep masks (NewIdeas' data/dm_masks/<arm>/<window>.npz, keep [n, 48, 10] bool, rank-sorted
as out/routing) and gates the bonus row of every live segment (the segment's last row when all its drafts were kept: today
exact at 10 ranks): the row keeps ranks 0-1 plus the experts some other row of the block keeps at that layer (the block's
union members under the arm's own masks), when the gate passes. Gate: 'all' = every real bonus row (the price bound; the
QOM cf 0 case), or a cf threshold on the bonus row's own cf: proxy "g" (default since Wed 06:5x) = the gate file's cf for the
token the bonus row emits (what the engine has: the draft after the cut); proxy "next" = the next segment's first draft cf
(the 00:34 builds; one position too far, as NewIdeas noted at 04:0x). Dirs built with "g" carry a trailing g.
Rows after a check's first miss are stand-ins in the arms (row p + 97): a segment that ends in a miss has no real bonus row
and is left alone. QOM's resident keep is not reproduced for the bonus row (the gated row keeps ranks 0-1 + union only), so
on qom_* arms this is conservative: a PASS here is a PASS for QOM's own form.
Out: ~/FableGod/fablec/masks/<arm>_bonus[<thr>]/<window>.npz (keep), plus a count line. Usage (BigMoE venv):
  python bytes10_bonus_masks.py qom_0_live all ; python bytes10_bonus_masks.py q1u_0.9_live 0.9 [g|next]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map")); sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))   # (copy: bytes10_replay from fablec, read-only)
import dm_gate  # noqa: E402
from bytes10_replay import live_segments, OUT  # noqa: E402
SRC = os.path.expanduser("~/Breakthrough/data/m1/masks")   # Breakthrough copy (Sun 16:5x): my builder copy's masks
DST = os.path.expanduser("~/Breakthrough/data/m1/masks")   # Breakthrough copy: out into my repo, not fablec


def main(arm, gate, proxy="g"):
    thr = None if gate == "all" else float(gate)
    tag = f"{arm}_bonus" + ("" if thr is None else str(thr).replace("0.", "") + ("g" if proxy == "g" else ""))
    od = os.path.join(DST, tag); os.makedirs(od, exist_ok=True)
    base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
    tot = dict(bonus=0, gated=0, pairs_before=0, pairs_after=0, rows=0)
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            src = os.path.join(SRC, arm, nm)
            if not os.path.exists(src): raise SystemExit(f"missing {src}")
            keep = np.load(src)["keep"].copy(); ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids)
            assert keep.shape == ids.shape, (keep.shape, ids.shape)
            g = np.nan_to_num(G[nm], nan=-1.0)
            segs = []
            for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
                segs += live_segments(k0_, m0_, A0_, dc0_)
            for j, (k, m, A, dc) in enumerate(segs):
                last, fm = min(A + m, N - 1), min(A + k, N - 1)
                if A > N - 1 or k != m or A + m > N - 1: continue          # no real bonus row
                tot["bonus"] += 1
                if proxy == "next": nxt = segs[j + 1][3][0] if j + 1 < len(segs) and segs[j + 1][3] else -1.0
                else: nxt = float(g[A + m])
                if thr is not None and nxt < thr: continue
                tot["gated"] += 1
                rows = [p if p <= fm else (p + 97) % N for p in range(A, last + 1)]
                b = A + m
                for l in range(48):
                    u = set()
                    for p in rows:
                        if p == b: continue
                        u.update(ids[p, l][keep[p, l]].tolist())
                    old = keep[b, l].copy()
                    keep[b, l] = False; keep[b, l, :2] = True
                    for r in range(2, 10):
                        if ids[b, l, r] in u: keep[b, l, r] = True
                    tot["pairs_before"] += int(old.sum()); tot["pairs_after"] += int(keep[b, l].sum())
            tot["rows"] += N - base[nm]["a"]
            np.savez_compressed(os.path.join(od, nm), keep=keep)
    line = (f"{tag}: bonus rows {tot['bonus']}, gated {tot['gated']}; their (row, layer, rank) pairs {tot['pairs_before']} -> "
            f"{tot['pairs_after']} (x{tot['pairs_after'] / max(tot['pairs_before'], 1):.2f}); answer rows {tot['rows']}; masks in {od}")
    print(line); open(os.path.join(DST, "bonus_masks.txt"), "a").write(line + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "all", sys.argv[3] if len(sys.argv) > 3 else "g")
