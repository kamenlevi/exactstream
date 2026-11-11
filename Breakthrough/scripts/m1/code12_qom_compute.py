#!/usr/bin/env python3
"""What QOM0's resident experts add to the COMPUTE of a check (Breakthrough, Tue 10 Nov 2026 10:0x) [K counts from the scored
masks, P for any time]. QOM0 keeps, on top of S1u's cut (ranks 0-1 + the check's union members), every rank 2-9 expert that
is resident in the base pool. Those cost no SSD reads (so the byte replays price QOM0 = S1u), but the engine still applies
them: more (row, expert) pairs and more distinct experts per layer, read from RAM. Fit12's next build is QOM0 at 51 + 32
with a lookup-only probe, then cutting S1u's measured ~102 ms non-read; this counts what QOM0 adds to that compute side.
Source of truth = the masks the decision map scored (keep[p, layer, rank] per window): qom_0_live_c51 (code PASS 1.16 %),
qom_0_live_c47, q1u_0_live (= S1u's cut), and EXACT (all 10). Checks = bytes10_core.live_blocks on the 12 code windows (the
replays' live checks, t 0.3), rows looked up in the routing ids (out/routing/<window>).
Two views: all rows of each check (as the replays count them), and real rows only. A check's first kept rows (A .. the
first mismatch) are real positions; the rest stand in for wrong drafts at (p + 97) % N, whose masks are that other
position's (a prompt row there keeps all 10 in every arm), so the real-rows view is the cleaner QOM0 / S1u ratio.
Per check: (row, expert) pairs kept, and distinct experts summed over the 48 layers (each distinct expert-layer is one
3-bit MSB record, 2,162,688 B, applied from RAM).
Output: ~/Breakthrough/data/m1/code12_qom_compute.out. Usage: nice -n 15 <python with numpy> -I -B code12_qom_compute.py
"""
import os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import code12_replay as R  # noqa: E402  (bytes10_core with the replays' rules; nothing is written by the import)

B = R.B
t0 = time.time(); lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


H = os.path.expanduser("~/Breakthrough/data")
ARMS = {"EXACT (all 10)": None, "q1u_0_live (S1u)": f"{H}/dm_masks/q1u_0_live",
        "qom_0_live_c47": f"{H}/m1/masks/qom_0_live_c47", "qom_0_live_c51": f"{H}/m1/masks/qom_0_live_c51"}
VIEWS = ("all rows (the replays' checks, stand-ins included)", "real rows only (the first kept rows of each check)")
W = B.windows()["code"]
tot = {v: {k: dict(pairs=0, dist=0, nchk=0, rows=0) for k in ARMS} for v in VIEWS}
per_layer_max = {v: {k: 0 for k in ARMS} for v in VIEWS}
for w in W:
    ids = w["ids"]                                           # [N, 48, 10]
    masks = {}
    for k, d in ARMS.items():
        if d is None:
            masks[k] = np.ones(ids.shape, bool); continue
        m = np.load(os.path.join(d, w["nm"]))["keep"].astype(bool)
        assert m.shape == ids.shape, (k, w["nm"], m.shape, ids.shape)
        masks[k] = m
    for rows, cfs, kp in B.live_blocks(w, None):
        for v, rr in zip(VIEWS, (rows, rows[:kp])):
            E = ids[rr]
            for k, m in masks.items():
                M = m[rr]
                t = tot[v][k]; t["pairs"] += int(M.sum()); t["nchk"] += 1; t["rows"] += len(rr)
                for l in range(48):
                    n = len(np.unique(E[:, l][M[:, l]])); t["dist"] += n
                    per_layer_max[v][k] = max(per_layer_max[v][k], n)
say(f"QOM0's compute per check vs S1u (code windows {len(W)}; live checks t 0.3 as the replays; masks = the scored map arms)")
for v in VIEWS:
    say(""); say(f"{v}:")
    say("| arm | checks | rows per check | (row, expert) pairs per check | distinct expert-layers per check | vs S1u | max distinct in a layer |")
    say("|---|---|---|---|---|---|---|")
    s = tot[v]["q1u_0_live (S1u)"]
    for k, t in tot[v].items():
        pc, dc = t["pairs"] / t["nchk"], t["dist"] / t["nchk"]
        say(f"| {k} | {t['nchk']} | {t['rows'] / t['nchk']:.2f} | {pc:.1f} | {dc:.1f} | pairs {pc / (s['pairs'] / s['nchk']) - 1:+.0%}, "
            f"distinct {dc / (s['dist'] / s['nchk']) - 1:+.0%} | {per_layer_max[v][k]} |")
    q = tot[v]["qom_0_live_c51"]
    extra = q["dist"] / q["nchk"] - s["dist"] / s["nchk"]
    say(f"qom_0_live_c51 adds {extra:.1f} distinct expert-layers per check over S1u = {extra * B.MR / 1e9:.2f} GB of 3-bit MSB "
        f"records applied from RAM per check: {extra * B.MR / 200e9 * 1e3:.1f} ms at 200 GB/s, {extra * B.MR / 300e9 * 1e3:.1f} ms at "
        f"300 GB/s [P: a bandwidth-bound price; the engine's own per-expert cost decides]")
say(f"done in {time.time() - t0:.0f} s")
open(os.path.join(H, "m1", "code12_qom_compute.out"), "w").write("\n".join(lines) + "\n")
