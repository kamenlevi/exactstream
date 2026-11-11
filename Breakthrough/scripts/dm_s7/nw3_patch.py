"""NEVER-WAIT v3 (b) on the decision map (Breakthrough, Fri 2026-11-06 21:3x; NewIdeas QUEUE dc9aff8 S1 + 6fd0224; Leap's
TABLE_350 'NEVER-WAIT v3 staged', ~/Leap/data/nw3/ref2_cand.py): the patch that adds the nw3 arms to the map's harness.

v3 with never-wait on (Leap): a cold demand use -> late (nw1, as installed Fri 04:01); a cold QOM optional use -> dropped
(weight 0); a cold LSB sidecar whose base is resident -> 3 bits now (MSB only, U3's -2 sidecar slot).
  (a) needs nothing here: the QOM masks already drop a cold optional use (keep = 0, weight 0), and nw1's late flags lie
      within keep (FAR's check_masks.py), so no late flag sits on a dropped use.
  (b) cannot be modelled from residency (the map's pools hold whole experts; the engine's sidecar pool is separate), so
      the arm is its WORST CASE (NewIdeas): every UNSURE answer row (draft cf < thr, or no draft; the engine's
      R57_U3_THR default 0.9) has a cold sidecar on ranks 0-1, so those ranks run at 3 bits on time (dm_t3: records3's
      MSB-3 record, the engine's 3-bit read), except a late rank (a cold base), which stays 4-bit and is written one
      layer late as in nw1. Ranks 2-9 stay 4-bit, as in every map arm, so D against nw1 is (b) alone.
How: measure() gates the unsure answer rows (CUR['k'] = 0, so _t3_add serves them), CUR['rm3'] = keep & ~late is the
3-bit path's rank mask, and the stock path keeps a gated row's ranks 0-1 only where late. nw3 is the only arm that runs
never-wait with t3 and a gate; every other arm takes the old code path unchanged (test: old arms bit-equal).

Usage: nw3_patch.py <dir with the live dm_common.py + dm_ablations.py> <out dir>. Exit 0 = written (or already in),
1 = an anchor is missing or not unique (nothing written)."""
import os, sys

MARK = "NEVER-WAIT v3 (b)"
SRC, OUT = sys.argv[1], sys.argv[2]

EDITS_COMMON = [
    # 1. the never-wait state: a counter of the 3-bit terms served
    ('NW = {"sl": None, "stash": {}, "inj": {}, "writes": 0, "terms": 0}\n',
     'NW = {"sl": None, "stash": {}, "inj": {}, "writes": 0, "terms": 0, "t3": 0}   # t3: NEVER-WAIT v3 (b)\'s 3-bit terms (nw3)\n'),
    ('    NW.update(sl=None, stash={}, inj={}, writes=0, terms=0)\n',
     '    NW.update(sl=None, stash={}, inj={}, writes=0, terms=0, t3=0)\n'),
    # 2. the guard: nw3 (and only nw3) runs never-wait with t3 and its own gate
    ('    assert not any(PERT.get(k_) for k_ in ("t3", "gate", "reuse", "q", "topk")), "never-wait is not wired with t3 / gate / reuse / q / topk"\n',
     '    bad_ = ("reuse", "q", "topk") + (() if PERT.get("nw3") else ("t3", "gate"))   # NEVER-WAIT v3 (b): nw3 adds t3 + the t3u gate\n'
     '    assert not any(PERT.get(k_) for k_ in bad_), "never-wait is not wired with t3 / gate / reuse / q / topk (nw3: with reuse / q / topk)"\n'
     '    assert not PERT.get("nw3") or (PERT.get("gate", (None,))[0] == "t3u" and PERT.get("t3") in (3, 4) and int(PERT.get("t3k", 2)) == 2\n'
     '                                   and not PERT.get("t3keep") and not PERT.get("t3L")), "nw3: gate (\'t3u\', thr), t3 3 or 4, t3k 2, no t3keep / t3L"\n'),
    # 3. the setup measure() calls per window (before _nw_stash, after _nw_late)
    ('def _nw_stash(li, delta):\n',
     'def _nw3_setup(g, a):\n'
     '    """NEVER-WAIT v3 (b), its worst case (Breakthrough, Fri 2026-11-06; nw3_patch.py): every UNSURE answer row (rows >= a\n'
     '    whose gate g < PERT[\'gate\'][1]; no draft is -1 in _gate) is gated (CUR[\'k\'] = 0: _t3_add serves its ranks 0-1 at\n'
     '    PERT[\'t3\'] bits from records3, on time); CUR[\'rm3\'] = keep & ~late is that path\'s rank mask, and the stock path keeps a\n'
     '    gated row\'s ranks 0-1 only where late (a cold base: 4-bit, written late as in nw1). Needs CUR[\'rm\'] and CUR[\'lm\']."""\n'
     '    n = len(g); u = np.zeros(n, bool); u[a:] = ~(g[a:] >= PERT["gate"][1])\n'
     '    kk = np.full(n, 10, np.int32); kk[u] = 0; CUR["k"] = kk\n'
     '    r, lm = CUR["rm"], CUR["lm"]; m = min(n, r.shape[0]); gr = np.flatnonzero(u[:m])\n'
     '    CUR["rm3"] = r & ~lm; r2 = r.copy(); r2[gr, :, :2] &= lm[gr, :, :2]; CUR["rm"] = r2\n'
     '\n'
     'def _nw_stash(li, delta):\n'),
    # 4. _t3_add: the gated rows' rank mask for nw3 (ranks 0-1 kept and not late), and the count
    ('        keep[ok_] = rm[gi[ok_], li]\n',
     '        keep[ok_] = rm[gi[ok_], li]\n'
     '    if PERT.get("nw3"):                                   # NEVER-WAIT v3 (b): the gated (unsure) rows\' ranks 0-1 at PERT[\'t3\']\n'
     '        r3 = CUR["rm3"]; keep = np.zeros((len(G), 10), bool); gi = p0 + G; ok_ = gi < r3.shape[0]   # bits, on time; not a late\n'
     '        keep[ok_] = r3[gi[ok_], li]; NW["t3"] += int(keep[:, :2].sum())     # rank (4-bit, written late on the stock path)\n'),
    # 5. clear(): no state leaks
    ('    CUR["lm"] = None; _nw_reset()\n',
     '    CUR["lm"] = None; CUR["rm3"] = None; _nw_reset()\n'),
    # 6. _gate_k: t3u is set in measure() (it needs the answer rows)
    ('    if spec[0] in ("t3mask", "t3keep"): return k        # set from the rank mask in measure()\n',
     '    if spec[0] in ("t3mask", "t3keep", "t3u"): return k   # set from the rank mask in measure() (t3u: NEVER-WAIT v3 (b), _nw3_setup)\n'),
    # 7. measure(): the nw3 setup per window, after the t3mask block
    ('            kk[:m][~r[:m, :, 0].any(axis=1)] = 0; CUR["k"] = kk                                 # rows are not served twice\n',
     '            kk[:m][~r[:m, :, 0].any(axis=1)] = 0; CUR["k"] = kk                                 # rows are not served twice\n'
     '        if PERT.get("nw3"): _nw3_setup(_gate(nm, len(ids)), a)   # NEVER-WAIT v3 (b): the unsure answer rows\' ranks 0-1 at 3 bits\n'),
    # 8. measure(): the 3-bit terms on record, checked against the masks
    ('                if NW["terms"] != nl_ or NW["stash"]:\n'
     '                    extra += f"; WARNING never-wait used {NW[\'terms\']} late terms, the mask holds {nl_} (rows misaligned)"\n'
     '                    res[nm]["warn"] = extra\n',
     '                if NW["terms"] != nl_ or NW["stash"]:\n'
     '                    extra += f"; WARNING never-wait used {NW[\'terms\']} late terms, the mask holds {nl_} (rows misaligned)"\n'
     '                    res[nm]["warn"] = extra\n'
     '                if PERT.get("nw3"):                         # NEVER-WAIT v3 (b): the 3-bit terms served == the masks\' count\n'
     '                    r3 = CUR["rm3"]; n3 = int(r3[np.flatnonzero(CUR["k"][:r3.shape[0]] == 0), :, :2].sum())\n'
     '                    res[nm]["nw"].update(t3=NW["t3"], t3rows=int((CUR["k"][a:len(ids) - 1] == 0).sum()), bits=int(PERT["t3"]))\n'
     '                    extra += f"; nw3: {NW[\'t3\']} rank 0-1 terms at {PERT[\'t3\']} bits on {res[nm][\'nw\'][\'t3rows\']} unsure rows"\n'
     '                    if NW["t3"] != n3:\n'
     '                        extra += f"; WARNING nw3 served {NW[\'t3\']} terms at {PERT[\'t3\']} bits, the masks hold {n3} (rows misaligned)"\n'
     '                        res[nm]["warn"] = extra\n'),
]

ABL_ADD = '''
# ---- NEVER-WAIT v3 (b) (Breakthrough, Fri 2026-11-06 21:3x; NewIdeas QUEUE dc9aff8 S1; scripts/dm_s7/nw3_patch.py) -------
def nw3_mask(path, thr=0.9, bits=3):
    """nw1 (nw_mask(path, 1)) + v3's (b) at its worst case: every UNSURE answer row's (draft cf < thr, or no draft; the
    engine's R57_U3_THR default 0.9) ranks 0-1 at `bits` from records3 on time (dm_t3; 3 = the MSB-3 record, as the
    engine's -2 sidecar slot; 4 = the control, rebuilt 4-bit through the same path), except a late rank (a cold base:
    4-bit, written one layer late as in nw1). v3's (a) changes nothing on the map: QOM's masks drop a cold optional use."""
    def setup():
        PERT.update(rmask=path, renorm=False, nw=1, nw3=True, gate=("t3u", thr), t3=bits, t3k=2)
    return setup
ABLATIONS["qom_0_live_ball_c208_nw3"] = nw3_mask(_NWD + "qom_0_live_ball_c208_nw1")   # NW v3 at 208 slots (twin: qom_0_live_ball_c208_nw1)
ABLATIONS["qom_0_live_ball_nw3"] = nw3_mask(_NWD + "qom_0_live_ball_nw1")             # NW v3 at the published pool (twin: qom_0_live_ball_nw1)
ABLATIONS["qom_0_live_ball_c208_nw3c"] = nw3_mask(_NWD + "qom_0_live_ball_c208_nw1", bits=4)   # the control: the same rows and path at 4 bits
'''

def patch(text, edits, name):
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            print(f"{name}: anchor found {n} times (need 1): {old[:90]!r}"); sys.exit(1)
        text = text.replace(old, new)
    return text

c = open(os.path.join(SRC, "dm_common.py")).read(); b = open(os.path.join(SRC, "dm_ablations.py")).read()
if MARK in c or "nw3_mask" in b:
    print("already in"); sys.exit(0)
assert "NEVER-WAIT" in c and "def nw_mask" in b and '_NWD = ' in b, "needs FAR's never-wait patch (install_nw.sh) first"
c2 = patch(c, EDITS_COMMON, "dm_common.py")
b2 = b + ("" if b.endswith("\n") else "\n") + ABL_ADD
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "dm_common.py"), "w").write(c2); open(os.path.join(OUT, "dm_ablations.py"), "w").write(b2)
print(f"written: {OUT}/dm_common.py (+{c2.count(chr(10)) - c.count(chr(10))} lines), {OUT}/dm_ablations.py (+{b2.count(chr(10)) - b.count(chr(10))} lines)")
