"""NEVER-WAIT mask builder, generated from NewIdeas' dm_oea_masks.py by anchored edits (Fable FAR, Thu 5 Nov 2026).
NEVERWAIT_TEST.md 2.1-2.3 and 8.3: the builder changes are the `_nw*` suffix strip before the `_t3` strip and the
`_c<N>` match, the late set M_l in the load loop, the `late` line after the keep rule, the tail rule, and the saved
`late` array. Every edit is an anchor that must occur exactly once in the source, so the generated builder is NewIdeas'
loop line for line plus these lines: an arm without `_nw*` runs the original code path (its keep is checked bit for bit
against NewIdeas' own builder by check_masks.py).
Usage: python -I mk_builder.py <dm_oea_masks.py> <out nw_masks.py>   (CPU, instant; writes only <out>)"""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

def rep(old, new, where):
    global s
    assert s.count(old) == 1, (where, s.count(old))
    s = s.replace(old, new)

# 0. docstring: what the generated file is
rep('''Reads ~/FableGod/decision_map (read-only). Usage (BigMoE venv): python dm_oea_masks.py <out_dir> <arm> [<arm> ...]
"""''',
    '''Reads ~/FableGod/decision_map (read-only). Usage (BigMoE venv): python dm_oea_masks.py <out_dir> <arm> [<arm> ...]

NEVER-WAIT (Fable FAR + ZERO, ~/FableGod/far/NEVERWAIT_TEST.md 2.1-2.3; this file = dm_oea_masks.py + mk_builder.py's
anchored edits, Thu 5 Nov 2026). <qom arm>[_c<N>]_nw<d>[r1|s], d in 0 / 1 / 2:
  late[p, l, r] = keep[p, l, r] AND ids[p, l, r] in M_l on the check's realized rows, every rank, where M_l = the
  experts of the check's union U_l that are NOT in the pool at the moment of their touch (the events the load-miss
  count already counts). The harness writes a late term d layers late through layer l's own inject gates.
  r1: an expert some row of the check (realized or stand-in) has at rank 0 is waited for (never late).
  s:  late only on realized rows whose draft cf >= 0.9.
  Tail (d >= 1, layers >= 48 - d, nothing later to write after): a late rank-0 term stays in place (late False: the
  engine waits there), any other late term is dropped (keep False). d = 0 has no tail (the lateness-0 self-check).
  Never with _t3 (a _t3 mask sets the sure rows' keep[:, :, :2] False, so keep & late would lose every late top-2).
  Out: keep as the twin's (but for the tail's drops) plus late [rows, 48, 10] bool, rank-sorted like keep; the pool
  evolves exactly as the twin's. Stats per domain: <out_dir>/<arm>.stats.json and the printed "nw" line.
"""
import sys as _sys
_sys.dont_write_bytecode = True                              # nothing is written beside the map's or NewIdeas' sources''', "docstring")

rep('''LIVE_T = 0.3
''', '''LIVE_T = 0.3
RANK_W = np.array([0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010])   # nw stats: AUDIT_1105 7.2's mean
# renormalised gate weight by rank (as ~/FableGod/zero/late_terms.py), for "late MoE mass per realized row-layer"
''', "RANK_W")

# 1. the suffix strip, before _t3 and _c<N>
rep('''    t3 = arm.endswith("_t3"); arm_ = arm[:-3] if t3 else arm   # <arm>_t3 (Tue 23:0x): sure rows' ranks 0-1 off in the mask, so
''', '''    mn = re.search(r"_nw([012])(r1|s)?$", arm)                # <arm>_nw<d>[r1|s] (Thu 11-05): stripped BEFORE _t3 and _c<N>
    nwd, nwv = (int(mn.group(1)), mn.group(2) or "") if mn else (None, "")
    armt = arm[:mn.start()] if mn else arm                    # the twin's name; `arm` stays the output folder's name
    if mn and armt.endswith("_t3"): raise SystemExit(f"{arm}: never-wait arms are built without _t3")
    t3 = armt.endswith("_t3"); arm_ = armt[:-3] if t3 else armt   # <arm>_t3 (Tue 23:0x): sure rows' ranks 0-1 off in the mask, so
''', "suffix strip")

rep('''    qom = arm0.startswith("qom_"); CL = ([cpool] * 48 if cpool else pool_sizes()) if qom else None
''', '''    qom = arm0.startswith("qom_"); CL = ([cpool] * 48 if cpool else pool_sizes()) if qom else None
    if mn and not qom: raise SystemExit(f"{arm}: never-wait needs a qom arm (the cold set is the pool replay's)")
''', "qom only")

rep('''    st, qs, bqn, tls = {}, {}, {}, {}
''', '''    st, qs, bqn, tls = {}, {}, {}, {}
    nws, nwr = {}, {}                                         # nw stats per domain: [top-2, members, undrafted tails, mass, tail
                                                              # rank 0 in place, tail dropped]; realized rows with a late term
''', "stats init")

rep('''            keep = np.ones((N, 48, 10), bool)
''', '''            keep = np.ones((N, 48, 10), bool)
            late = np.zeros((N, 48, 10), bool); hit = np.zeros(N, bool)   # nw: the late flags; rows with >= 1 late term
''', "late init")

# 2. M_l: the loads not in the pool at the moment of their touch (the same event q[4] counts)
rep('''                            for e in U.tolist():
                                q[3] += 1; q[4] += e not in c; touch(c, e, CL[l])
''', '''                            Ml = []                              # nw: M_l, in touch order
                            for e in U.tolist():
                                q[3] += 1; q[4] += e not in c
                                if mn and e not in c: Ml.append(e)
                                touch(c, e, CL[l])
''', "M_l")

# 3. the late line (after the keep rule of the realized rows), with the tail rule
rep('''                        if qom:                                  # tiles: distinct kept experts of the block, every row
''', '''                        if mn and Ml:                            # NEVER-WAIT 2.3: late = keep AND expert in M_l, realized rows, every rank
                            Mx = set(Ml) - set(ids[rows, l, 0].tolist()) if nwv == "r1" else set(Ml)   # r1: rank-0 experts wait
                            Ma = np.fromiter(Mx, np.int64, len(Mx)); w = nws.setdefault(dom, [0, 0, 0, 0.0, 0, 0])
                            for j, p in enumerate(range(A, fm + 1)):
                                if not len(Ma) or (nwv == "s" and cfs[j] < 0.9): continue   # s: rows whose draft cf >= 0.9 only
                                lt = keep[p, l] & np.isin(ids[p, l], Ma)
                                if not lt.any(): continue
                                w[0] += int(lt[:2].sum()); w[1 if k0[j] < 10 else 2] += int(lt[2:].sum())
                                w[3] += float(RANK_W[lt].sum()); hit[p] = True
                                if nwd and l >= 48 - nwd:        # the tail: a late rank 0 stays in place, the rest is dropped
                                    w[4] += int(lt[0]); w[5] += int(lt[1:].sum()); keep[p, l, 1:] &= ~lt[1:]
                                else: late[p, l] = lt
                        if qom:                                  # tiles: distinct kept experts of the block, every row
''', "late line")

# 4. save and print
rep('''            np.savez_compressed(os.path.join(od, nm), keep=keep)
''', '''            if mn:
                nwr[dom] = nwr.get(dom, 0) + int(hit.sum())
                assert not (late & ~keep).any() and not late[:base[nm]["a"]].any() and (nwd == 0 or not late[:, 48 - nwd:].any()), nm
                np.savez_compressed(os.path.join(od, nm), keep=keep, late=late)
            else: np.savez_compressed(os.path.join(od, nm), keep=keep)
''', "save")

rep('''                  f"load miss rate {100 * ms / max(ld, 1):.1f} % of {ld} loads", flush=True)
''', '''                  f"load miss rate {100 * ms / max(ld, 1):.1f} % of {ld} loads", flush=True)
        if mn:
            w = nws.get(dom, [0, 0, 0, 0.0, 0, 0]); rr = rl / 48; kt = tls[dom][1]
            S = dict(arm=arm, dom=dom, pool=cpool or os.environ.get("QOM_C", "m"), lateness=nwd, variant=nwv, realized_rows=int(rr), kept_tokens=int(kt),
                     misses_per_kept=qs[dom][4] / max(kt, 1), late_per_row=(w[0] + w[1] + w[2]) / max(rr, 1), late_top2_per_row=w[0] / max(rr, 1),
                     late_members_per_row=w[1] / max(rr, 1), late_undrafted_tails_per_row=w[2] / max(rr, 1), late_mass_pct=100 * w[3] / max(rl, 1),
                     rows_with_late_pct=100 * nwr.get(dom, 0) / max(rr, 1), tail_rank0_in_place=w[4], tail_dropped=w[5])
            json.dump(S, open(os.path.join(out_dir, f"{arm}.{dom}.stats.json"), "w"), indent=1)
            print(f"  nw {arm} {dom}: demand misses per kept token {S['misses_per_kept']:.2f}; late terms per realized row {S['late_per_row']:.2f} "
                  f"(top-2 {S['late_top2_per_row']:.2f} / members {S['late_members_per_row']:.2f} / undrafted tails {S['late_undrafted_tails_per_row']:.2f}); "
                  f"late MoE mass per realized row-layer {S['late_mass_pct']:.2f} %; realized rows with a late term {S['rows_with_late_pct']:.0f} % of "
                  f"{S['realized_rows']}; tail (layers >= {48 - nwd}): {w[4]} late rank-0 terms in place, {w[5]} late terms dropped "
                  f"(all counted before the tail rule)", flush=True)
''', "print")

open(dst, "w").write(s)
print(f"wrote {dst}: {len(s.splitlines())} lines from {src}")
