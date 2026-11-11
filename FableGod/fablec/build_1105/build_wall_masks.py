"""build_wall_masks.py (Fable C, Thu 5 Nov 2026): a COPY of NewIdeas' ~/NewIdeas/scripts/dm_oea_masks.py (md5 c6b88322a4b571c222a062a3519f68ac)
with the WALL 1105 extensions for the qom_<thr>_split<T> arms. Every changed or added line carries "WALL". With no extension
token the code path is the original's, and check_build.py verifies the rebuilt qom_0_split0.99_live_ball, qom_0_split0.9_live_ball
and qom_0_live_ball equal the installed masks on 36 / 36 windows.
Extension tokens, appended to qom_<thr>_split<T> before _live (any subset, in this order):
  _k<cap>   a NON-strict gated row keeps ranks 0-1, the block's union members, and its resident experts at ranks < cap only
            (resident ranks >= cap are dropped and not touched). K8 = _k8, K6 = _k6, K4 = _k4.
  _r<tau>   REACH-CUT in the split form: a drafted row with cf < T whose reach (the product of the cf of the drafts before
            it in its live segment; 1 for the anchor row; dm_oea_masks' own REACH-CUT definition) is < tau is strict too.
  _m<thr>   the middle-layer split: a row with cf >= thr (and < T) is strict in layers 12-35 only.
  _z<thr>   the middle-layer skip: a row whose REAL draft cf is >= thr keeps NO routed expert in layers 8-35 (nothing loaded
            for it there). A last row gated by _ball's synthesized cf 1.0 (no next draft) is not skipped.
Arms of WALL_1105.md section 4: K8 qom_0_split0.99_k8_live_ball, K6 ..._k6_..., K4 ..._k4_..., A qom_0_split0.99_r0.7_live_ball,
A' qom_0_split0.99_m0.9_live_ball, B qom_0_split0.99_r0.7_z0.999_live_ball.

---- the original docstring follows ----
OEA / hybrid keep masks for Breakthrough's decision map (NewIdeas, Tue 3 Nov 2026 06:4x).

OEA = Opportunistic Expert Activation (arXiv 2511.02237, ICML 2026; Lynx arXiv 2411.08982 is related): every row keeps its
top-k0 experts, then adds any lower-ranked expert that is already in the batch's union. Here the batch is one 16-row
verify check, as in Q1-UNION (dm_q1u_masks.py), whose base k0 is set per row by the drafter's certainty (2 on sure
rows, 10 on the rest). Arms:
  q1u_<thr>          Q1-UNION itself (self-check: must reproduce out/q1u_cf0<..>/ bit for bit)
  oea_k<k>           OEA, the same k0 on every row (published method)
  hyb_<thr>_u<k>     the stack: sure rows (cf >= thr) k0 2, unsure rows k0 k, every row adds union members
  top_k<k>           control: plain top-k on the realized rows (no union members), prompt rows exact as in every arm
  <arm>_live         any arm on LIVE checks: the default's running-confidence cut (t 0.3) shortens each walk check as
                     the engine would (the plain arms use all 16 rows, so their unions are larger than live ones)
  reach_<thr>_<tau>  Q1-UNION <thr> plus REACH-CUT (Tue 10:3x): an unsure row whose chance of being reached, the product
                     of the walk's draft cf before it (prod_{i<j} dcf_i; 1 for the anchor row), is < tau also gets k0 2
                     here (the map's own topk<k> ablation also cuts prompt rows)
  qom_<thr>          Q1-ON-MISS (Fable C's C1-2; pre-registered SCOREBOARD 00d485c, then superseded by Q1-UNION; Tue 17:0x
                     revived for chat after the live-cut result): sure rows keep ranks 0-1, every rank 2-9 expert in the
                     check's union (= Q1-UNION) AND every rank 2-9 expert resident in the base pool after the check's
                     loads. Q1's SSD reads by construction. CPU LRU replay per layer, carried across a domain's windows
                     in walk order, the prompt rows' top-10 warm it (PF_WARM), each check loads its union (all ranks of
                     unsure rows, ranks 0-1 of sure rows, stand-ins by their draft cf), then the kept resident experts
                     are touched. Pool per layer: QOM_C = "m" (default: laysz45_256.json's C_m, the 45 GB default's base
                     pool, mean 295 of 512), an integer (uniform), or 0 (self-check: must equal q1u_<thr> bit for bit).
                     Plain LRU (the engine adds 30 x a popularity prior), so slightly pessimistic on the kept share.
Blocks, stand-ins and the masked rows follow dm_q1u_masks.py exactly: the HPD-0 walks' checks (rows A .. A + m); rows
after a check's first miss enter the union with the wrong-draft stand-in (row p + 97) and their own drafts' cf; only a
check's realized rows (A .. first miss) are masked; prompt rows and rows outside the walk keep all 10.
Out: <out_dir>/<arm>/<window>.npz, keep [rows, 48, 10] bool, rank-sorted as DAY9's routing capture (the harness's
rmask: PERT.update(rmask=<absolute dir>, renorm=False); os.path.join(OUT, abs, nm) resolves to the absolute dir).
Reads ~/FableGod/decision_map (read-only). Usage (BigMoE venv): python dm_oea_masks.py <out_dir> <arm> [<arm> ...]
"""
import json, os, re, sys
from collections import OrderedDict
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map")
sys.path.insert(0, DM)
import dm_gate  # noqa: E402  (load() only reads out/gate_dfl2.json)
OUT = os.path.join(DM, "out")


LIVE_T = 0.3


def live_segments(k, m, A, dc):
    """<arm>_live: the live engine's running-confidence cut (t 0.3) applied to one walk check. Drafts are kept while the
    product of their cf is >= 0.3. A cut before the first miss ends the check at the kept drafts' bonus row, and the next
    check starts at the next row with the walk's remaining drafts (their cf as the walk recorded them: an approximation
    for the 4-8 % of checks the cut ends early). Returns [(k, m, A, dc)] in the walk's own form."""
    segs, a0, kk, dd = [], A, k, [float(x) for x in dc[:m]]
    while True:
        L = int((np.cumprod(dd) >= LIVE_T).sum()) if dd else 0
        if L >= kk or L == len(dd):
            segs.append((kk, L, a0, dd)); return segs
        segs.append((L, L, a0, dd[:L]))                 # the L kept drafts are all accepted; bonus row a0 + L
        a0, kk, dd = a0 + L + 1, kk - L - 1, dd[L + 1:]


def parse(arm):
    m = re.fullmatch(r"q1u_([0-9.]+)", arm)
    if m: return float(m.group(1)), 2, 10
    m = re.fullmatch(r"oea_k(\d+)", arm)
    if m: return None, int(m.group(1)), int(m.group(1))
    m = re.fullmatch(r"hyb_([0-9.]+)_u(\d+)", arm)
    if m: return float(m.group(1)), 2, int(m.group(2))
    m = re.fullmatch(r"hyb_([0-9.]+)_k(\d+)_u(\d+)", arm)    # Wed 04:0x: sure rows keep top-k (k 1 = BYTES10 screen 2), unsure top-u
    if m: return float(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r"reach_([0-9.]+)_([0-9.]+)", arm)        # Q1-UNION <thr> + REACH-CUT below tau (k0 2 on those rows)
    if m: return float(m.group(1)), 2, 10
    m = re.fullmatch(r"top_k(\d+)", arm)                     # control: plain top-k on realized rows, no union members
    if m: return None, int(m.group(1)), int(m.group(1))
    m = re.fullmatch(r"qom_([0-9.]+)", arm)                  # Q1-ON-MISS: Q1-UNION + experts resident in the base pool
    if m: return float(m.group(1)), 2, 10
    m = re.fullmatch(r"qom_([0-9.]+)_k(\d+)", arm)          # Wed 04:0x: QOM with sure rows at top-k (k 1: + resident ranks 1-9)
    if m: return float(m.group(1)), int(m.group(2)), 10
    m = re.fullmatch(r"qom_([0-9.]+)_u(\d+)(?:_reach[0-9.]+)?", arm)   # Thu 07:0x (Cloud3 C3-R15 / R15b): QOM with unsure rows
    if m: return float(m.group(1)), 2, int(m.group(2))      # at top-u + union members + resident; _reach<tau>: REACH-CUT too
    m = re.fullmatch(r"qom_([0-9.]+)_split([0-9.]+)", arm)  # Thu 07:2x (Fable C audit F4 / test 1): QOM, but drafted rows with
    if m: return float(m.group(1)), 2, 10                   # cf >= the split keep ranks 0-1 + members only (no resident keep)
    raise SystemExit(f"unknown arm {arm}")


def live_segments_nxt(k, m, A, dc):
    """live_segments plus, per segment, the cf of the draft its last row verifies (-1.0: none left). The drafter drafts
    the whole block in one pass and the live cut drops only inputs, so for a cut segment that draft already exists."""
    segs, a0, kk, dd = [], A, k, [float(x) for x in dc[:m]]
    while True:
        L = int((np.cumprod(dd) >= LIVE_T).sum()) if dd else 0
        if L >= kk or L == len(dd):
            segs.append((kk, L, a0, dd, dd[L] if L < len(dd) else -1.0)); return segs
        segs.append((L, L, a0, dd[:L], dd[L]))
        a0, kk, dd = a0 + L + 1, kk - L - 1, dd[L + 1:]


def pool_sizes():
    q = os.environ.get("QOM_C", "m")
    if q == "m":
        return json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))["C_m"]
    return [int(q)] * 48


def touch(c, e, C):
    if e in c: c.move_to_end(e)
    else:
        c[e] = 1
        if len(c) > C: c.popitem(last=False)


def masks(out_dir, arm):
    t3 = arm.endswith("_t3"); arm_ = arm[:-3] if t3 else arm   # <arm>_t3 (Tue 23:0x): sure rows' ranks 0-1 off in the mask, so
    # the harness's ('t3mask',) gate serves them from records3 at 3 bits (Breakthrough's hyb_t3 convention); pools unchanged
    mc = re.search(r"_c(\d+)$", arm_)                      # <arm>_c<N> (Tue 23:2x): qom's base pool = N slots per layer
    cpool = int(mc.group(1)) if mc else None                  # (e.g. _c164 = the uniform C 160 line), not laysz45_256's C_m
    if mc: arm_ = arm_[:mc.start()]
    ball = arm_.endswith("_ball"); arm_ = arm_[:-5] + "_bq" if ball else arm_   # <arm>_live_ball (Wed 13:5x, DRAFT17): like
    # _bq, but a last row with no next draft is gated too (a drafted 17th position; at cf 0 any drafted row is sure)
    bq = arm_.endswith("_bq"); arm_ = arm_[:-3] if bq else arm_   # <arm>_live_bq (Wed 04:4x, BYTES10 screen 1 in QOM's form): each
    # live segment's last row (the bonus row, or the discarded last row after a miss) is gated by the draft it verifies,
    # like a drafted row (QOM: ranks 0-1 + union members + resident ranks 2-9); exact when no draft is left
    live = arm_.endswith("_live"); arm0 = arm_[:-5] if live else arm_
    if bq and not live: raise SystemExit("_bq needs a _live arm")
    wx = re.fullmatch(r"(qom_[0-9.]+_split[0-9.]+)((?:_(?:k\d+|r[0-9.]+|m[0-9.]+|z[0-9.]+))*)", arm0)   # WALL: the extension tokens
    wtok = dict((t[0], t[1:]) for t in wx.group(2).split("_") if t) if wx else {}                           # WALL
    if wx: arm0 = wx.group(1)                                                                               # WALL
    cap = int(wtok["k"]) if "k" in wtok else 10                                                             # WALL: resident ranks < cap
    wr = float(wtok["r"]) if "r" in wtok else None                                                          # WALL: reach below this is strict
    mthr = float(wtok["m"]) if "m" in wtok else None                                                        # WALL: strict in layers 12-35 from this cf
    zthr = float(wtok["z"]) if "z" in wtok else None                                                        # WALL: no routed expert in layers 8-35 from this cf
    thr, ks, ku = parse(arm0); plain = arm0.startswith("top_k")
    tau = float(arm0.split("_")[2]) if arm0.startswith("reach_") else None
    mr = re.search(r"^qom_[0-9.]+_u\d+_reach([0-9.]+)$", arm0)   # Thu 07:0x: QOM's REACH-CUT (rows keep residents, the qom path)
    if mr: tau = float(mr.group(1))
    ms_ = re.search(r"^qom_[0-9.]+_split([0-9.]+)$", arm0)   # Thu 07:2x: the strict band (loads unchanged; strict rows touch
    scf = float(ms_.group(1)) if ms_ else None               # only what they keep, as audit_split_replay.py)
    qom = arm0.startswith("qom_"); CL = ([cpool] * 48 if cpool else pool_sizes()) if qom else None
    base = json.load(open(os.path.join(OUT, "baseline.json")))
    G = dm_gate.load()
    od = os.path.join(out_dir, arm); os.makedirs(od, exist_ok=True)
    st, qs, bqn, tls = {}, {}, {}, {}
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        cache = [OrderedDict() for _ in range(48)]               # qom: the base pool per layer, carried across windows
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids)
            g = np.nan_to_num(G[nm], nan=-1.0)
            keep = np.ones((N, 48, 10), bool)
            d = st.setdefault(dom, [0, 0, 0, 0])               # union, exact union, kept row-layer experts, row-layers
            q = qs.setdefault(dom, [0, 0, 0, 0, 0])            # qom: sure-row rank 2-9 pairs, kept as members, kept as
                                                               # resident, loads, load misses
            if qom:
                for p in range(base[nm]["a"]):                 # prefill warms the pool (PF_WARM)
                    for l in range(48):
                        for e in ids[p, l].tolist(): touch(cache[l], e, CL[l])
            for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
              segs_ = live_segments_nxt(k0_, m0_, A0_, dc0_) if live else [(k0_, m0_, A0_, dc0_, -1.0)]
              if live: assert [x[:4] for x in segs_] == live_segments(k0_, m0_, A0_, dc0_)
              for (k, m, A, dc, nxt) in segs_:
                    last, fm = min(A + m, N - 1), min(A + k, N - 1)
                    rows, cfs = [], []; syn = []                 # WALL: syn = cf synthesized by _ball
                    for p in range(A, last + 1):
                        if p <= fm: rows.append(p); c_ = -1.0 if live and k == m and p == A + m else g[p]  # live
                                                     # cut: the kept drafts' bonus row verifies no draft (exact)
                        else: rows.append((p + 97) % N); c_ = dc[p - A] if p - A < m else -1.0
                        if bq and p - A == m and (nxt >= 0 or ball):   # _bq: the last row verifies the next draft (engine);
                            c_ = nxt if nxt >= 0 else 1.0                # _ball: a drafted 17th position when none is left
                            bqn[dom] = bqn.get(dom, 0) + int(p <= fm)
                        cfs.append(c_); syn.append(bool(bq and p - A == m and nxt < 0 and ball))   # WALL (syn only)
                    if not rows: continue                        # a live segment past the window's end
                    cfs = np.array(cfs)
                    k0 = np.full(len(rows), ku) if thr is None else np.where(cfs >= thr, ks, ku)
                    if tau is not None:                          # P(reach row A + j) = prod of the drafts' cf before it
                        dca = np.array(dc, float); reach = np.r_[1.0, np.cumprod(dca)][:len(rows)]
                        k0 = np.where((k0 == ku) & (reach < tau), ks, k0)
                    wreach = np.r_[1.0, np.cumprod(np.array(dc, float))][:len(rows)] if wr is not None else None   # WALL
                    zrow = [zthr is not None and cfs[j] >= zthr and not syn[j] for j in range(len(rows))]             # WALL
                    def strict_(j, l):                                                                               # WALL: the strict test
                        if scf is not None and cfs[j] >= scf: return True                                           # (the original rule)
                        if wr is not None and cfs[j] >= 0 and wreach[j] < wr: return True
                        return mthr is not None and cfs[j] >= mthr and 12 <= l <= 35
                    for l in range(48):
                        zl = [zrow[j] and 8 <= l <= 35 for j in range(len(rows))]                                    # WALL
                        ul = [ids[r, l, :k0[j]] for j, r in enumerate(rows) if not zl[j]]                            # WALL: skipped rows load nothing
                        U = np.unique(np.concatenate(ul)) if ul else np.zeros(0, ids.dtype)                          # WALL (original: every row)
                        d[0] += len(U); d[1] += len(np.unique(ids[rows, l, :]))
                        if qom:                                  # the check's loads, then what is resident after them
                            c = cache[l]
                            for e in U.tolist():
                                q[3] += 1; q[4] += e not in c; touch(c, e, CL[l])
                            res = np.fromiter(c.keys(), np.int64, len(c))
                        for j, p in enumerate(range(A, fm + 1)):
                            if zl[j]:                                                                                # WALL: the middle-layer skip
                                keep[p, l, :] = False; continue
                            if k0[j] < 10:
                                keep[p, l, k0[j]:] = False if plain else np.isin(ids[p, l, k0[j]:], U)
                                if qom:
                                    mem = keep[p, l, k0[j]:].copy()
                                    if not strict_(j, l):                                                            # WALL (original: scf is None or cfs[j] < scf)
                                        keep[p, l, k0[j]:] |= np.isin(ids[p, l, k0[j]:], res) & (np.arange(k0[j], 10) < cap)   # WALL: & the cap
                                    q[0] += 10 - k0[j]; q[1] += int(mem.sum()); q[2] += int((keep[p, l, k0[j]:] & ~mem).sum())
                                if t3 and thr is not None and k0[j] == ks:   # a sure row of this check (never the
                                    keep[p, l, :ks] = False                 # live bonus row: its cf is -1); its top-ks
                                    # ranks go to the t3 gate (harness t3k = ks; ks 2 = every arm before Wed 04:0x)
                        if qom:                                  # tiles: distinct kept experts of the block, every row
                            Us, rs = set(U.tolist()), set(res.tolist()); kset = set()   # (stand-ins too), as the audit
                            for j, r in enumerate(rows):
                                if zl[j]: continue                                                                   # WALL
                                rr = ids[r, l].tolist(); kk = int(k0[j]); kset.update(rr[:kk])
                                sj = strict_(j, l)                                                                   # WALL (original: scf is not None and cfs[j] >= scf)
                                kset.update(e for q_, e in enumerate(rr[kk:], start=kk) if e in Us or (not sj and q_ < cap and e in rs))   # WALL: the cap
                            tls.setdefault(dom, [0, 0])[0] += len(kset)
                        if qom:                                  # kept resident experts are used: touch them (all sure rows)
                            for j, r in enumerate(rows):
                                if zl[j]: continue                                                                   # WALL
                                if k0[j] < 10:
                                    sj = strict_(j, l)                         # a strict row touches only its members (WALL: strict_)
                                    for q_, e in enumerate(ids[r, l, k0[j]:].tolist(), start=int(k0[j])):          # WALL: rank index for the cap
                                        if e in c and (e in Us or (not sj and q_ < cap)): c.move_to_end(e)          # WALL (original: not sj or e in Us)
                        d[2] += int(keep[A:fm + 1, l].sum()); d[3] += fm + 1 - A
                    if qom: tls.setdefault(dom, [0, 0])[1] += min(k, fm - A) + 1   # kept tokens of this segment
            np.savez_compressed(os.path.join(od, nm), keep=keep)
    for dom, (u, ux, kp, rl) in st.items():
        print(f"{arm} {dom}: union x{u / ux:.3f} per check, experts kept per realized row-layer {kp / max(rl, 1):.2f} of 10",
              flush=True)
        if bq: print(f"  {arm} {dom}: realized bonus rows gated by their next draft: {bqn.get(dom, 0)}", flush=True)
        if qom and dom in tls: print(f"  {arm} {dom}: tiles per kept token {tls[dom][0] / max(tls[dom][1], 1):.1f}", flush=True)
        if qom:
            n, mem, res, ld, ms = qs[dom]
            print(f"  qom {dom} (pool {cpool or os.environ.get('QOM_C', 'm')}): sure-row rank 2-9 pairs {n}: kept as union members "
                  f"{100 * mem / max(n, 1):.1f} %, + resident {100 * res / max(n, 1):.1f} % = {100 * (mem + res) / max(n, 1):.1f} %; "
                  f"load miss rate {100 * ms / max(ld, 1):.1f} % of {ld} loads", flush=True)
    if arm.startswith("q1u_") and not live:                 # self-check against Breakthrough's own masks
        ref = os.path.join(OUT, "q1u_cf0" + str(thr)[2:])
        bad = sum(not np.array_equal(np.load(os.path.join(od, f))["keep"], np.load(os.path.join(ref, f))["keep"])
                  for f in sorted(os.listdir(od)) if os.path.exists(os.path.join(ref, f)))
        print(f"self-check vs {ref}: {'ALL EQUAL' if bad == 0 else f'{bad} windows DIFFER'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3: raise SystemExit(__doc__)
    for a in sys.argv[2:]: masks(os.path.abspath(sys.argv[1]), a)
