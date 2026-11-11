"""COLD-CUT keep masks (Breakthrough, Tue 2026-11-03 19:1x; Fable C's ~/FableGod/fablec/UNION_STACK.md section 5, the
orchestrator 18:5x; pre-registered in SCOREBOARD before the run). Unsure rows are cut by the expert's POPULARITY, not its rank:
  hyb_cold4_t3  sure rows (the realizable gate dm_gate: DFlash2 cf >= 0.9) keep ranks 0-1 as base, served at 3 bits by
                dm_t3 (ranks 0-1 off in the mask: the gate ("t3mask",), as hyb_0.9_u8_t3); unsure rows keep ranks 0-3 and
                every rank 4-9 whose expert is popular (share >= 0.05 in Fable C's pop_cap45.npy: the share of cap45's 176
                live checks whose exact layer-union holds the expert; six prompts, none of them a map window).
  cold6_all     the chat form: every row keeps ranks 0-5 + the popular ranks 6-9 (no drafter gate, no 3-bit rows).
  hyb_cold4_lowo_t3  (Tue 19:5x, SCOREBOARD 0a565a6: is the chat failure the prior's domain?) hyb_cold4_t3's rule with an
                IN-DOMAIN prior: per window, the share of the same domain's OTHER map windows' live checks whose exact
                layer-union holds the expert (leave-one-window-out on the routing captures; pop_cap45's definition).
Every realized row also keeps the experts already in its check's union (the union of the rows' base sets; the rows after the
first miss enter with the wrong-draft stand-in p + 97 and their drafts' cf, as dm_q1u_masks.py). Checks at LIVE lengths
(live_segments: the engine's running-cf cut, t 0.3). Prompt rows and rows outside the walk keep all 10.
Bytes check (Fable C: <= x0.55 of EXACT's union on code, or the mask is wrong): per live check and layer, the union's base
experts x 2,162,688 B (MSB-3) + the ranks 0-1 of its 4-bit rows x 770,048 B (LSB-1; 3-bit sure rows read none), against
EXACT (all ten ranks, ranks 0-1 LSB), priced as Fable C's replay_c4.py does, with no cache; u8_t3 (hyb_0.9_u8_t3's rule at
live lengths) is priced beside as the arm COLD-CUT would replace.
Self-check (selfcheck): this code's u8_t3 masks equal dm_q1u_masks.py's hyb_0.9_u8:t3 at live lengths bit for bit; the
latter stay as the CONTROL arm's masks (out/hyb_0.9_u8_live_t3: the arm COLD-CUT would replace, at the same live checks).
  hyb_cold4_pmt_t3 / hyb_cold4_pmtc_t3  (Tue 20:2x, SCOREBOARD 4236037) the table from the request's own prefill:
                popular iff its prompt rows routed to the expert at least once (pmt), or that OR cap45 >= 0.05 (pmtc).
  hyb_cold4_len_t3  (Tue 22:3x) the length-aware table: pmt's when the prompt has >= 128 rows, else pmtc's.
Usage (BigMoE venv): python dm_cold_masks.py [hyb_cold4_t3] [cold6_all] [hyb_cold4_lowo_t3] [hyb_cold4_pmt_t3] [hyb_cold4_pmtc_t3] | selfcheck
  -> out/<arm>/<window>.npz, out/COLD_bytes.json"""
import json, os, sys
import numpy as np
import dm_gate
from dm_q1u_masks import live_segments, OUT, base, G
POP = np.load(os.path.expanduser("~/FableGod/fablec/masks/pop_cap45.npy"))   # [48, 512] float32 (Fable C's, read-only)
MSB, LSB = 2162688, 770048
RULES = {"hyb_cold4_t3": (0.9, 4, 0.05, True, "cap45"), "cold6_all": (None, 6, 0.05, False, "cap45"),   # gate thr, r,
         "hyb_cold4_lowo_t3": (0.9, 4, 0.05, True, "lowo"),                                             # popularity p,
         "hyb_cold4_pmt_t3": (0.9, 4, 0.05, True, "prompt"), "hyb_cold4_pmtc_t3": (0.9, 4, 0.05, True, "prompt+cap45"),
         "hyb_cold4_len_t3": (0.9, 4, 0.05, True, "len"),                                               # pmt if >= 128 rows
         "u8_t3": (0.9, 8, None, True, None), "exact": (None, 10, None, False, None)}                  # t3, prior
WRITABLE = ("hyb_cold4_t3", "cold6_all", "hyb_cold4_lowo_t3", "hyb_cold4_pmt_t3", "hyb_cold4_pmtc_t3", "hyb_cold4_len_t3")
LEN_MIN = 128                                             # prompt rows: the length-aware table's switch (pre-registered)

def windows():
    """(dom, i, nm, Z, ids, g) for the measured windows, in walk order"""
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            yield dom, i, nm, Z, np.load(os.path.join(OUT, "routing", nm))["ids"], np.nan_to_num(G[nm], nan=-1.0)

def segments(Z, i, N, g):
    """per live check: routing rows (realized A .. fm, then the stand-ins p + 97), their cf, and nr = the realized count"""
    for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
        for (k, m, A, dc) in live_segments(k0_, m0_, A0_, dc0_):
            last, fm = min(A + m, N - 1), min(A + k, N - 1)
            if A > N - 1: continue                            # a live segment past the window's end
            rows, cf = [], []
            for p in range(A, last + 1):
                if p <= fm: rows.append(p); cf.append(-1.0 if (k == m and p == A + m) else g[p])   # a live cut's bonus row: unsure
                else: rows.append((p + 97) % N); cf.append(dc[p - A] if p - A < m else -1.0)
            yield np.array(rows), np.array(cf), fm - A + 1

_LOWO = {}
def prior(kind, dom, nm, ids=None):
    """cap45: Fable C's table; lowo: the share of the domain's other windows' live checks whose exact union holds the expert;
    prompt: 1 where the request's own prompt rows routed to the expert (any rank, at least once), else 0 (prefill, known
    before decoding); prompt+cap45: the larger of the two"""
    if kind == "cap45": return POP
    if kind in ("prompt", "prompt+cap45", "len"):
        if ids is None: ids = np.load(os.path.join(OUT, "routing", nm))["ids"]
        seen = np.zeros((48, 512), np.float32); a = base[nm]["a"]
        for l in range(48): seen[l, np.unique(ids[:a, l])] = 1.0
        if kind == "len": kind = "prompt" if a >= LEN_MIN else "prompt+cap45"   # Tue 22:3x: long prompts reveal enough
        return seen if kind == "prompt" else np.maximum(seen, POP)
    if not _LOWO:
        cnt = {}
        for d, i, w, Z, ids, g in windows():
            c = np.zeros((48, 512), np.float64); n = 0
            for rows, cf, nr in segments(Z, i, len(ids), g):
                n += 1
                for l in range(48): c[l, np.unique(ids[rows, l])] += 1
            cnt[w] = (d, c, n)
        for w, (d, c, n) in cnt.items():
            oc = sum(c2 for w2, (d2, c2, n2) in cnt.items() if d2 == d and w2 != w)
            on = sum(n2 for w2, (d2, c2, n2) in cnt.items() if d2 == d and w2 != w)
            _LOWO[w] = (oc / max(on, 1)).astype(np.float32)
    return _LOWO[nm]

def run(write, dirs=None):
    dirs = dirs or {a: os.path.join(OUT, a) for a in write}
    for a in write: os.makedirs(dirs[a], exist_ok=True)
    st = {}                                                   # (dom, arm) -> union B, MSB groups, kept sure, sure rows, kept unsure, unsure rows
    for dom, i, nm, Z, ids, g in windows():
        N = len(ids); keep = {a: np.ones((N, 48, 10), bool) for a in write}
        P = {a: prior(R[4], dom, nm, ids) for a, R in RULES.items() if R[4]}
        for rows, cf, nr in segments(Z, i, N, g):
            for arm, (thr, r, pth, t3, pk) in RULES.items():
                sure = ~(cf < thr) if thr is not None else np.zeros(len(rows), bool)   # dm_q1u_masks' test (a NaN cf is sure)
                s = st.setdefault((dom, arm), np.zeros(6))
                for l in range(48):
                    E = ids[rows, l]                          # [rows, 10] rank-sorted
                    b = np.zeros(E.shape, bool); b[sure, :2] = True; b[~sure, :r] = True
                    if pth is not None: b[~sure] |= P[arm][l][E[~sure]] >= pth
                    u = np.unique(E[b]); l4 = np.unique(E[~sure if t3 else slice(None), :2])
                    s[0] += len(u) * MSB + len(l4) * LSB; s[1] += len(u)
                    kp = b[:nr] | np.isin(E[:nr], u)
                    s[2] += kp[sure[:nr]].sum(); s[3] += sure[:nr].sum(); s[4] += kp[~sure[:nr]].sum(); s[5] += (~sure[:nr]).sum()
                    if arm in keep:
                        kq = kp.copy()
                        if t3: kq[sure[:nr], :2] = False      # this check's sure rows: ranks 0-1 from records3 (dm_t3)
                        keep[arm][rows[:nr], l] = kq
        for a in write: np.savez(os.path.join(dirs[a], nm), keep=keep[a])
    res = {}
    for dom in ("code", "ifeval", "multi"):
        ex = st[(dom, "exact")]
        for arm in RULES:
            s = st[(dom, arm)]
            res.setdefault(arm, {})[dom] = dict(bytes=s[0] / ex[0], groups=s[1] / ex[1], kept_sure=s[2] / max(s[3], 1),
                                                kept_unsure=s[4] / max(s[5], 1), sure_share=s[3] / max(s[3] + s[5], 1))
            print(f"{arm:17s} {dom:6s}: union bytes x{s[0] / ex[0]:.3f} of EXACT's (groups x{s[1] / ex[1]:.3f}); realized rows keep "
                  f"{s[2] / max(s[3], 1):.2f} (sure, {100 * s[3] / max(s[3] + s[5], 1):.0f} % of row-layers) / {s[4] / max(s[5], 1):.2f} (unsure) of 10", flush=True)
    if "u8_t3" not in write: json.dump(res, open(os.path.join(OUT, "COLD_bytes.json"), "w"), indent=1)
    return res

if __name__ == "__main__":
    if sys.argv[1:] == ["selfcheck"]:
        RULES = {k: v for k, v in RULES.items() if k in ("u8_t3", "exact")}
        mine = os.path.join(OUT, "_selfcheck_cold_u8_t3"); run(["u8_t3"], {"u8_t3": mine})
        import dm_q1u_masks; os.environ["MASK_TAG"] = "hyb_0.9_u8_live_t3"; dm_q1u_masks.masks(None, "hyb_0.9_u8", t3=True, live=True)
        ref = os.path.join(OUT, "hyb_0.9_u8_live_t3"); fs = sorted(os.listdir(ref))
        bad = [f for f in fs if not np.array_equal(np.load(os.path.join(mine, f))["keep"], np.load(os.path.join(ref, f))["keep"])]
        print(f"self-check u8_t3 vs dm_q1u_masks hyb_0.9_u8:t3 (live): {f'ALL EQUAL ({len(fs)})' if not bad and len(fs) == 36 else f'{len(bad)} of {len(fs)} DIFFER: {bad[:3]}'}")
        sys.exit(0)
    arms = sys.argv[1:] or ["hyb_cold4_t3", "cold6_all"]
    assert all(a in WRITABLE for a in arms), arms
    r = run(arms)
    for a in arms:
        if RULES[a][3]:
            c = r[a]["code"]["bytes"]
            print(f"bytes check (Fable C: <= x0.55 on code): {a} x{c:.3f} -> {'OK' if c <= 0.55 else 'MASK WRONG: do not run the arm'}")
