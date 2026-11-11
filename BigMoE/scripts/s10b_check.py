"""S10b trace check (b31): are the files consistent, and how far does the plain forecaster get? CPU only (numpy).
- every verify pass (a seq with all trunk layers 0..NL-1) has the same rows at every layer, taking each layer's FIRST
  call in the pass: after a partly accepted verify, MTPLX's commit re-runs layer 1 over the kept rows under the same seq
  (b31: 567 of 4405 passes, always L1, fewer rows, all experts resident so the ext loader never sees it): skipped;
- the ext's miss record of each (pass, base layer) holds exactly the union of the Python trace's ids there;
- misses: per verify row and trunk layer, the row's routed experts that were not resident (miss.bin flags);
- baseline (what TrainLab's predictor must beat): router_T applied to layer L's MoE input as is (qwen4_exp: the engine's
  look-ahead), top-16 per row, recall of the row's misses at T = L+1 / L+2 / L+3, per domain (the S10b kill line is
  lead-2 recall@16 of the misses on held-out prompts < 55 %; b30's FC1 log measured the engine's own lists at lead 2);
- b32 (res.bin present): the DEPLOYABLE score, the same recall with the top-16 taken among the experts NOT resident at
  the engine's prefetch decision for (seq, L -> T) (it skips resident experts); NewIdeas 09:08: held-out lead 2 >= 65 %
  -> engine A/B, 55-65 % -> NewIdeas, < 55 % -> stop S10b. The pools hold ~190 of 512 experts per layer.
usage: s10b_check.py TRACE_DIR   -> prints a summary, writes TRACE_DIR/check.json"""
import glob, json, os, sys
import numpy as np

D = sys.argv[1]
LD = os.environ.get("S10B_LABEL") == "ld"                    # score late + demand misses only (b33 kinds; -> check_ld.json)
meta = json.load(open(os.path.join(D, "meta.json")))
NL, LP = meta["n_layers"], meta["layers_pred"]
raw = np.fromfile(os.path.join(D, "miss.bin"), np.int32)
miss, dup, i = {}, 0, 0
while i < len(raw):
    seq, L, n = int(raw[i]), int(raw[i + 1]), int(raw[i + 2])
    v = raw[i + 3:i + 3 + n]
    if (seq, L) in miss:
        dup += 1
    else:                                                    # S10B_LABEL=ld (b33+): drop on-time prefetches (kind 1)
        miss[(seq, L)] = (v & 0xFFFF, ((v >> 16) & 1) * (((v >> 17) & 3) != 1 if LD else 1))
    i += 3 + n
W = ({k: v.astype(np.float32).T.copy() for k, v in np.load(os.path.join(D, "routers.npz")).items()}   # [D, E]
     if os.path.exists(os.path.join(D, "routers.npz")) else {})
RES, rr, Wd = {}, None, 0                                    # (seq, L, T) -> offset of its residency words in rr
if os.path.exists(os.path.join(D, "res.bin")):
    rr, i, Wd = np.fromfile(os.path.join(D, "res.bin"), "<i4"), 0, meta["res"]["W"]
    while i < len(rr):
        seq, L, nT = int(rr[i]), int(rr[i + 1]), int(rr[i + 2])
        i += 3
        for _ in range(nT):
            RES.setdefault((seq, L, int(rr[i])), i + 1)
            i += 1 + Wd


def resident(k):
    """Expert residency [E] (bool) at the prefetch decision of (seq, L) for pool T, or None if not recorded."""
    o = RES.get(k)
    return None if o is None else np.unpackbits(rr[o:o + Wd].view(np.uint8), bitorder="little")[:meta["E"]].astype(bool)
res = {"miss_records": len(miss), "miss_dups": dup, "res_records": len(RES), "domains": {}}
bad_rows = bad_union = no_rec = rerun = no_res = 0
acc = {}                                                     # (domain, lead) -> [hit, miss]; (domain, "all") counters
acch = {}                                                    # lead -> the same counters on the held-out prompts (idx 4)
accf, accfh = {}, {}                                         # the same for the deployable (non-resident) top-16
KS = (2, 4, 8, 12, 16)                                       # read budgets: top-K among non-resident per row
bud, budh = {}, {}                                           # (lead, policy) -> [caught, missed, reads, pass-layers, unused]


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


for f in sorted(glob.glob(os.path.join(D, "p*.npz"))):
    p = dict(np.load(f))
    q = next(q for q in meta["prompts"] if q["name"] == os.path.basename(f)[:-4])
    dom, held = q["domain"], q["idx"] == 4
    dd = res["domains"].setdefault(dom, {"prompts": 0, "passes": 0, "rows": 0, "row_layers": 0, "routed": 0, "missed": 0})
    dd["prompts"] += 1
    by = {}
    for j in range(len(p["seq"])):
        calls = by.setdefault(int(p["seq"][j]), {})
        if int(p["L"][j]) in calls:
            rerun += 1                                       # the commit's layer-1 re-run: keep the verify call
        else:
            calls[int(p["L"][j])] = j
    for seq, calls in by.items():
        if not all(L in calls for L in range(NL)):
            continue                                         # MTP draft calls (L = NL) and anything partial
        R = {int(p["rows"][calls[L]]) for L in range(NL)}
        if len(R) != 1:
            bad_rows += 1
            continue
        R = R.pop()
        dd["passes"] += 1; dd["rows"] += R
        ids = {L: p["ids"][p["row_off"][calls[L]]:p["row_off"][calls[L]] + R].astype(np.int64) for L in range(NL)}
        mset = {}
        for L in range(NL):
            m = miss.get((seq, L))
            if m is None:
                no_rec += 1
                continue
            if set(m[0].tolist()) != set(ids[L].ravel().tolist()):
                bad_union += 1
            mset[L] = set(m[0][m[1] == 1].tolist())
            dd["row_layers"] += R; dd["routed"] += ids[L].size
            dd["missed"] += sum(len(set(r.tolist()) & mset[L]) for r in ids[L])
        for L in LP:
            jo = int(p["x_off"][calls[L]])
            if jo < 0 or not W:
                continue
            x = bf16(p["x"][jo:jo + R])
            for lead in (1, 2, 3):
                T = L + lead
                if T >= NL or f"w{T}" not in W or T not in mset:
                    continue
                lg = x @ W[f"w{T}"]
                top = np.argpartition(-lg, 15, axis=-1)[:, :16]
                rb = resident((seq, L, T))
                if RES and rb is None:
                    no_res += 1
                topf = None if rb is None else np.argpartition(-np.where(rb[None, :], -np.inf, lg), 15, axis=-1)[:, :16]
                a = acc.setdefault((dom, lead), [0, 0, 0, 0])   # hits on misses, misses, hits on routed, routed
                h = acch.setdefault(lead, [0, 0, 0, 0]) if held else [0, 0, 0, 0]
                af = accf.setdefault((dom, lead), [0, 0]) if topf is not None else [0, 0]
                hf = accfh.setdefault(lead, [0, 0]) if topf is not None and held else [0, 0]
                for r in range(R):
                    rm = set(ids[T][r].tolist()) & mset[T]
                    tp = set(top[r].tolist())
                    inc = (len(rm & tp), len(rm), len(set(ids[T][r].tolist()) & tp), ids[T].shape[1])
                    for c in range(4):
                        a[c] += inc[c]; h[c] += inc[c]
                    if topf is not None:
                        hit = len(rm & set(topf[r].tolist()))
                        af[0] += hit; af[1] += len(rm); hf[0] += hit; hf[1] += len(rm)
                if rb is not None:                           # read budget: what the SSD would see per (pass, T)
                    RT = set(ids[T].ravel().tolist())           # routed at T by any row of the pass
                    M = RT & mset[T]                             # the pass's misses at T
                    order = np.argsort(-np.where(rb[None, :], -np.inf, lg), axis=-1, kind="stable")
                    pols = [(f"k{K}", order[:, :K]) for K in KS]
                    if lead == 1:                            # the shipped engine: top-12 over all 512, resident skipped
                        pols.append(("engine12", np.argpartition(-lg, 11, axis=-1)[:, :12]))
                    for name, pk in pols:
                        U = {e for e in pk.ravel().tolist() if not rb[e]}
                        for bag in (bud, budh if held else None):
                            if bag is not None:
                                v = bag.setdefault((lead, name), [0, 0, 0, 0, 0])
                                v[0] += len(U & M); v[1] += len(M); v[2] += len(U); v[3] += 1; v[4] += len(U - RT)
res.update(bad_rows=bad_rows, bad_union=bad_union, no_record=no_rec, rerun_calls=rerun, no_res=no_res)
if RES:
    print(f"res.bin: {len(RES)} (pass, L -> T) residency records; scored (pass, L -> T) without one: {no_res}")
print(f"miss.bin: {len(miss)} (pass, layer) records, {dup} duplicates; verify passes with unequal rows: {bad_rows}; "
      f"trace/ext union mismatches: {bad_union}; (pass, layer) without an ext record: {no_rec}; "
      f"later calls of a layer in the same pass (commit re-runs, skipped): {rerun}")
tot = {"passes": 0, "rows": 0}
for dom, dd in res["domains"].items():
    tot["passes"] += dd["passes"]; tot["rows"] += dd["rows"]
    mr = dd["missed"] / max(dd["routed"], 1)
    line = f"[{dom}] {dd['prompts']} prompts, {dd['passes']} verify passes, {dd['rows']} rows; {100 * mr:.1f} % of routed ids missed"
    for lead in (1, 2, 3):
        a = acc.get((dom, lead))
        if a and a[1]:
            dd[f"recall16_miss_lead{lead}"] = a[0] / a[1]; dd[f"recall16_all_lead{lead}"] = a[2] / a[3]
            line += f"; lead {lead}: recall@16 of misses {100 * a[0] / a[1]:.1f} % (of all routed {100 * a[2] / a[3]:.1f} %)"
    print(line)
for lead in (1, 2, 3):
    a = np.sum([v for (dm, ld), v in acc.items() if ld == lead], axis=0) if any(ld == lead for _, ld in acc) else None
    if a is not None and a[1]:
        res[f"recall16_miss_lead{lead}"] = float(a[0] / a[1])
        print(f"all domains, lead {lead}: plain router recall@16 of misses {100 * a[0] / a[1]:.1f} %")
for lead in (1, 2, 3):
    a = acch.get(lead)
    if a and a[1]:
        res[f"heldout_recall16_miss_lead{lead}"] = a[0] / a[1]
        print(f"held-out (prompt 4 of each domain), lead {lead}: plain router recall@16 of misses {100 * a[0] / a[1]:.1f} %")
for lead in (1, 2, 3):                                       # b32: the deployable score (top-16 among non-resident)
    per = {dm: v for (dm, ld), v in accf.items() if ld == lead and v[1]}
    if not per:
        continue
    tot_f = np.sum(list(per.values()), axis=0)
    hb = accfh.get(lead)
    res[f"deploy_recall16_miss_lead{lead}"] = float(tot_f[0] / tot_f[1])
    if hb and hb[1]:
        res[f"deploy_heldout_recall16_miss_lead{lead}"] = hb[0] / hb[1]
    print(f"DEPLOYABLE lead {lead} (top-16 among non-resident at the decision): all prompts {100 * tot_f[0] / tot_f[1]:.1f} %"
          + (f", held-out {100 * hb[0] / hb[1]:.1f} %" if hb and hb[1] else "") + "; "
          + ", ".join(f"{dm} {100 * v[0] / v[1]:.1f}" for dm, v in per.items()))
    for dm, v in per.items():
        res["domains"][dm][f"deploy_recall16_miss_lead{lead}"] = v[0] / v[1]
if bud:                                                      # per (pass, T): the pass's misses vs the unique reads
    print("BUDGET (plain router; per pass and target layer: recall of the pass's misses, unique non-resident reads"
          " (of them never routed at T in the pass), precision = caught / reads; all prompts | held-out):")
    for (lead, name), v in sorted(bud.items(), key=lambda kv: (kv[0][0], kv[0][1] != "engine12", len(kv[0][1]), kv[0][1])):
        hv = budh.get((lead, name), [0, 1, 0, 1, 0])
        res[f"budget_lead{lead}_{name}"] = dict(recall=v[0] / max(v[1], 1), reads=v[2] / max(v[3], 1),
                                                unused=v[4] / max(v[3], 1), misses=v[1] / max(v[3], 1),
                                                heldout_recall=hv[0] / max(hv[1], 1))
        print(f"  lead {lead} {name:>8}: recall {100 * v[0] / max(v[1], 1):5.1f} % | {100 * hv[0] / max(hv[1], 1):5.1f} %,"
              f" reads {v[2] / max(v[3], 1):5.1f} ({v[4] / max(v[3], 1):5.1f} unused) vs misses {v[1] / max(v[3], 1):4.1f}"
              f" per pass-layer, precision {100 * v[0] / max(v[2], 1):4.1f} %")
print(f"total: {tot['passes']} verify passes, {tot['rows']} verify rows")
res["total"] = tot
json.dump(res, open(os.path.join(D, "check_ld.json" if LD else "check.json"), "w"), indent=1)
