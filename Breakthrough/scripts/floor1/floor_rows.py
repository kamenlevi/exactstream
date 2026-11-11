#!/usr/bin/env python3 -I
"""floor_rows.py (FableGod, mission 13, Mon 2 Nov 2026). CPU only, plain json, no numpy.
Re-reads the decision map's night-1 / night-2 runs (~/FableGod/decision_map/out/*.json: per answer row the top-8 token
ids and log-probs under each perturbation) per ROW:
  1. sanity: flip % by the 125B's own class (sure / tie / mid) vs report.md;
  2. flip % by the DRAFTER's confidence band (gate_dfl2.json: DFlash2's cf for the draft the row verifies): the
     realizable re-classing of every measured cut (caveat: every row ran perturbed, so these are upper bounds);
  3. an additive-margin model for COMBINED cuts (margin loss per cut on the baseline token, summed), validated on the
     one measured combination (12-35 at once vs its halves g1 + g2);
  4. the per-row oracle floor (cheapest level per row that keeps the argmax) and the realizable floor (cheapest level
     per drafter band whose flip rate is under a bar), in expert records per token;
  5. answers touched per scheme, the list of code-sure top-2 flips, and the sure-run Markov structure.
"""
import json, math, os, sys
from collections import defaultdict, Counter

OUT = os.path.expanduser("~/Breakthrough/data/floor1")
base = json.load(open(f"{OUT}/baseline.json"))
gate = json.load(open(f"{OUT}/gate_dfl2.json"))
PERTS = ["topk2", "topk2r", "topk5", "shared_zero", "moe_reuse_prev_g0", "moe_reuse_prev_g1", "moe_reuse_prev_g2",
         "moe_reuse_prev_g3", "moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35", "moe_reuse_prev_L36_39",
         "moe_reuse_lag2_L12_35", "moe_reuse_lag4_L12_35", "moe_reuse_chain_L12_35", "att_reuse_prev", "gdn_reuse_prev",
         "hc_static", "hc_q4", "gdn_q4", "att_q4"]
EXTRA = ["moe_reuse_chain1_L12_35", "moe_reuse_chain1_L08_35", "moe_reuse_chain2_L08_35", "moe_reuse_chain1_L08_35_topk2",
         "moe_reuse_prev_L04_07", "moe_reuse_prev_L00_03"]
P = {}
for p in PERTS + EXTRA:
    f = f"{OUT}/{p}.json"
    if os.path.exists(f): P[p] = json.load(open(f))
print("loaded perturbations:", ", ".join(P))
BANDS = [(0.999, ">=0.999"), (0.99, "0.99-0.999"), (0.9, "0.9-0.99"), (-1.0, "<0.9")]
DOMS = ["code", "ifeval", "multi"]
CLS = ["sure", "tie", "mid"]

def band_of(cf):
    if cf is None: return "none"
    for lo, lab in BANDS:
        if cf >= lo: return lab
    return "<0.9"

rows = []
for nm, b in base.items():
    a, dom = b["a"], b["dom"]
    g = gate[nm]
    for j, (tk, lp) in enumerate(zip(b["tk"], b["lp"])):
        p1, p2 = math.exp(lp[0]), math.exp(lp[1])
        cls = "sure" if p1 >= 0.99 else ("tie" if p1 - p2 <= 0.1 else "mid")
        cf = g[a + j] if a + j < len(g) else None
        r = dict(nm=nm, dom=dom, j=j, i=a + j, cls=cls, cf=cf, band=band_of(cf), t0=tk[0], p1=p1, m0=lp[0] - lp[1],
                 flip={}, m={})
        for p, D in P.items():
            w = D.get(nm)
            if w is None: continue
            tkp, lpp = w["tk"][j], w["lp"][j]
            r["flip"][p] = tkp[0] != tk[0]
            if tk[0] in tkp:
                k = tkp.index(tk[0]); l0 = lpp[k]
                other = max(v for x, v in enumerate(lpp) if x != k)
                r["m"][p] = l0 - other
            else:
                r["m"][p] = lpp[-1] - lpp[0] - 1.0          # not in the top-8: a bound, the flip is certain
        rows.append(r)
print("rows:", len(rows), Counter(r["dom"] for r in rows))

def pct(n, d): return 100.0 * n / d if d else float("nan")
def table(title, header, lines):
    print(f"\n### {title}\n| {' | '.join(header)} |\n|{'---|' * len(header)}")
    for l in lines: print(f"| {' | '.join(str(x) for x in l)} |")

# 1. sanity vs report.md ------------------------------------------------------------------------------------------
lines = []
for p in ["topk2", "topk5", "moe_reuse_prev_L12_35", "moe_reuse_chain_L12_35", "hc_static", "moe_reuse_prev_L08_11"]:
    if p not in P: continue
    cells = [p]
    for d in DOMS:
        c = []
        for cl in CLS:
            R = [r for r in rows if r["dom"] == d and r["cls"] == cl and p in r["flip"]]
            c.append(f"{pct(sum(r['flip'][p] for r in R), len(R)):.2f}")
        cells.append(" / ".join(c))
    lines.append(cells)
table("1. sanity: flip % sure / tie / mid (must equal report.md)", ["cut", "code", "ifeval", "multi"], lines)

# 2. by drafter band ------------------------------------------------------------------------------------------------
BL = [l for _, l in BANDS] + ["none"]
for d in DOMS:
    lines = []
    R0 = [r for r in rows if r["dom"] == d]
    lines.append(["rows (share %)"] + [f"{len([r for r in R0 if r['band'] == bl])} ({pct(len([r for r in R0 if r['band'] == bl]), len(R0)):.1f})" for bl in BL]
                 + [f"{len([r for r in R0 if r['cf'] is not None and r['cf'] >= 0.99])} ({pct(len([r for r in R0 if r['cf'] is not None and r['cf'] >= 0.99]), len(R0)):.1f})", str(len(R0))])
    lines.append(["oracle-sure share %"] + [f"{pct(len([r for r in R0 if r['band'] == bl and r['cls'] == 'sure']), len([r for r in R0 if r['band'] == bl])):.1f}" for bl in BL]
                 + [f"{pct(len([r for r in R0 if r['cf'] is not None and r['cf'] >= 0.99 and r['cls'] == 'sure']), len([r for r in R0 if r['cf'] is not None and r['cf'] >= 0.99])):.1f}", f"{pct(len([r for r in R0 if r['cls'] == 'sure']), len(R0)):.1f}"])
    for p in ["topk2", "topk5", "moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35", "moe_reuse_prev_g1", "moe_reuse_prev_g2",
              "moe_reuse_chain_L12_35", "moe_reuse_prev_L36_39", "shared_zero", "hc_static", "hc_q4", "gdn_q4", "att_q4",
              "moe_reuse_chain1_L12_35", "moe_reuse_chain1_L08_35", "moe_reuse_chain2_L08_35", "moe_reuse_chain1_L08_35_topk2"]:
        if p not in P: continue
        cells = [p]
        for bl in BL + ["cf>=0.99", "all"]:
            if bl == "cf>=0.99": R = [r for r in R0 if r["cf"] is not None and r["cf"] >= 0.99 and p in r["flip"]]
            elif bl == "all": R = [r for r in R0 if p in r["flip"]]
            else: R = [r for r in R0 if r["band"] == bl and p in r["flip"]]
            n = sum(r["flip"][p] for r in R)
            cells.append(f"{pct(n, len(R)):.2f} ({n})")
        lines.append(cells)
    table(f"2. {d}: flip % (flips) by DRAFTER band", ["cut"] + BL + ["cf>=0.99", "all"], lines)

# 3. additive-margin model, validated on 12-35 vs g1 + g2 -----------------------------------------------------------
def add_pred(r, parts):
    """predicted margin of the baseline token after all cuts in `parts`, first order: m0 - sum(m0 - m_p)."""
    if any(p not in r["m"] for p in parts): return None
    return r["m0"] - sum(r["m0"] - r["m"][p] for p in parts)

if all(p in P for p in ["moe_reuse_prev_g1", "moe_reuse_prev_g2", "moe_reuse_prev_L12_35"]):
    lines = []
    for d in DOMS:
        for cl in CLS:
            R = [r for r in rows if r["dom"] == d and r["cls"] == cl]
            meas = sum(r["flip"]["moe_reuse_prev_L12_35"] for r in R)
            union = sum(r["flip"]["moe_reuse_prev_g1"] or r["flip"]["moe_reuse_prev_g2"] for r in R)
            addp = [add_pred(r, ["moe_reuse_prev_g1", "moe_reuse_prev_g2"]) for r in R]
            pred = sum(1 for v in addp if v is not None and v < 0)
            both = sum(1 for r, v in zip(R, addp) if v is not None and v < 0 and r["flip"]["moe_reuse_prev_L12_35"])
            lines.append([d, cl, len(R), f"{pct(meas, len(R)):.2f} ({meas})", f"{pct(union, len(R)):.2f} ({union})",
                          f"{pct(pred, len(R)):.2f} ({pred})", f"{both}/{meas} measured flips predicted"])
    table("3. additive-margin model vs the measured 12-35 run (lag 1)", ["dom", "class", "n", "measured 12-35", "union of g1 / g2 flips", "additive prediction", "hits"], lines)
    # calibration: measured margin change of 12-35 vs the sum of the halves' changes, on sure rows
    for d in DOMS:
        R = [r for r in rows if r["dom"] == d and r["cls"] == "sure"]
        s_sum = sum((r["m0"] - r["m"]["moe_reuse_prev_g1"]) + (r["m0"] - r["m"]["moe_reuse_prev_g2"]) for r in R) / len(R)
        s_meas = sum(r["m0"] - r["m"]["moe_reuse_prev_L12_35"] for r in R) / len(R)
        print(f"margin loss on {d}-sure rows: g1 + g2 summed {s_sum:.3f} nats, 12-35 measured {s_meas:.3f} nats, ratio {s_meas / s_sum:.2f}")

# 4. predicted combinations (additive) by class and band ------------------------------------------------------------
COMBOS = {
    "top2 + reuse 8-35 (lag 1)": ["topk2", "moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35"],
    "top2 + reuse 12-35": ["topk2", "moe_reuse_prev_L12_35"],
    "top5 + reuse 8-35": ["topk5", "moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35"],
    "reuse 8-35 (lag 1)": ["moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35"],
    "top2 + 4-bit att/gdn/hc outputs": ["topk2", "att_q4", "gdn_q4", "hc_q4"],
    "top2 + reuse 8-35 + 4-bit outputs": ["topk2", "moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35", "att_q4", "gdn_q4", "hc_q4"],
    "top2 + hc_static": ["topk2", "hc_static"],
}
for d in DOMS:
    lines = []
    for name, parts in COMBOS.items():
        if any(p not in P for p in parts): continue
        cells = [name]
        for cl in CLS:
            R = [r for r in rows if r["dom"] == d and r["cls"] == cl]
            pr = [add_pred(r, parts) for r in R]
            un = sum(any(r["flip"][p] for p in parts) for r in R)
            cells.append(f"{pct(sum(1 for v in pr if v is not None and v < 0), len(R)):.2f} / {pct(un, len(R)):.2f}")
        for bl in [">=0.999", "0.99-0.999", "0.9-0.99", "<0.9"]:
            R = [r for r in rows if r["dom"] == d and r["band"] == bl]
            pr = [add_pred(r, parts) for r in R]
            cells.append(f"{pct(sum(1 for v in pr if v is not None and v < 0), len(R)):.2f}")
        lines.append(cells)
    table(f"4. {d}: predicted flip % of combinations (additive / union-of-measured) by class, then additive by drafter band",
          ["combination", "sure", "tie", "mid", ">=0.999", "0.99-0.999", "0.9-0.99", "<0.9"], lines)

# 5. per-row oracle floor and realizable floor, in expert records per token ----------------------------------------
# levels: (k, reuse 8-35) ; records per row = k * (48 - 28 * reuse). Bytes: 4-bit record 2.8 MB; 45 GB tiers: ranks 0-1
# at 2.8 MB, ranks 2-9 at 2.1 MB (TRIED: top-10 x 48 x 2.2 MB = 1.06 GB per token).
def rec_mb(k, reuse, tiers):
    layers = 48 - 28 * reuse
    if not tiers: return k * layers * 2.8
    return layers * (min(k, 2) * 2.8 + max(k - 2, 0) * 2.1)

def ok(r, k, reuse):
    parts = []
    if k == 2: parts.append("topk2")
    elif k == 5: parts.append("topk5")
    if reuse: parts += ["moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35"]
    if not parts: return True
    if len(parts) == 1: return not r["flip"][parts[0]]
    v = add_pred(r, parts)
    return v is not None and v >= 0 and not any(r["flip"][p] for p in parts)

LEVELS = [(2, 1), (2, 0), (5, 1), (5, 0), (10, 1), (10, 0)]   # cheapest first
need = ["topk2", "topk5", "moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35"]
if all(p in P for p in need):
    # oracle: per row the cheapest level that keeps the argmax; reuse only if the previous row did not reuse (S = 1)
    by_nm = defaultdict(list)
    for r in rows: by_nm[r["nm"]].append(r)
    for nm, R in by_nm.items():
        R.sort(key=lambda r: r["j"]); prev_reuse = False
        for r in R:
            for k, re in LEVELS:
                if re and prev_reuse: continue
                if ok(r, k, re):
                    r["oracle"] = (k, re); break
            prev_reuse = r["oracle"][1] == 1
    lines = []
    for d in DOMS:
        for grp, sel in [("all", lambda r: True), ("sure", lambda r: r["cls"] == "sure"), ("tie", lambda r: r["cls"] == "tie"),
                         ("mid", lambda r: r["cls"] == "mid")] + [(f"band {bl}", (lambda bl: lambda r: r["band"] == bl)(bl)) for bl in BL]:
            R = [r for r in rows if r["dom"] == d and sel(r)]
            if not R: continue
            cnt = Counter(r["oracle"] for r in R)
            mb4 = sum(rec_mb(*r["oracle"], False) for r in R) / len(R)
            mbt = sum(rec_mb(*r["oracle"], True) for r in R) / len(R)
            recs = sum(r["oracle"][0] * (48 - 28 * r["oracle"][1]) for r in R) / len(R)
            lines.append([d, grp, len(R), " ".join(f"{k}{'R' if re else ''}:{pct(cnt[(k, re)], len(R)):.0f}%" for k, re in LEVELS),
                          f"{recs:.0f}", f"{mb4:.0f}", f"{mbt:.0f}"])
    table("5a. ORACLE floor per row (cheapest level keeping the argmax; R = reuse 8-35 from the previous row, S=1): level shares, expert records per token (exact = 480), MB per token at 4-bit / at the 45 GB tiers (exact = 1344 / 1056)",
          ["dom", "rows", "n", "levels", "records", "MB 4-bit", "MB tiers"], lines)
    # realizable: one level per drafter band, chosen as the cheapest whose flip rate on the band is <= bar
    for bar in (1.0, 0.3):
        lines = []
        for d in DOMS:
            tot_mb = 0.0; tot_n = 0; parts_desc = []
            for bl in BL:
                R = [r for r in rows if r["dom"] == d and r["band"] == bl]
                if not R: continue
                chosen = (10, 0); rate = 0.0
                for k, re in LEVELS:
                    if re:  # a band-level reuse: every other row in the band's runs; here the rate is the lag-1 rate
                        pr = [ok(r, k, 1) for r in R]
                        fr = pct(sum(1 for v in pr if not v), len(R))
                    else:
                        fr = pct(sum(1 for r in R if not ok(r, k, 0)), len(R))
                    if fr <= bar:
                        chosen = (k, re); rate = fr; break
                k, re = chosen
                # with reuse the band's rows alternate: half reuse, half compute at the same k
                mb = (rec_mb(k, 1, True) + rec_mb(k, 0, True)) / 2 if re else rec_mb(k, 0, True)
                tot_mb += mb * len(R); tot_n += len(R)
                parts_desc.append(f"{bl}: {k}{'R' if re else ''} ({rate:.2f}%, {len(R)} rows, {mb:.0f} MB)")
            lines.append([d, f"bar {bar}%", "; ".join(parts_desc), f"{tot_mb / tot_n:.0f}"])
        table(f"5b. REALIZABLE floor by drafter band (level per band = cheapest with band flip % <= {bar}; MB per token at the tiers, exact = 1056)",
              ["dom", "bar", "band: level (flip %, rows, MB/row)", "MB per token"], lines)

# 6. answers (windows) touched per scheme ---------------------------------------------------------------------------
lines = []
schemes = {
    "oracle-sure rows at top-2 (Q1 as measured)": lambda r: r["cls"] == "sure" and r["flip"]["topk2"],
    "drafter cf>=0.99 rows at top-2 (Q1 realizable)": lambda r: r["cf"] is not None and r["cf"] >= 0.99 and r["flip"]["topk2"],
    "drafter cf>=0.999 rows at top-2": lambda r: r["cf"] is not None and r["cf"] >= 0.999 and r["flip"]["topk2"],
    "NEW-1 bands (>=0.999 top-2, 0.99-0.999 top-5 proxy, rest exact)": lambda r: (r["cf"] is not None and ((r["cf"] >= 0.999 and r["flip"]["topk2"]) or (0.99 <= r["cf"] < 0.999 and r["flip"]["topk5"]))),
    "cf>=0.99 rows: top-2 + reuse 8-35 (additive, every row)": lambda r: r["cf"] is not None and r["cf"] >= 0.99 and not ok(r, 2, 1),
}
for name, f in schemes.items():
    cells = [name]
    for d in DOMS:
        W = defaultdict(int); n = 0; nr = 0
        for r in rows:
            if r["dom"] != d: continue
            nr += 1
            if f(r): W[r["nm"]] += 1; n += 1
        cells.append(f"{len(W)}/12 windows, {n} flips ({pct(n, nr):.2f} % of all rows)")
    lines.append(cells)
table("6. answers touched (teacher-forced windows with >= 1 flip)", ["scheme", "code", "ifeval", "multi"], lines)

# 7. the code-sure top-2 flips ---------------------------------------------------------------------------------------
print("\n### 7. code-sure rows that flip under top-2 (window, answer index, p1, baseline token id -> new token id, margin before / after)")
for r in rows:
    if r["dom"] == "code" and r["cls"] == "sure" and r["flip"]["topk2"]:
        w = P["topk2"][r["nm"]]
        print(f"- {r['nm']} j={r['j']} p1={r['p1']:.4f} {r['t0']} -> {w['tk'][r['j']][0]} margin {r['m0']:.2f} -> {r['m']['topk2']:.2f}; also flips under top-5: {r['flip']['topk5']}, reuse 12-35: {r['flip']['moe_reuse_prev_L12_35']}")
print("\n### 7b. per-window share of sure rows and the flips under top-2 / reuse 12-35 (code)")
for nm in sorted(by_nm):
    R = by_nm[nm]
    if R[0]["dom"] != "code": continue
    s = [r for r in R if r["cls"] == "sure"]
    print(f"- {nm}: rows {len(R)}, sure {pct(len(s), len(R)):.0f} %, cf>=0.99 {pct(sum(1 for r in R if r['cf'] is not None and r['cf'] >= 0.99), len(R)):.0f} %, top-2 flips sure/all {sum(r['flip']['topk2'] for r in s)}/{sum(r['flip']['topk2'] for r in R)}, reuse12-35 sure/all {sum(r['flip']['moe_reuse_prev_L12_35'] for r in s)}/{sum(r['flip']['moe_reuse_prev_L12_35'] for r in R)}")

# 8. Markov structure of sureness and of the drafter's band ---------------------------------------------------------
print("\n### 8. run structure")
for d in DOMS:
    ss = sm = ms = mm = 0; ds = dm = 0; prev = None
    for nm, R in by_nm.items():
        if R[0]["dom"] != d: continue
        for a_, b_ in zip(R, R[1:]):
            if a_["cls"] == "sure" and b_["cls"] == "sure": ss += 1
            elif a_["cls"] == "sure": sm += 1
            elif b_["cls"] == "sure": ms += 1
            else: mm += 1
            if a_["cf"] is not None and a_["cf"] >= 0.99:
                if b_["cls"] == "sure": ds += 1
                else: dm += 1
    print(f"{d}: P(sure_t | sure_t-1) = {pct(ss, ss + sm):.1f} %, P(sure_t | not sure_t-1) = {pct(ms, ms + mm):.1f} %; P(125B-sure_t | drafter cf_t-1 >= 0.99) = {pct(ds, ds + dm):.1f} %")
json.dump([{k: v for k, v in r.items() if k in ("nm", "dom", "j", "i", "cls", "cf", "band", "p1", "m0", "oracle")} | {"flip": r["flip"]} for r in rows],
          open(os.path.expanduser("~/Breakthrough/data/floor1/floor_rows.json"), "w"))
print("\nwrote ~/Breakthrough/data/floor1/floor_rows.json (per-row classes, bands, flips, oracle level)")
