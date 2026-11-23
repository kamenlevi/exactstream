"""cc3_report.py (Leap, Sat 14 Nov 2026; v2 21:2x): the report of cc3_runner.sh (WEEK98 route 3 screen 3). Per run tag:
  - the probe's passes ({tag}_probe.jsonl): d1 / d16 / d32 dense only (the routed-expert call stubbed; d16 = the check's
    own rows: 16 at 45 GB, <= 13 at 24 GB), s32 the real 32-row verify (slot path; at 24 GB with the fast gate off);
  - the loop's per-check records ({TAG}_checks.json) and warm-split samples ({TAG}_warmsplit.json). v2: both are read
    where the loop writes them, the clone's engine/data/r57 (run.py moves only tokens / runs into --out; v1 looked in the
    run folder and would have found neither), by the run's TAG (the cli_* name of its files in the run folder).
Split at the check's rows:
  wsp (45 GB, R57_WARMSPLIT): dense d16 + experts (warm16 - d16) + misses (cold16 - warm16) = cold16.
  own (24 GB, CC3_EVERY): the live check right after the probe (the stub passes load nothing) - d16 = experts + misses
       (its SSD wait and misses shown), joined on the check index ci; records with s32 are left out of the join.
In-run reference (the orchestrator 19:1x: compare only within the run): the live checks at the run's top row count,
leaving out each probed check and the one after it. Medians; per layer = / 48. A widened-gate 32-row estimate (45 GB) =
d32 + E16 + 16 x slope(8->16) x GROW + M16 x 1.5. v2.2 (21:4x): GROW = 0.67 from BigMoE's step 0 (64c9d2d: mean distinct
experts per layer 45.5 / 72.5 / 108.5 at windows 8 / 16 / 32, so new distinct per added row is 3.38 at 8-16 rows and 2.25
at 16-32). v1 used 0.72 (3.0 -> 2.15, D32 / L1-24). The misses' x 1.5 = the 32 / 16 union ratio (108.5 / 72.5 = 1.50).
Usage: cc3_report.py <data dir> [<loop out dir>]"""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1]
R57 = sys.argv[2] if len(sys.argv) > 2 else "/Users/kamen/Leap/release/cleanclone/v121/clone/engine/data/r57"
TAGS = ("a45", "a45b", "a45c", "b24", "b24b")


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def pl(x):
    return "-" if x is None else f"{x / 48:.2f}"


def jload(p, default):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


for tag in TAGS:
    pj = os.path.join(D, f"{tag}_probe.jsonl")
    rr = sorted(glob.glob(os.path.join(D, tag, "cli_*_runs.json")))
    if not os.path.exists(pj) and not rr:
        continue
    TAG = os.path.basename(rr[-1])[:-len("_runs.json")] if rr else None
    probes = [json.loads(x) for x in open(pj)] if os.path.exists(pj) else []
    checks = jload(os.path.join(R57, f"{TAG}_checks.json"), []) if TAG else []
    wsall = jload(os.path.join(R57, f"{TAG}_warmsplit.json"), []) if TAG else []
    txt = open(os.path.join(D, f"{tag}.txt")).read() if os.path.exists(os.path.join(D, f"{tag}.txt")) else ""
    agree = re.findall(r"tokens == REF2 row (\d+) / (\d+)", txt)
    peak = re.findall(r"lifetime peak ([0-9.]+) GB", txt)
    pmf = os.path.join(D, f"{tag}_pmode.txt")
    pmode = open(pmf).read().strip() if os.path.exists(pmf) else "power mode ?"
    errs = [p["err"] for p in probes if p.get("err")]
    trig = sorted({p.get("trig", "wsp") for p in probes})
    print(f"== {tag} ({TAG}): probe checks {len(probes)} (trigger {'/'.join(trig) or '-'}, ref_ok {sum(bool(p.get('ref_ok')) for p in probes)}, "
          f"loose {sum(bool(p.get('loose')) for p in probes)}), errors {len(errs)} {errs[:1]}; tokens == REF2 {agree}; "
          f"peak {peak[-1] if peak else '-'} GB; {pmode}")
    # the loop's per-check records by key; the probed checks (and the one after each) are left out of the live reference
    byk = {}
    for pr in checks:
        byk.setdefault(pr.get("key"), []).append(pr)
    skip = {}
    for p in probes:
        skip.setdefault(p["key"], set()).update({p.get("ci"), (p.get("ci") or 0) + 1})
    for d_ in wsall:
        for s_ in d_.get("samples", []):
            skip.setdefault(d_.get("key"), set()).update({s_.get("n"), (s_.get("n") or 0) + 1})
    top = max((r_ for pr in checks for r_ in (pr.get("rows") or [])), default=None)
    live = [(c_, m_, io_) for k_, prs in byk.items() for pr in prs
            for i_, (r_, c_, m_, io_) in enumerate(zip(pr.get("rows") or [], pr.get("ck") or [], pr.get("miss") or [], pr.get("io") or []))
            if r_ == top and i_ not in skip.get(k_, set())]
    lref = med([x[0] for x in live])
    print(f"   in-run reference: live checks at {top} rows (probed checks and the next left out) median {f(lref)} ms "
          f"(n {len(live)}; misses {f(med([x[1] for x in live]), 0)}, SSD wait {f(med([x[2] for x in live]))} ms); per layer {pl(lref)}")
    # the probe's passes
    Pp = {}
    for p in probes:
        for q in p.get("passes", []):
            Pp.setdefault(q["p"], []).append(q)
    pm = {k: med([q["ms"] for q in v]) for k, v in Pp.items()}
    prow = {k: med([q["rows"] for q in v]) for k, v in Pp.items()}
    print(f"   probe: dense only 1 / {f(prow.get('d16'), 0)} / 32 rows {f(pm.get('d1'))} / {f(pm.get('d16'))} / {f(pm.get('d32'))} ms "
          f"(n {len(Pp.get('d1', []))} / {len(Pp.get('d16', []))} / {len(Pp.get('d32', []))}); real 32 rows (slot path"
          f"{', fast gate off' if 'own' in trig else ''}) {f(pm.get('s32'))} ms, misses {f(med([q['miss'] for q in Pp.get('s32', [])]), 0)}, "
          f"SSD wait {f(med([q['io'] for q in Pp.get('s32', [])]))} ms (n {len(Pp.get('s32', []))})")
    print("   per layer (/ 48): " + ", ".join(f"{k} {pl(pm.get(k))}" for k in ("d1", "d16", "d32", "s32") if pm.get(k) is not None))
    if wsall:   # 45 GB: the loop's warm split
        W = {"cold": [], "cmiss": [], "w1": [], "w4": [], "w8": [], "w16": []}
        for d_ in wsall:
            for s_ in d_.get("samples", []):
                c = s_.get("cold")
                if c:
                    W["cold"].append(c[1]); W["cmiss"].append(c[2])
                for w in s_.get("warm", []):
                    k = {1: "w1", 4: "w4", 8: "w8"}.get(w[0], "w16" if w[0] >= 13 else None)
                    if k:
                        W[k].append(w[1])
        m = {k: med(v) for k, v in W.items()}
        print(f"   warm split (n {len(W['cold'])}): cold 16 {f(m['cold'])} ms (misses {f(m['cmiss'], 0)}); warm 1 / 4 / 8 / 16 rows "
              f"{f(m['w1'])} / {f(m['w4'])} / {f(m['w8'])} / {f(m['w16'])} ms")
        E1 = m["w1"] - pm["d1"] if m["w1"] is not None and pm.get("d1") is not None else None
        E16 = m["w16"] - pm["d16"] if m["w16"] is not None and pm.get("d16") is not None else None
        M16 = m["cold"] - m["w16"] if m["cold"] is not None and m["w16"] is not None else None
        sl = (m["w16"] - m["w8"]) / 8 if m["w16"] is not None and m["w8"] is not None else None
        print(f"   split at 16 rows: dense {f(pm.get('d16'))} + experts {f(E16)} + misses {f(M16)} = cold {f(m['cold'])} ms "
              f"(per layer {pl(pm.get('d16'))} + {pl(E16)} + {pl(M16)}); 1 row: dense {f(pm.get('d1'))} + experts {f(E1)}; "
              f"expert slope 8->16 {f(sl, 2)} ms per row")
        if pm.get("d32") is not None and E16 is not None and sl is not None and M16 is not None:
            GROW = ((108.5 - 72.5) / 16) / ((72.5 - 45.5) / 8)   # 0.67: BigMoE step 0 (64c9d2d), mean distinct at windows 8 / 16 / 32
            est = pm["d32"] + E16 + 16 * sl * GROW + 1.5 * M16
            # v3 (Mon 16 Nov 03:4x): est is a forward pass built from probe parts, so like for like it is over the probe's own
            # cold 16-row pass (d16 + E16 + M16). v2 printed only est / lref, the whole live check (the loop's per-check
            # work and its extra misses included): a mixed basis that read low (a45: 1.35x vs 1.70x like for like).
            xo = lref - m["cold"] if lref and m["cold"] is not None else None
            print(f"   32 rows: two 16-row passes ~{f(2 * m['cold'])} ms; the slot path (measured) {f(pm.get('s32'))} ms; a gate "
                  f"carrying 320 ids ~{f(est)} ms (estimate) = {f(est / m['cold'] if m['cold'] else None, 2)}x the cold 16-row "
                  f"pass, like for like (line 1.35-1.4x); {f((est + xo) / lref if xo is not None else None, 2)}x the in-run 16-row "
                  f"check with its extra {f(xo)} ms over the cold pass added to both (v2's mixed basis est / in-run: "
                  f"{f(est / lref if lref else None, 2)}x); dense 32 / 16 = {f(pm['d32'] / pm['d16'], 2)}")
    if "own" in trig:   # 24 GB: the live check right after the probe
        J = []
        for p in probes:
            if p.get("err") or any(q["p"] == "s32" for q in p.get("passes", [])):
                continue
            d16 = next((q for q in p.get("passes", []) if q["p"] == "d16"), None)
            for pr in byk.get(p["key"], []):
                i_ = p.get("ci")
                if d16 and i_ is not None and i_ < len(pr.get("ck") or []) and pr["rows"][i_] == d16["rows"]:
                    J.append((pr["rows"][i_], pr["ck"][i_], d16["ms"], pr["miss"][i_], pr["io"][i_]))
        for lab_, JJ in ((f"{top} rows", [x for x in J if x[0] == top]), ("all rows", J)):
            if not JJ:
                print(f"   split at {lab_}: no joined checks")
                continue
            em = med([x[1] - x[2] for x in JJ])
            print(f"   split at {lab_} (n {len(JJ)}, rows {f(med([x[0] for x in JJ]), 0)}): live check {f(med([x[1] for x in JJ]))} ms = "
                  f"dense {f(med([x[2] for x in JJ]))} + experts and misses {f(em)} (misses {f(med([x[3] for x in JJ]), 0)}, SSD wait "
                  f"{f(med([x[4] for x in JJ]))} ms); per layer {pl(med([x[1] for x in JJ]))} = {pl(med([x[2] for x in JJ]))} + {pl(em)}")
    # where each pass's time goes: graph build (CPU) vs wait; thread CPU in each
    for k in ("d1", "d16", "d32", "s32"):
        qs = Pp.get(k, [])
        if qs:
            print(f"   {k}: build {f(med([q.get('bld') for q in qs]))} ms (thread CPU {f(med([q.get('cpu_bld') for q in qs]))}) + wait "
                  f"{f(med([q.get('wt') for q in qs]))} ms (thread CPU {f(med([q.get('cpu_wt') for q in qs]))})")
