"""B12_FOLD on vs off, row by row (docs/PREREG_fold_2026-10-27.md): tokens, decode tok/s and prefill time (paired,
rows with a PAUSE or > 1 s frozen in either arm left out of the timing), footprint against 11.95 GB, C_m, SSD MB per
token. Usage: .venv/bin/python tools/fold_compare.py [OFF ON]   (default b12llfoff b12llfon; results/<v>/decode.jsonl)"""
import json, math, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF, ON = (sys.argv[1:3] if len(sys.argv) > 2 else ("b12llfoff", "b12llfon"))
LINE = 11.95


def rows(v):
    p = os.path.join(ROOT, "results", v, "decode.jsonl")
    return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def chunks(v):
    p = os.path.join(ROOT, "results", v, "chunks.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def peak(r):
    return max(r["mem"]["max"], r["mem"]["life"]) if "mem" in r else float("nan")


def timed(r):
    m = r.get("mem", {})
    return m.get("paused_s", 0) == 0 and m.get("frozen_s", 0) <= 1 and r.get("ssd", {}).get("dec_s")


a, b = rows(OFF), rows(ON)
keys = [k for k in a if k in b]
sets = sorted({a[k]["set"] for k in keys}, key=["agent", "math", "ifeval", "humaneval"].index)
print(f"{OFF} (off) vs {ON} (on): {len(keys)} paired rows ({len(a)} / {len(b)} rows)\n")

print("1. tokens")
same = 0
for s in sets:
    ks = [k for k in keys if a[k]["set"] == s]
    diff = sorted((first_diff(a[k]["completion_ids"], b[k]["completion_ids"]), k) for k in ks
                  if a[k]["completion_ids"] != b[k]["completion_ids"])
    same += len(ks) - len(diff)
    print(f"  {s:9s} {len(ks) - len(diff)}/{len(ks)} token-identical"
          + (f"; differ: {', '.join(f'{k} @ {i}' for i, k in diff[:8])}" if diff else ""))
print(f"  all       {same}/{len(keys)} token-identical\n")

print("2. speed (paired rows timed in both arms; on/off: > 1 = fold on faster or larger)")
print(f"  {'set':9s} {'rows':>4s} {'dec tok/s off':>13s} {'on':>6s} {'gmean on/off':>12s} {'total dec s off':>15s} {'on':>7s}"
      f" {'prefill s off':>13s} {'on':>7s} {'gmean pf on/off':>15s}")
for s in sets + ["all"]:
    ks = [k for k in keys if (s == "all" or a[k]["set"] == s) and timed(a[k]) and timed(b[k])]
    if not ks:
        continue
    na, nb = sum(a[k]["ssd"]["n"] for k in ks), sum(b[k]["ssd"]["n"] for k in ks)
    da, db = sum(a[k]["ssd"]["dec_s"] for k in ks), sum(b[k]["ssd"]["dec_s"] for k in ks)
    pa, pb = sum(a[k]["ssd"]["pf_s"] for k in ks), sum(b[k]["ssd"]["pf_s"] for k in ks)
    r_dec = gmean([(b[k]["ssd"]["n"] / b[k]["ssd"]["dec_s"]) / (a[k]["ssd"]["n"] / a[k]["ssd"]["dec_s"]) for k in ks])
    r_pf = gmean([b[k]["ssd"]["pf_s"] / a[k]["ssd"]["pf_s"] for k in ks if a[k]["ssd"]["pf_s"] > 0 and b[k]["ssd"]["pf_s"] > 0])
    print(f"  {s:9s} {len(ks):4d} {na / da:13.2f} {nb / db:6.2f} {r_dec:12.3f} {da:15.0f} {db:7.0f} {pa:13.1f} {pb:7.1f} {r_pf:15.3f}")
untimed = [k for k in keys if not (timed(a[k]) and timed(b[k]))]
print(f"  left out of the timing (PAUSE / frozen in either arm): {len(untimed)} rows\n")

print(f"3. footprint (GB; row peak = max of the 25 ms sampled max and the lifetime peak after the row; line {LINE})")
for v, R in ((OFF, a), (ON, b)):
    pk = sorted(((peak(r), k) for k, r in R.items()), reverse=True)
    over = [(p, k) for p, k in pk if p > LINE]
    print(f"  {v}: max row peak {pk[0][0]:.2f} ({pk[0][1]}); rows over {LINE}: {len(over)}"
          + (f" -> {', '.join(f'{k} {p:.2f} (n {R[k]['n_prompt']}, C_m {R[k].get('ssd', {}).get('cm')})' for p, k in over[:6])}" if over else ""))
    for s in sets:
        ps = [peak(r) for r in R.values() if r["set"] == s]
        st = [r["mem"]["start"] for r in R.values() if r["set"] == s and "mem" in r]
        print(f"    {s:9s} row peak median {statistics.median(ps):.2f} max {max(ps):.2f}; footprint at row start "
              f"min {min(st):.2f} max {max(st):.2f}")
    print("    chunk lifetime peaks: " + ", ".join(f"{c['footprint_peak_gb']:.2f}" for c in chunks(v)))
print()

print("4. C_m, ALLWAIT, PFR (fold on)")
for s in sets:
    rs = [b[k] for k in keys if b[k]["set"] == s and "ssd" in b[k]]
    cms = [r["ssd"].get("cm") for r in rs]
    print(f"  {s:9s} C_m {min(cms)}-{max(cms)}; ALLWAIT layer calls per row median "
          f"{statistics.median(r['ssd'].get('aw', 0) for r in rs):.0f}; rows with a PFR release "
          f"{sum(bool(r['ssd'].get('pfr')) for r in rs)}/{len(rs)}")
print()

print("5. SSD, decode phase (MB / token, reads / token)")
for s in sets + ["all"]:
    ks = [k for k in keys if s == "all" or a[k]["set"] == s]
    line = []
    for R in (a, b):
        n = sum(R[k]["ssd"]["n"] for k in ks)
        line.append(f"{sum(R[k]['ssd']['disk_mb'] for k in ks) / n:6.1f} MB {sum(R[k]['ssd']['reads'] for k in ks) / n:6.1f} reads")
    print(f"  {s:9s} off {line[0]} | on {line[1]}")
print()

print("6. provenance per chunk (bt_b12 / engine build hashes)")
for v in (OFF, ON):
    pv = [{k: x for k, x in c["provenance"].items() if k.startswith(("sha/", "breakthrough", "specdecode"))} for c in chunks(v)]
    uniq = {json.dumps(p, sort_keys=True) for p in pv}
    print(f"  {v}: {len(pv)} chunks, {len(uniq)} distinct: {'; '.join(sorted(uniq))}")
